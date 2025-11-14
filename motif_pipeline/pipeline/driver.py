"""
High-level drivers used by CLI & notebooks.
"""
from __future__ import annotations
import os, glob, json
from pathlib import Path
from typing import Optional

import pandas as pd

from ..io import process_fastq_parallel, get_output_dir
from ..qc import apply_read_support_filter, filter_valid_reads, map_group_labels
from ..utils.summary import compute_motif_summary, compute_summary_statistics
from ..stats import (
    run_hierarchical_tests, compute_variant_signatures, summarize_variant_signatures, call_genotypes, append_strand_counts_to_summary
)
from ..viz import (
    plot_motif_density, plot_proportion_differences,
    plot_confident_interruptions, plot_variant_signature_summary,
)

DEFAULT_OUTCOME_COLS = ["repeat_units"]

# ──────────────────────────────────────────────────────────────────────────────
# FASTQ resolver
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_fastqs(marker: str, cfg: dict) -> list[str]:
    if "fastq_files" in cfg:
        return cfg["fastq_files"]

    pattern = cfg.get("fastq_glob", f"./fastq/{marker}")
    if os.path.isdir(pattern):
        pattern = os.path.join(pattern, "*.fastq*")
    if "*" not in pattern:
        pattern = os.path.join(pattern, "*.fastq*")

    return sorted(glob.glob(pattern))

# ──────────────────────────────────────────────────────────────────────────────
# Single marker
# ──────────────────────────────────────────────────────────────────────────────
def run_marker_pipeline(
    marker: str,
    cfg: dict,
    group_mapping: Optional[dict] = None,
    *,
    do_hierarchical_tests: bool = True,
    do_variant_summary: bool = True,
    match_threshold: float = 0.85,
    outcome_cols: Optional[list[str]] = None,
    num_workers: int = 1,
    batch_size: int = 500,
):
    # --- inherit global defaults (if a batch runner injected them) ---
    # Anything under cfg["_global_defaults_"] acts as global and is overridden by per-marker keys.
    global_defaults = cfg.pop("_global_defaults_", {}) or {}
    outcome_cols = cfg.pop("outcome_cols", outcome_cols) or DEFAULT_OUTCOME_COLS

    if group_mapping is None:
        group_mapping = cfg.pop("group_map", None) or cfg.pop("group_mapping", None) or {}

    # --- input resolution ---
    fastq_files = _resolve_fastqs(marker, cfg)
    if not fastq_files:
        raise FileNotFoundError(f"No FASTQs found for marker '{marker}'")

    seq1, seq2, motif = cfg["seq1"], cfg["seq2"], cfg["motif"]
    print(f"🔬 {marker}: {len(fastq_files)} FASTQ(s)")

    reads_df = pd.DataFrame(
        process_fastq_parallel(
            fastq_files, seq1, seq2, motif,
            match_threshold, num_workers=num_workers, batch_size=batch_size
        )
    )

    # QC readouts
    print(*compute_summary_statistics(reads_df), sep="\n")

    # --- filtering (merge global → marker) ---
    global_filters = (global_defaults.get("filters") or {})
    marker_filters = (cfg.get("filters") or {})
    filt_cfg = {**global_filters, **marker_filters}

    # If you have module defaults in the filter function, we can just pass what exists.
    # If you prefer explicit fallbacks, change 3/5 to your true defaults.
    if filt_cfg:
        reads_df = apply_read_support_filter(
            reads_df,
            min_sup_repeat=int(filt_cfg.get("min_sup_repeat", 3)),
            min_sup_flank=int(filt_cfg.get("min_sup_flank", 5)),
        )
    else:
        reads_df = apply_read_support_filter(reads_df)

    # --- ensure valid reads + group labels BEFORE genotyping ---
    reads_df = filter_valid_reads(reads_df)
    # map_group_labels typically returns (df, mapping or something); use [0] if that's your API
    reads_df = map_group_labels(reads_df, group_mapping)[0]

    out_dir = get_output_dir(marker)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # --- GENOTYPING (default ON) ---
    # Merge global → marker configs; default enabled unless explicitly disabled.
    global_geno = (global_defaults.get("genotype") or {})
    marker_geno = (cfg.get("genotype") or {})
    geno_cfg = {**global_geno, **marker_geno}
    geno_enabled = bool(geno_cfg.get("enabled", True))

    geno_df = pd.DataFrame()
    if geno_enabled:
        try:
            geno_df = call_genotypes(
                reads_df,
                group_col="group_id",
                repeat_units_col="repeat_units",
                min_support    = int(geno_cfg.get("min_support", 5)),
                sep_min_units  = int(geno_cfg.get("sep_min_units", 1)),
                min_major_frac = float(geno_cfg.get("min_major_frac", 0.60)),
                min_minor_frac = float(geno_cfg.get("min_minor_frac", 0.20)),
            )

            if not geno_df.empty:
                # Add bp columns if motif is present
                if motif:
                    motif_len = len(motif)
                    for col in ("allele1_units", "allele2_units"):
                        bpcol = col.replace("_units", "_bp")
                        geno_df[bpcol] = (geno_df[col] * motif_len).where(geno_df[col].notna())

                # Straglr-like one-liners
                for r in geno_df.itertuples(index=False):
                    print(
                        f"🧬 {marker} :: {r.group_id}\t{r.genotype_str}  "
                        f"[n={r.n_reads}, alleles={r.num_alleles}, conf={r.confidence:.2f}, {r.status}]"
                    )

                # Write compact TSV
                geno_out = f"{out_dir}/genotype_{marker}.tsv"
                keep = [
                    "group_id","n_reads","n_support","num_alleles",
                    "allele1_units","allele1_bp","allele1_support","allele1_fraction",
                    "allele2_units","allele2_bp","allele2_support","allele2_fraction",
                    "genotype_str","model","confidence","status","reason",
                ]
                for k in keep:
                    if k not in geno_df.columns:
                        geno_df[k] = pd.NA
                geno_df[keep].to_csv(geno_out, sep="\t", index=False)
                print(f"💾 Genotypes written → {geno_out}")
        except Exception as e:
            print(f"⚠️  Genotyping skipped for {marker}: {e}")

    # --- summaries & plots ---
    # NOTE: pass the real group_mapping, not None
    summ_df = compute_motif_summary(reads_df, outcome_cols, group_mapping=group_mapping)
    if summ_df.empty or "outcome" not in summ_df.columns:
        print("⚠️  No outcomes detected – stopping early.")
        return

    # -- OPTIONAL: per-strand counts appended to summary tables --
    # Default ON unless explicitly disabled in config.
    summary_cfg = {**(global_defaults.get("summary") or {}), **(cfg.get("summary") or {})}
    strand_counts_enabled = bool(summary_cfg.get("strand_counts", True))

    if strand_counts_enabled:
        try:
            summ_df = append_strand_counts_to_summary(
                summ_df,
                reads_df,
                outcome_cols,
                group_col="group_id",  
                strand_col="strand",
                strand_alias=summary_cfg.get(
                    "strand_alias",
                    {"Forward": "Forward", "Reverted": "Reverted", "Reverse": "Reverse"}
                ),
            )
            print("ℹ️  Strand counts appended to summaries.")
        except Exception as e:
            print(f"⚠️  Skipped strand-count augmentation: {e}")

    for outcome in outcome_cols:
        sub = summ_df[summ_df["outcome"] == outcome]
        if sub.empty:
            continue
        plot_motif_density(sub, group_mapping, outcome, marker, out_dir)

    # hierarchical tests
    if do_hierarchical_tests and len(group_mapping) > 1:
        summ_df, other = run_hierarchical_tests(reads_df, group_mapping, outcome_cols)
        for outcome in outcome_cols:
            sub = summ_df[summ_df["outcome"] == outcome]
            if not sub.empty:
                plot_proportion_differences(sub, group_mapping, outcome, marker, out_dir)
        other.to_csv(f"{out_dir}/hierarchical_tests_{marker}.csv", index=False)

    # save csv per outcome (+ attach genotype strings to repeat_units summary)
    for outcome in outcome_cols:
        sub = summ_df[summ_df["outcome"] == outcome].copy()
        if sub.empty:
            continue
        if outcome == "repeat_units" and not geno_df.empty:
            for gid, gstr in geno_df[["group_id", "genotype_str"]].itertuples(index=False):
                sub[f"genotype_{gid}"] = gstr
        sub.to_csv(f"{out_dir}/summary_{marker}_{outcome}.csv", index=False)

    # save reads table (Feather → Parquet fallback)
    feather = f"{out_dir}/reads_{marker}.feather"
    try:
        reads_df.drop(columns=["sequence", "quality"], errors="ignore").to_feather(feather)
    except ImportError:
        pq = feather.replace(".feather", ".parquet")
        reads_df.drop(columns=["sequence", "quality"], errors="ignore").to_parquet(pq)
        print("⚠️  pyarrow missing – wrote Parquet:", pq)

    # variant-signature plots
    if do_variant_summary:
        reads_df = compute_variant_signatures(reads_df)
        plot_confident_interruptions(reads_df, group_mapping, output_dir=out_dir)
        sig_sum, dm = summarize_variant_signatures(reads_df, "group_id", strict=False)
        plot_variant_signature_summary(sig_sum, group_mapping, output_dir=out_dir)
        sig_sum.to_csv(f"{out_dir}/variant_signatures_{marker}.csv")
        dm.to_csv(f"{out_dir}/variant_signatures_dm_stats_{marker}.csv")
    else:
        print("ℹ️  Skipping variant signature summary.")
    print(f"✅ Finished {marker} → {out_dir}")
    return summ_df


# ──────────────────────────────────────────────────────────────────────────────
# Batch runner
# ──────────────────────────────────────────────────────────────────────────────
def run_batch_pipeline(
    manifest: Path,
    *,
    skip_tests: bool = False,
    skip_variant_summary: bool = False,
    global_threads: int = 1,
):
    data = json.loads(Path(manifest).read_text())
    if "markers" not in data:          # accept wrapped style
        data = {"markers": data}

    globals_ = {k: v for k, v in data.items() if k != "markers"}

    for marker, cfg in data["markers"].items():
        cfg = {"_global_defaults_": globals_, **cfg}
        gmap = cfg.pop("group_mapping", None) or cfg.pop("group_map", None)
        run_marker_pipeline(
            marker,
            cfg,
            gmap,
            do_hierarchical_tests=not skip_tests,
            do_variant_summary=not skip_variant_summary,
            match_threshold=globals_.get("match_threshold", 0.85),
            num_workers=global_threads,
        )
