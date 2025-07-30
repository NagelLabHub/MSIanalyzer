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
    run_hierarchical_tests, compute_variant_signatures, summarize_variant_signatures
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
    # inherit global defaults from batch runner
    for k, v in cfg.pop("_global_defaults_", {}).items():
        cfg.setdefault(k, v)

    outcome_cols = cfg.pop("outcome_cols", outcome_cols) or DEFAULT_OUTCOME_COLS

    if group_mapping is None:
        group_mapping = cfg.pop("group_map", None) or cfg.pop("group_mapping", None) or {}

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

    # filtering & group labels
    reads_df = map_group_labels(
        filter_valid_reads(apply_read_support_filter(reads_df)), group_mapping
    )[0]

    out_dir = get_output_dir(marker)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    summ_df = compute_motif_summary(reads_df, outcome_cols, group_mapping=None)
    if summ_df.empty or "outcome" not in summ_df.columns:
        print("⚠️  No outcomes detected – stopping early.")
        return

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

    # save csv per outcome
    for outcome in outcome_cols:
        sub = summ_df[summ_df["outcome"] == outcome]
        if not sub.empty:
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
