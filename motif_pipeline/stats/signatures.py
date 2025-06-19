from __future__ import annotations
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from ..stats.hierarchical import _beta_binom_category_test  # relative path
from ..stats.hierarchical import _zidm_test  # relative path

def compute_variant_signatures(df: pd.DataFrame,
                                repeat_ref: int = 19,
                                error_cutoff: float = 0.01,
                                interruption_col: str = "interruptions",
                                group_col: str = "group_label",
                                read_class_col: str = "read_class",
                                repeat_units_col: str = "repeat_units"
                               ) -> pd.DataFrame:
    """
    Add a robust 'variant_sig' column to the input DataFrame based on:
      - deviation from repeat_ref length
      - high-confidence interruptions per group

    Parameters:
        df               : filtered reads DataFrame (must include 'interruptions')
        repeat_ref       : reference AAAG copy number (default = 19)
        error_cutoff     : minimum group-specific frequency to keep an interruption
        interruption_col : column name with interruptions (list of dicts)
        group_col        : sample group column
        read_class_col   : 'pure_repeat' or 'repeat_interruption'
        repeat_units_col : number of AAAG units in the read

    Returns:
        Updated DataFrame with a new column:
            df["variant_sig"]
    """

    df = df.copy()
    grp_n_reads = df.groupby(group_col).size()

    # Explode interruptions
    intr_tbl = (
        df[df[read_class_col] == "repeat_interruption"]
        .explode(interruption_col)
        .dropna(subset=[interruption_col])
    )
    intr_tbl = pd.concat(
        [intr_tbl[[group_col]],
         intr_tbl[interruption_col].apply(pd.Series)[["unit_idx", "type"]]],
        axis=1
    )

    # Group-wise interruption frequency
    freq_lookup = (
        intr_tbl.groupby([group_col, "unit_idx", "type"])
                .size()
                .div(grp_n_reads, level=0)
                .to_dict()
    )

    # Signature builder
    def build_variant_sig(row):
        delta = int(row[repeat_units_col]) - repeat_ref
        rc = row[read_class_col]
        if delta == 0 and rc == "pure_repeat":
            len_tag = f"Perfect{repeat_ref}"
        elif delta == 0:
            len_tag = f"Int{repeat_ref}"
        elif delta < 0:
            len_tag = f"Con{delta}"
        else:
            len_tag = f"Exp+{delta}"

        intr_tags = []
        for d in row[interruption_col]:
            key = (row[group_col], d["unit_idx"], d["type"])
            if freq_lookup.get(key, 0) >= error_cutoff:
                intr_tags.append(f'{d["type"]}@{d["unit_idx"]}')
        intr_tags = ",".join(sorted(intr_tags))

        return len_tag if not intr_tags else f"{len_tag}|{intr_tags}"

    df["variant_sig"] = df.apply(build_variant_sig, axis=1)

    # QC message
    n_intr_total = (df[read_class_col] == "repeat_interruption").sum()
    n_intr_retained = df["variant_sig"].str.contains("|", regex=False).sum()
    n_dropped = n_intr_total - n_intr_retained
    print(f"🧪 {n_dropped:,} reads had ≥1 interruption, but all were below the {error_cutoff:.1%} threshold — classified by length only.")

    return df


def summarize_variant_signatures(df,
                                    group_col="group_id",
                                    sig_col="variant_sig",
                                    min_pct=1.0,
                                    strict=True,
                                    add_dm=True,
                                    dm_blocks=10,
                                    seed=42):

    from numpy.random import default_rng
    rng = default_rng(seed)
    df  = df.copy()

    # ---------- 1. simplify ----------------------------------------------
    def simplify(sig):
        base = sig.split("|")[0]
        if not strict:
            return sig
        if base.startswith("Con-"):
            try:
                n = int(base.replace("Con-", ""))
                return f"Con-{n}" if n <= 1 else ("Con-2-3" if n <= 3 else "Con-4+")
            except ValueError:
                return base
        if base.startswith("Exp+"):
            try:
                n = int(base.replace("Exp+", ""))
                return f"Exp+{n}" if n == 1 else ("Exp+2-3" if n <= 3 else "Exp+4+")
            except ValueError:
                return base
        return base

    df["sig_initial"] = df[sig_col].apply(simplify)

    # ---------- 2. per-group counts (% for rare-filter) -------------------
    counts = (df.groupby([group_col, "sig_initial"])
                .size()
                .rename("n")
                .reset_index())

    totals = counts.groupby(group_col)["n"].sum().rename("total").reset_index()
    counts = counts.merge(totals, on=group_col)
    counts["pct"] = 100 * counts["n"] / counts["total"]

    if strict:
        counts["sig_final"] = counts["sig_initial"]
    else:
        def collapse(row):
            if row["pct"] >= min_pct:
                return row["sig_initial"]
            return row["sig_initial"].split("|")[0]
        counts["sig_final"] = counts.apply(collapse, axis=1)

    # ▶  ---- NEW  (propagate back to per-read df) -------------------------
    df = df.merge(counts[[group_col, "sig_initial", "sig_final"]],
                  on=[group_col, "sig_initial"],
                  how="left")
    # ---------------------------------------------------------------------

    # ---------- 3. wide summary table ------------------------------------
    summary = (counts.groupby([group_col, "sig_final"])["n"]
                      .sum()
                      .unstack(fill_value=0)
                      .astype(int))

    summary["total_reads"] = summary.sum(axis=1)
    pct = (summary.drop(columns="total_reads")
                 .div(summary["total_reads"], axis=0)
                 .mul(100).round(1)
                 .add_suffix(" (%)"))
    summary = pd.concat([summary, pct], axis=1)

    # ---------- 4. DM + β-binomial per signature -------------------------
    if not add_dm or summary.shape[0] < 2:
        return summary, pd.DataFrame()

    groups = summary.index.tolist()
    wt_id  = groups[0]
    wt_df  = df[df[group_col] == wt_id]
    rng    = default_rng(seed)

    dm_frames  = []
    bb_frames  = []

    for ko_id in groups[1:]:
        ko_df = df[df[group_col] == ko_id]

        # ---- Dirichlet–multinomial -------------------------------------
        dm = _zidm_test(wt_df, ko_df, col="sig_final",
                seed=seed, min_tot=10, p_adjust="fdr_bh")

        if not dm.empty:
            dm = dm.rename(columns={"pval":"p_raw",
                                    "pval_adj":"p_adj"})
            dm.columns = pd.MultiIndex.from_product(
                [[f"{ko_id} vs {wt_id}"], dm.columns],
                names=["comparison", "stat"])
            dm_frames.append(dm)

        # ---- β-binomial per signature -----------------------------------
        cats     = summary.columns[~summary.columns.str.endswith(" (%)")]
        bb_p     = [ _beta_binom_category_test(wt_df, ko_df,
                                               "sig_final", sig,
                                               nsamp=50_000, rng=rng)
                     for sig in cats ]
        bb_adj   = multipletests(bb_p, method="fdr_bh")[1]
        bb_df    = pd.DataFrame({"p_raw": bb_p,
                                 "p_adj": bb_adj},
                                index=cats)
        bb_df.columns = pd.MultiIndex.from_product(
            [[f"{ko_id} vs {wt_id} – ββ"], bb_df.columns],
            names=["comparison", "stat"])
        bb_frames.append(bb_df)

    dm_stats = (pd.concat(dm_frames, axis=1)
                if dm_frames else pd.DataFrame())
    bb_stats = (pd.concat(bb_frames, axis=1)
                if bb_frames else pd.DataFrame())

    stats = pd.concat([dm_stats, bb_stats], axis=1).sort_index(axis=1)
    sig_order = [c for c in summary.columns
                 if not c.endswith(" (%)") and c != "total_reads"]
    stats = stats.reindex(sig_order)

    return summary, stats
