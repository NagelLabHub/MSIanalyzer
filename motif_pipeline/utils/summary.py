"""
Shared statistical helpers.  Imported by both
`stats.hierarchical` and `pipeline.driver`.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

# ───────────────────────────────────────────────────────────────────────────
def compute_summary_statistics(reads_df: pd.DataFrame):
    """Strand counts and read-quality summary per group."""
    strand = reads_df.groupby("group_id")["strand"].value_counts().unstack()

    qual_mean = reads_df.groupby("group_id")["quality"].apply(
        lambda x: np.mean([np.mean(q) for q in x])
    )
    qual_tbl = pd.DataFrame({
        "Min":    reads_df.groupby("group_id")["quality"].apply(lambda x: min(map(np.mean, x))),
        "Max":    reads_df.groupby("group_id")["quality"].apply(lambda x: max(map(np.mean, x))),
        "Mean":   qual_mean,
        "Median": reads_df.groupby("group_id")["quality"].apply(lambda x: np.median(list(map(np.mean, x))))
    })
    return strand, qual_tbl


# ───────────────────────────────────────────────────────────────────────────
def compute_motif_summary(
        reads_df: pd.DataFrame,
        outcome_cols: list[str] | None = None,
        group_mapping: dict | None = None) -> pd.DataFrame:
    """
    Tall, tidy table of counts / % / Δ%  for each outcome variable.
    The **first** group in `group_mapping` is treated as the reference.
    """
    if reads_df.empty:
        print("⚠️  No valid reads to summarize.")
        return pd.DataFrame()

    outcome_cols = outcome_cols or ["repeat_units"]
    out_frames   = []

    if group_mapping:
        group_ids = list(group_mapping)
    else:
        group_ids = sorted(reads_df["group_id"].dropna().unique())

    ref_id, comp_ids = group_ids[0], group_ids[1:]

    for col in outcome_cols:
        if col not in reads_df.columns:
            print(f"⚠️ Column '{col}' not in DataFrame – skipped")
            continue

        sub = (reads_df[["group_id", col, "id"]]
               .dropna()
               .loc[lambda d: d[col] > 0])

        # integerise if possible
        if sub[col].dropna().apply(float.is_integer).all():
            sub[col] = sub[col].astype(int)

        counts = (sub.groupby("group_id")[col]
                      .value_counts()
                      .unstack(fill_value=0)
                      .T
                      .reindex(group_ids, axis=1, fill_value=0))
        counts.index.name  = "count_dist"
        counts.columns     = [f"count_{c}" for c in counts.columns]

        totals = sub.groupby("group_id")["id"].count()
        props  = counts.copy()
        for gid in group_ids:
            c = f"count_{gid}"
            props[f"proportion_{gid}"] = counts[c] / totals.get(gid, 1e-9) * 100

        for gid in comp_ids:
            props[f"diff_{gid}_vs_{ref_id}"] = (
                props[f"proportion_{gid}"] - props[f"proportion_{ref_id}"]
            )

        df_out = props.reset_index()
        df_out["outcome"] = col
        out_frames.append(df_out)

    return pd.concat(out_frames, ignore_index=True)
