# motif_pipeline/stats/strand.py
from __future__ import annotations
from typing import Dict, Optional, Sequence
import pandas as pd
import numpy as np

def append_strand_counts_to_summary(
    summary_df: pd.DataFrame,
    reads_df: pd.DataFrame,
    outcome_cols: Sequence[str],
    *,
    group_col: str = "group_label",
    strand_col: str = "strand",
    strand_alias: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Append per-strand counts (per group × strand) to the existing wide summary.
    - Keeps all existing columns/rows.
    - Adds new columns like "<Group>_<Strand>" aligned by the summary's X-axis column.

    Expects summary_df to contain a row subset per 'outcome' and some X-axis "bin" column.
    For repeat_units in your package, this bin column is often 'count_dist'.
    """
    if group_col not in reads_df.columns:
        if "group_id" in reads_df.columns:
            group_col = "group_id"
        else:
            return summary_df  # no safe grouping column

    if strand_col not in reads_df.columns:
        return summary_df

    out = summary_df.copy()
    s_alias = strand_alias or {"Forward": "Forward", "Reverted": "Reverted", "Reverse": "Reverse"}

    # columns we will never treat as x-axis candidates
    reserved_prefixes = ("count_", "proportion_", "diff_")

    for outcome in outcome_cols:
        mask = (out["outcome"] == outcome)
        if not mask.any():
            continue

        sub = out.loc[mask].copy()

        # ---- X-AXIS DETECTION (FIXED) ----
        # try exact 'outcome' (e.g., 'amplicon_len')
        x_candidates = []
        if outcome in sub.columns:
            x_candidates.append(outcome)
        # try '<outcome>_dist' (e.g., 'repeat_units_dist'), then common names and 'count_dist'
        for name in (f"{outcome}_dist", "count_dist", "value", "bin", "x"):
            if name in sub.columns:
                x_candidates.append(name)

        # as a last resort, pick the FIRST numeric column that is NOT:
        # - 'outcome' itself
        # - a count/proportion/diff column
        if not x_candidates:
            numeric_cols = [
                c for c in sub.columns
                if np.issubdtype(sub[c].dtype, np.number)
                and c != "outcome"
                and not any(c.startswith(p) for p in reserved_prefixes)
            ]
            if numeric_cols:
                x_candidates.append(numeric_cols[0])

        if not x_candidates:
            # nothing we can align on; skip safely
            continue

        xcol = x_candidates[0]

        # ---- BUILD PER-STRAND COUNTS ON THE SAME X-AXIS ----
        if outcome not in reads_df.columns:
            # If your outcome is derived, adapt here accordingly.
            continue

        rsub = reads_df[[group_col, strand_col, outcome]].dropna()

        # Convert the outcome value to the same "bin" used in summary.
        # For count-based outcomes (e.g., repeat_units), the summary bin is integer.
        # If your summary uses a different binning scheme, adapt this line.
        rsub = rsub.assign(**{xcol: rsub[outcome].astype(float).round().astype(int)})

        # Apply strand alias for cleaner column names
        rsub[strand_col] = rsub[strand_col].map(s_alias).fillna(rsub[strand_col])

        g = (
            rsub.groupby([xcol, group_col, strand_col], dropna=False)
                .size()
                .rename("count")
                .reset_index()
        )
        if g.empty:
            continue

        wide = (
            g.pivot_table(index=xcol, columns=[group_col, strand_col],
                          values="count", fill_value=0, aggfunc="sum")
             .sort_index(axis=1)
        )
        # flatten MultiIndex columns → "<Group>_<Strand>"
        wide.columns = [f"{str(gname)}_{str(sname)}" for gname, sname in wide.columns.to_flat_index()]
        wide = wide.reset_index()

        # Left-merge on xcol so row count and order stay identical to 'sub'
        merged = sub.merge(wide, on=xcol, how="left")

        # Fill new columns with 0 where no reads exist for that cell
        new_cols = [c for c in merged.columns if c not in sub.columns]
        if new_cols:
            merged[new_cols] = merged[new_cols].fillna(0).astype(int)

        # Write back only the columns we touched/added (preserves all others)
        out.loc[mask, merged.columns] = merged.values

    return out
