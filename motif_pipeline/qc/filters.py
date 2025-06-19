from __future__ import annotations
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
def apply_read_support_filter(
        reads_df,
        min_sup_repeat  = 5,
        min_sup_flank   = 10,
        flank_max       = None,
        pair_flank_rule = True,
        per_group       = True,
        verbose         = True):

    import numpy as np

    df = reads_df.copy()
    if "valid" not in df.columns:
        df["valid"] = True
    pre_valid = df["valid"]

    # ---------- grouping keys ---------------------------------------------
    keys = []
    if per_group and "group_id" in df.columns: keys.append("group_id")
    if "marker" in df.columns:                 keys.append("marker")

    def _support(colnames, label, min_sup):
        grp_cols = keys + colnames
        sup = df.groupby(grp_cols).size().rename(f"sup_{label}")
        return df.join(sup, on=grp_cols)[f"sup_{label}"].fillna(0) >= min_sup

    # ======================================================================
    # STAGE A: Flank support check (pair or side-specific)                 |
    # ======================================================================
    if pair_flank_rule:
        flank_support_pass = _support(["flank1_len", "flank2_len"], "flank_pair", min_sup_flank)
    else:
        flank1_pass = _support(["flank1_len"], "flank1", min_sup_flank)
        flank2_pass = _support(["flank2_len"], "flank2", min_sup_flank)
        flank_support_pass = flank1_pass & flank2_pass

    if flank_max is not None:
        flank_ceiling_pass = (
            (df["flank1_len"] <= flank_max) & (df["flank2_len"] <= flank_max)
        )
    else:
        flank_ceiling_pass = pd.Series(True, index=df.index)

    flank_pass = flank_support_pass & flank_ceiling_pass

    # ======================================================================
    # STAGE B: Repeat support — only for reads that pass flank + pre_valid |
    # ======================================================================
    df_flank_passed = df[flank_pass & pre_valid]
    rep_grp = (
        df_flank_passed.groupby(keys + ["repeat_units"])
        .size()
        .rename("sup_repeat")
    )
    repeat_pass = (
        df.join(rep_grp, on=keys + ["repeat_units"])["sup_repeat"]
        .fillna(0) >= min_sup_repeat
    )

    # ======================================================================
    # Combine results and diagnostics                                      |
    # ======================================================================
    post_valid = pre_valid & flank_pass & repeat_pass

    df["flank_len_valid"] = flank_pass
    df["repeat_valid"] = repeat_pass
    df["amplicon_len_valid"] = flank_pass & repeat_pass
    df["valid"] = post_valid

    if verbose:
        print(f"[Pre]     invalid reads entering filter   : {(~pre_valid).sum()}")
        print(f"[Flank]   newly invalid after flank check : {(pre_valid & ~flank_pass).sum()}")
        print(f"[Repeat]  newly invalid after repeat check: {(pre_valid & flank_pass & ~repeat_pass).sum()}")
        if flank_max is not None:
            print(f"[FlankMax] new invalid due to flank > {flank_max}     : {(pre_valid & flank_support_pass & ~flank_ceiling_pass).sum()}")
        print(f"[Post]    total valid reads              : {df['valid'].sum()} / {len(df)}")

    return df

def filter_valid_reads(reads_df,
                       strands: tuple[str, ...] = ("Forward", "Reverted"),
                       require_valid: bool = True,
                       drop_columns: bool = True) -> pd.DataFrame:
    """
    Filters reads based on strand and validity, and removes unused columns.
    """
    df = reads_df.copy()
    df = df[df["strand"].isin(strands)]

    if require_valid and "valid" in df.columns:
        df = df[df["valid"]]

    if drop_columns:
        drop_cols = ["sequence", "quality", "valid",
                     "seq_flank1", "seq_repeat", "seq_flank2"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df

def map_group_labels(reads_df, user_mapping):
    """
    Maps user-defined group IDs to custom labels and ensures the order of `group_id` follows the input mapping.
    """
    # Ensure a copy of the DataFrame to avoid SettingWithCopyWarning
    reads_df = reads_df.copy()

    # Map group labels using .loc[] to avoid warnings
    reads_df.loc[:, "group_label"] = reads_df["group_id"].map(user_mapping)

    # Ensure group_id follows the order from user_mapping
    reads_df.loc[:, "group_id"] = pd.Categorical(
        reads_df["group_id"], categories=user_mapping.keys(), ordered=True
    )

    # Ensure group_label follows the user-defined order
    reads_df.loc[:, "group_label"] = pd.Categorical(
        reads_df["group_label"], categories=user_mapping.values(), ordered=True
    )

    # Sort DataFrame by `group_id`
    reads_df = reads_df.sort_values("group_id")

    return reads_df, user_mapping
