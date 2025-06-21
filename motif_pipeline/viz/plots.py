import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from matplotlib.ticker import PercentFormatter, MultipleLocator
from adjustText import adjust_text

def plot_motif_density(summary_df, group_mapping, outcome,
                       marker="", output_dir=None):
    """
    Bar-plot the per-group distribution for any outcome.
    - auto-rotates x-tick labels if crowding (>20)
    - trims empty integer bins on the left / right
    - legend centred below the x-axis
    """
    # ── subset ───────────────────────────────────────────────────────────
    df = summary_df[summary_df["outcome"] == outcome].copy()
    if df.empty:
        print(f"⚠️ No data for outcome '{outcome}'.")
        return

    group_ids   = list(group_mapping.keys())
    count_dist  = sorted(df["count_dist"].unique())          # integer x-categories
    n_groups    = len(group_ids)
    n_dist      = len(count_dist)
    bar_w       = min(0.8 / n_groups, 0.15)                 # adaptive width

    # ── figure ───────────────────────────────────────────────────────────
    plt.figure(figsize=(7, 4))
    plotted = 0

    for g_idx, gid in enumerate(group_ids):
        ccol = f"count_{gid}"
        if ccol not in df.columns:
            continue
        vals   = df.set_index("count_dist")[ccol].reindex(count_dist, fill_value=0)
        total  = vals.sum()
        if total == 0:
            continue

        percent = vals.values / total * 100
        xpos    = np.arange(n_dist) + (g_idx - n_groups/2) * bar_w + bar_w/2
        plt.bar(xpos, percent, width=bar_w,
                label=group_mapping[gid], align="center")
        plotted += 1

    if plotted == 0:
        print("⚠️ No non-zero counts – nothing to plot.")
        return

    # ── aesthetics ───────────────────────────────────────────────────────
    # 1) trim empty borders
    nz = (df[[f"count_{gid}" for gid in group_ids]].sum(axis=1) > 0).values
    if nz.any():
        first, last = np.where(nz)[0][[0, -1]]
        plt.xlim(first - 0.5, last + 0.5)

    # 2) x-tick labels
    plt.xticks(np.arange(n_dist),
               count_dist,
               rotation=45 if n_dist > 20 else 0,
               ha="right" if n_dist > 20 else "center")

    # 3) labels / title
    plt.xlabel(outcome.replace("_", " ").title())
    plt.ylabel("% Reads")
    plt.title(f"{marker}_{outcome}: group comparison")

    # 4) grid: horizontal only
    ax = plt.gca()
    ax.grid(axis="y", alpha=0.3)
    ax.grid(visible=False, axis="x")          # <── no vertical grid

    # 5) legend below plot
    if plotted:
        plt.legend(title="Group",
                   bbox_to_anchor=(0.5, -0.2),
                   loc="upper center",
                   ncol=min(plotted, 3),
                   frameon=False)

    plt.tight_layout()

    # ── optional save ────────────────────────────────────────────────────
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = f"{marker}_{outcome}_bar.svg"
        path  = os.path.join(output_dir, fname)
        plt.savefig(path, format="svg")
        print(f"✅ Saved plot → {path}")

    plt.close()

def plot_proportion_differences(summary_df, group_mapping, outcome, marker="", output_dir=None, cutoff_value=None):
    """
    Plots bar plots of proportion differences between multiple groups and the reference group
    for a specified outcome only.

    Parameters:
    - summary_df (DataFrame): Output from compute_motif_summary().
    - group_mapping (dict): Mapping of group_id to label (first key = reference).
    - outcome (str): Outcome column to subset and plot (e.g., 'repeat_units').
    - marker (str): Marker label for title/filename.
    - output_dir (str): Optional folder path to save SVG plots.
    - cutoff_value (int or float): Optional cutoff value for visual reference line.
    """

    group_ids = list(group_mapping.keys())
    reference_id = group_ids[0]
    reference_label = group_mapping[reference_id]

    df = summary_df[summary_df["outcome"] == outcome].copy()
    if df.empty:
        print(f"⚠️ No data available for outcome '{outcome}'. Skipping.")
        return

    # Identify columns to compare
    diff_cols, comparisons = [], []
    for gid in group_ids[1:]:
        diff_col = f"diff_{gid}_vs_{reference_id}"
        if diff_col in df.columns:
            diff_cols.append(diff_col)
            comparisons.append(f"{group_mapping[gid]} vs {reference_label}")

    if not diff_cols:
        print(f"⚠️ No valid diff columns found for outcome '{outcome}'.")
        return

    # Determine cutoff
    proportion_col = f"proportion_{reference_id}"
    if cutoff_value is not None:
        cutoff = cutoff_value
    elif proportion_col in df.columns:
        max_idx = df[proportion_col].idxmax()
        cutoff = df.loc[max_idx, "count_dist"]
    else:
        print(f"⚠️ Missing '{proportion_col}' for outcome '{outcome}'. Skipping.")
        return

    # Melt and prepare
    melted = df.melt(id_vars="count_dist", value_vars=diff_cols,
                     var_name="Comparison", value_name="Proportion Difference")
    for gid, label in group_mapping.items():
        melted["Comparison"] = melted["Comparison"].str.replace(gid, label)
    melted["Comparison"] = melted["Comparison"].str.replace("diff_", "").str.replace("_vs_", " vs ")

    melted["count_dist"] = pd.to_numeric(melted["count_dist"], errors="coerce")

    def assign_color(row):
        if row["count_dist"] == cutoff:
            return "#999999"
        left = row["count_dist"] < cutoff
        positive = row["Proportion Difference"] >= 0
        if left and positive: return "#145DA0"
        elif left: return "#8DB3E2"
        elif positive: return "#A03333"
        else: return "#E89A9A"

    melted["bar_color"] = melted.apply(assign_color, axis=1)

    # Plotting
    n_comparisons = melted["Comparison"].nunique()
    fig, axes = plt.subplots(n_comparisons, 1, figsize=(6, 3 * n_comparisons), constrained_layout=True)
    if n_comparisons == 1: axes = [axes]

    for ax, comp in zip(axes, melted["Comparison"].unique()):
        comp_df = melted[melted["Comparison"] == comp].sort_values("count_dist")
        xtick_labels = comp_df["count_dist"].astype(int).astype(str).tolist()
        ax.bar(range(len(comp_df)), comp_df["Proportion Difference"], color=comp_df["bar_color"], width=0.8)
        ax.set_xticks(range(len(comp_df)))
        if len(comp_df) > 20:
            ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=8)
        else:
            ax.set_xticklabels(xtick_labels, rotation=0, ha="center", fontsize=9)
        ax.axhline(0, color="gray", lw=1, linestyle="--")
        ax.set_ylabel("Δ % Reads")
        ax.set_title(f"{marker} – {outcome}: {comp}")
        ax.grid(axis="y", alpha=0.3)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{marker}_{outcome}_proportion_diff.svg")
        fig.savefig(path, format="svg")
        print(f"✅ Saved plot: {path}")

    plt.close()


def plot_variant_signature_summary(summary_df,
                                   group_mapping: dict = None,
                                   plot_pct=True,
                                   stacked=True,
                                   figsize=(10, 5),
                                   palette="tab20",
                                   legend_title="variant_sig",
                                   output_dir=None):
    """
    Plot variant signature summary as stacked or grouped bar chart.
    
    Parameters:
        summary_df     : output from summarize_variant_signatures()
        group_mapping  : dict to map group_id → label (e.g. {"BCXH4Z_3": "XPA3"})
        plot_pct       : if True, use % columns; else use absolute counts
        stacked        : if True, use stacked bar chart
        figsize        : (width, height) of figure
        palette        : seaborn/matplotlib palette
        legend_title   : title for the legend
    """

    # Choose relevant columns
    cols = [c for c in summary_df.columns if c.endswith(" (%)")] if plot_pct else \
           [c for c in summary_df.columns if c not in ["total_reads"] and not c.endswith(" (%)")]

    # Melt into long format
    data = summary_df[cols].copy()
    data.index.name = "group_id"
    data = data.reset_index().melt(id_vars="group_id", var_name="signature", value_name="value")

    # Clean signature names
    if plot_pct:
        data["signature"] = data["signature"].str.replace(r" \(%\)", "", regex=True)

    # Apply group_mapping
    if group_mapping:
        data["group_label"] = data["group_id"].map(group_mapping)
    else:
        data["group_label"] = data["group_id"]

    # Plot
    plt.figure(figsize=figsize)

    if stacked:
        pivoted = data.pivot(index="group_label", columns="signature", values="value")
        pivoted.plot(kind="bar", stacked=True, figsize=figsize, colormap=palette)
        plt.ylabel("% of reads" if plot_pct else "Read count")
        plt.title("Variant signature spectrum per sample (stacked)")
        plt.xticks(rotation=0)
        plt.legend(title=legend_title, bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.tight_layout()
    else:
        sns.barplot(data=data,
                    x="group_label", y="value", hue="signature", palette=palette)
        plt.ylabel("% of reads" if plot_pct else "Read count")
        plt.title("Variant signature spectrum per sample")
        plt.xticks(rotation=0)
        plt.legend(title=legend_title, bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, f"variant_signature_summary.png"), bbox_inches="tight")
        print("✅ Variant signature summary saved to:", os.path.join(output_dir, f"variant_signature_summary.png"))

    plt.close()

def plot_confident_interruptions(df,
                                 group_mapping: dict = None,
                                 min_support=3,
                                 ymax=0.03,
                                 annotate=True,
                                 highlight_top=2,
                                 figsize_per_panel=(6, 3),
                                 output_dir=None):
    """
    Plot confident interruptions along the repeat region.

    Parameters:
        df               : DataFrame with 'interruptions' and QC-passed reads
        group_mapping    : dict mapping group_id → readable sample label
        min_support      : min reads supporting interruption at unit_idx
        ymax             : max y-axis value
        annotate         : label seq (top/high-freq only)
        highlight_top    : number of interruptions to bold per group
        figsize_per_panel: per-panel width, height
    """

    df = df.copy()

    # 1. Explode interruptions
    intr = (df[df["read_class"] == "repeat_interruption"]
            .explode("interruptions")
            .dropna(subset=["interruptions"]))
    intr = pd.concat([
        intr[["id", "group_id", "group_label"]],
        intr["interruptions"].apply(pd.Series)[["unit_idx", "type", "edit", "seq"]]
    ], axis=1)

    # 2. Filter by support
    support = intr.groupby(["group_id", "unit_idx"]).size().reset_index(name="support")
    valid_idx = support[support["support"] >= min_support][["group_id", "unit_idx"]]
    intr = intr.merge(valid_idx, on=["group_id", "unit_idx"], how="inner")

    # 3. Frequency table
    n_reads = df.groupby("group_id").size().reset_index(name="n_total")
    count_tbl = (intr.groupby(["group_id", "group_label", "unit_idx", "type", "seq"])
                      .agg(n_intr=("id", "nunique"))
                      .reset_index())
    freq = count_tbl.merge(n_reads, on="group_id")
    freq["freq"] = freq["n_intr"] / freq["n_total"]

    # 4. Map group labels
    if group_mapping:
        freq["group_label"] = freq["group_id"].map(group_mapping)

    # 5. Setup layout
    groups = freq["group_label"].unique()
    n_rows = len(groups)
    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(figsize_per_panel[0], figsize_per_panel[1]*n_rows),
                             squeeze=False)
    axes = axes.flatten()

    palette = {'ins': '#2ca02c', 'del': '#d62728', 'sub': '#1f77b4'}
    markers = {'ins': 'o', 'del': 'v', 'sub': 's'}
    all_handles, all_labels = None, None

    for i, group in enumerate(groups):
        ax = axes[i]
        data = freq[freq["group_label"] == group]

        # Plot (no size by edit)
        plot = sns.scatterplot(data=data,
                               x="unit_idx", y="freq",
                               hue="type", style="type",
                               palette=palette, markers=markers,
                               ax=ax, legend=True)

        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_xlabel("Repeat unit index")
        ax.set_ylabel("% reads with interruption")
        ax.set_title(group)

        # Capture legend once
        if i == 0:
            all_handles, all_labels = ax.get_legend_handles_labels()
        ax.legend_.remove()

        # Annotate
        if annotate:
            top_rows = data.sort_values("freq", ascending=False).head(highlight_top)
            top_seqs = top_rows["seq"].tolist()
            texts = []
            for _, row in data.iterrows():
                if isinstance(row["seq"], str) and (row["freq"] >= 0.002 or row["seq"] in top_seqs):
                    txt = ax.text(row["unit_idx"], row["freq"] + 0.001,
                                  row["seq"], fontsize=7, ha="center", va="bottom",
                                  fontweight="bold" if row["seq"] in top_seqs else "normal")
                    texts.append(txt)
            try:
                adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.3))
            except:
                pass

    # Shared legend
    fig.legend(all_handles, all_labels, title="Type",
               bbox_to_anchor=(0.98, 0.94), loc="upper left",)
    fig.suptitle("Interruptions inside repeat region", y=0.995)
    plt.tight_layout(rect=[0, 0, 0.93, 0.96])
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"repeat_interruptions.png"), bbox_inches="tight")
        print("✅ Confident interruptions saved to:", os.path.join(output_dir, f"repeat_interruptions.png"))
    plt.close()
