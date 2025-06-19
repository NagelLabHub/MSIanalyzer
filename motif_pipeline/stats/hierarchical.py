# ---------------------------------------------------------------------------
#                    CLUSTER-AWARE STATISTICS  ✧ 2025-05-06
# ---------------------------------------------------------------------------
from ..utils.summary import compute_motif_summary

import numpy as np
import pandas as pd
from collections import Counter
from numpy.random import default_rng, RandomState
from sklearn.utils import resample
from skbio.stats.composition import dirmult_ttest          # ≥ 0.6.3
from scipy.stats import beta
# ---------------------------------------------------------------------------
# 1. β-BINOMIAL ­– for “indel vs modal allele” questions
# ---------------------------------------------------------------------------
def _beta_binom_indel_test(wt_df, ko_df, col, indel_rule=None,
                           nsamp=50_000, rng=None):
    """
    Bayesian β-binomial for the difference in the *indel proportion*
    between KO and WT.
    """
    rng = default_rng() if rng is None else rng
    if indel_rule is None:                                  # default rule
        mode = Counter(wt_df[col]).most_common(1)[0][0]
        indel_rule = lambda v: v != mode

    wt_indel = wt_df[col].apply(indel_rule).sum()
    ko_indel = ko_df[col].apply(indel_rule).sum()

    a_wt, b_wt = 1 + wt_indel, 1 + (len(wt_df) - wt_indel)
    a_ko, b_ko = 1 + ko_indel, 1 + (len(ko_df) - ko_indel)

    diff = rng.beta(a_ko, b_ko, nsamp) - rng.beta(a_wt, b_wt, nsamp)
    ci = np.quantile(diff, [0.025, .975])
    ci = (float(ci[0]), float(ci[1]))
    Pr_gt0   = float((diff > 0).mean())
    p_two    = 2*min(Pr_gt0, 1-Pr_gt0)

    return dict(test="beta_binom_indel",
                posterior_mean=float(diff.mean()),
                CrI95=tuple(round(v, 3) for v in ci), 
                Pr_diff_gt0=Pr_gt0,
                p_two_sided=p_two)
def _beta_binom_category_test(wt_df, ko_df, col, category,
                              nsamp=50_000, rng=None):
    """
    Bayesian β-binomial difference-of-proportions test for a single category.
    Returns a two-sided p-value (Monte-Carlo).
    """
    rng = default_rng() if rng is None else rng
    wt_succ = (wt_df[col] == category).sum()
    ko_succ = (ko_df[col] == category).sum()

    a_wt, b_wt = 1 + wt_succ, 1 + len(wt_df) - wt_succ
    a_ko, b_ko = 1 + ko_succ, 1 + len(ko_df) - ko_succ

    diff = rng.beta(a_ko, b_ko, nsamp) - rng.beta(a_wt, b_wt, nsamp)
    p_two = 2 * min((diff > 0).mean(), (diff < 0).mean())
    return p_two

# ---------------------------------------------------------------------------
# 2. DIRICHLET-MULTINOMIAL t-test  – any integer-valued outcome
# ---------------------------------------------------------------------------
# NOTE 2025-05-17: _dm_ttest superseded by _zidm_test (zero-inflated hurdle)

from numpy.random import default_rng
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from skbio.stats.composition import clr
# ≈500 reads per block unless n_blocks is smaller
def _make_blocks(idx, n_blocks=10, target_block_reads=500, seed=42):
    rng = default_rng(seed)
    idx = rng.permutation(idx)
    if len(idx) // target_block_reads < n_blocks:
        n_blocks = max(2, len(idx) // target_block_reads)
    return [block for block in np.array_split(idx, n_blocks) if len(block) > 0]
def _dm_ttest(wt_df, ko_df, col,
              n_blocks=10, seed=42,
              target_block_reads=500,
              p_adjust="fdr_bh"):
    """
    Dirichlet-multinomial 't-test' using CLR-transformed block counts.
    Each block is a pseudo-replicate (~500 reads by default) so we can
    estimate the within-sample dispersion parameter.
    """
    # 0) guards -------------------------------------------------------------
    if wt_df.empty or ko_df.empty:
        return pd.DataFrame(columns=["logfold", "pval", "pval_adj"])

    cats = sorted(set(wt_df[col].dropna()) | set(ko_df[col].dropna()))
    if len(cats) < 2:
        return pd.DataFrame(columns=["logfold", "pval", "pval_adj"])

    # 1) block × category count matrix -------------------------------------
    wt_blocks = _make_blocks(wt_df.index, n_blocks,
                             target_block_reads, seed)
    ko_blocks = _make_blocks(ko_df.index, n_blocks,
                             target_block_reads, seed)

    def block_counts(df, blocks):
        return [[(df.loc[b, col] == c).sum() for c in cats] for b in blocks]

    mat = np.array(block_counts(wt_df, wt_blocks) +
                   block_counts(ko_df, ko_blocks), dtype=float)

    # discard any zero-row just to be safe
    mat = mat[(mat.sum(axis=1) > 0), :]

    grouping = np.array([0]*len(wt_blocks) + [1]*len(ko_blocks))

    # 2) CLR transform + Welch t-test --------------------------------------
    clr_mat = clr(mat + 0.5)          # 0.5 pseudocount avoids log(0)
    p_raw  = []
    lfc    = []

    for j in range(len(cats)):
        wt_vals = clr_mat[grouping == 0, j]
        ko_vals = clr_mat[grouping == 1, j]
        stat, p = ttest_ind(wt_vals, ko_vals, equal_var=False)
        p_raw.append(p)
        lfc.append(float(np.log2(ko_vals.mean() / wt_vals.mean())))
    
    # ---- multiple–testing correction ------------------------------------
    _alias = {"bh": "fdr_bh", "fdr": "fdr_bh",
              "holm": "holm", "bonf": "bonferroni",
              "bonferroni": "bonferroni"}
    method = _alias.get(p_adjust.lower(), p_adjust)   # fall back to user string
    p_adj  = multipletests(p_raw, method=method)[1]

    # harmonise column names so downstream code can use "pval"/"pval_adj"
    out = pd.DataFrame({"logfold": lfc,
                        "pval":   p_raw,
                        "pval_adj": p_adj},
                    index=cats)
    out.columns.name = None         # keep it simple
    return out
from scipy.stats import fisher_exact, norm
from statsmodels.stats.multitest import multipletests
from skbio.stats.composition import clr
from numpy.random import default_rng

def _zidm_test(wt_df, ko_df, col,
               seed=42, min_tot=10,
               p_adjust="fdr_bh"):
    """
    Two-part zero-inflated DM test:
      • Fisher (presence/absence)  +  DM Welch-t on positives
      • combine with Stouffer’s method
    """
    rng = default_rng(seed)
    cats = sorted(set(wt_df[col]) | set(ko_df[col]))

    results = {}
    for c in cats:
        wt_present = (wt_df[col] == c).sum()
        ko_present = (ko_df[col] == c).sum()
        tot = wt_present + ko_present
        if tot < min_tot:
            continue                       # too rare → skip

        # ---- Part A: presence / absence ---------------------------------
        table = [[bool(wt_present), wt_present == 0],
                 [bool(ko_present), ko_present == 0]]
        pA = fisher_exact([[wt_present, len(wt_df)-wt_present],
                           [ko_present, len(ko_df)-ko_present]])[1]
        zA = norm.isf(pA/2) * np.sign(ko_present - wt_present)

        # ---- Part B: DM test on positives -------------------------------
        wt_pos = wt_df[wt_df[col] == c]
        ko_pos = ko_df[ko_df[col] == c]
        if wt_pos.empty or ko_pos.empty:
            pB = 1.0; zB = 0.0           # no positive rows in one group
        else:
            mat  = np.array([[wt_present, len(wt_df)-wt_present],
                             [ko_present, len(ko_df)-ko_present]], float)
            clr_mat = clr(mat + 0.5)
            stat, pB = ttest_ind(clr_mat[0], clr_mat[1], equal_var=False)
            zB = norm.isf(pB/2) * np.sign(stat)

        # ---- combine ----------------------------------------------------
        z = (zA + zB) / np.sqrt(2)
        p = 2*norm.sf(abs(z))
        results[c] = p

    if not results:
        return pd.DataFrame(columns=["pval","pval_adj"])

    # multiple testing ----------------------------------------------------
    cats, p_raw = zip(*results.items())
    p_adj = multipletests(p_raw, method=p_adjust)[1]
    return pd.DataFrame({"pval": p_raw,
                         "pval_adj": p_adj},
                        index=cats)

# ---------------------------------------------------------------------------
# 3. CLUSTER BOOTSTRAP  – mean / median shift
# ---------------------------------------------------------------------------
def _cluster_boot_diff(wt_df, ko_df, col, stat=np.mean,
                       n_boot=10_000, seed=42):
    rng   = RandomState(seed)
    boot  = []
    for _ in range(n_boot):
        wt_s = resample(wt_df[col], replace=True, n_samples=len(wt_df),
                        random_state=rng)
        ko_s = resample(ko_df[col], replace=True, n_samples=len(ko_df),
                        random_state=rng)
        boot.append(stat(ko_s) - stat(wt_s))
    boot  = np.asarray(boot)
    est   = stat(ko_df[col]) - stat(wt_df[col])
    ci_vals = np.quantile(boot, [0.025, 0.975])
    ci = (float(ci_vals[0]), float(ci_vals[1])) 
    p     = float((np.abs(boot) >= abs(est)).mean())
    # --- build null distribution by random label swapping ----------------
    null = []
    for _ in range(n_boot):
        combined = np.concatenate([wt_df[col].values, ko_df[col].values])
        rng.shuffle(combined)
        wt_n = combined[:len(wt_df)]
        ko_n = combined[len(wt_df):]
        null.append(stat(ko_n) - stat(wt_n))
    null = np.asarray(null)
    p = 2 * min((null >= est).mean(), (null <= est).mean())  # two-sided

    return dict(point_estimate=float(est), CI95=tuple(round(v, 3) for v in ci), bootstrap_p=p,
                statistic=stat.__name__)

# ---------------------------------------------------------------------------
# 4. HIGH-LEVEL WRAPPER  –  **patched column-name logic**
# ---------------------------------------------------------------------------
def run_hierarchical_tests(reads_df, group_mapping,
                           outcome_cols=("repeat_units",
                                         "amplicon_len",
                                         "flank1_len", "flank2_len"),
                           n_boot=10_000,
                           nsamp_beta=50_000, indel_rule=None):

    summary_df = compute_motif_summary(reads_df,
                                       outcome_cols=list(outcome_cols),
                                       group_mapping=group_mapping)

    groups  = list(group_mapping)
    wt_id   = groups[0]
    wt_df   = reads_df[reads_df["group_id"] == wt_id]
    rng_glb = default_rng()
    other_stats = []

    for outcome in outcome_cols:
        mask_outcome = summary_df["outcome"] == outcome
        if not mask_outcome.any():
            continue

        for ko_id in groups[1:]:
            ko_df = reads_df[reads_df["group_id"] == ko_id]

            # ---- DM test (zero-inflated) ---------------------------------
            dm = _zidm_test(wt_df, ko_df, outcome,
                seed=42, min_tot=10, p_adjust="fdr_bh").reset_index()

            dm = dm.rename(columns={"index": "category"})
            p_raw  = dm.set_index("category")["pval"]
            p_adj  = dm.set_index("category")["pval_adj"]

            col_p   = f"p_DM_{ko_id}_vs_{wt_id}"
            col_pad = f"padj_DM_{ko_id}_vs_{wt_id}"

            summary_df.loc[mask_outcome, col_p]   = (
                summary_df.loc[mask_outcome, "count_dist"].map(p_raw))
            summary_df.loc[mask_outcome, col_pad] = (
                summary_df.loc[mask_outcome, "count_dist"].map(p_adj))

            # --- β-binomial for every category ------------------------------------
            bb_pvals = {cat: _beta_binom_category_test(
                            wt_df, ko_df, outcome, cat,
                            nsamp_beta, rng_glb)
                        for cat in p_raw.index}

            # ✸ guard against empty dict
            if bb_pvals:
                bb_adj = multipletests(list(bb_pvals.values()), method="fdr_bh")[1]
                bb_adj = dict(zip(p_raw.index, bb_adj))
            else:
                bb_adj = {}

            summary_df.loc[mask_outcome,
                        f"p_BB_{ko_id}_vs_{wt_id}"] = (
                summary_df.loc[mask_outcome, "count_dist"].map(bb_pvals))
            summary_df.loc[mask_outcome,
                        f"padj_BB_{ko_id}_vs_{wt_id}"] = (
                summary_df.loc[mask_outcome, "count_dist"].map(bb_adj))

           # ---- bootstrap mean shift ------------------------------------
            bs = _cluster_boot_diff(wt_df, ko_df, outcome,
                                    np.mean, n_boot,
                                    seed=int(rng_glb.integers(1e9)))
            bs.update(dict(test="boot_mean",
                           outcome=outcome,
                           KO=ko_id, WT=wt_id))
            other_stats.append(bs)

            # ---- β-binomial (repeat_units only) --------------------------
            if outcome == "repeat_units":
                bb = _beta_binom_indel_test(wt_df, ko_df, outcome,
                                            indel_rule, nsamp_beta,
                                            rng_glb)
                bb.update(dict(outcome=outcome, KO=ko_id, WT=wt_id))
                other_stats.append(bb)

    other_stats_df = pd.DataFrame(other_stats)
    return summary_df, other_stats_df