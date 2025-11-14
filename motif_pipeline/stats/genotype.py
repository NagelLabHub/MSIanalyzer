from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd

@dataclass
class GenotypeCall:
    group_id: str
    n_reads: int
    n_support: int
    alleles: List[int]
    supports: List[int]
    fractions: List[float]
    model: str
    confidence: float
    status: str
    reason: str

def _peak_based_alleles(values: np.ndarray,
                        min_support: int = 5,
                        sep_min_units: int = 1,
                        min_major_frac: float = 0.60,
                        min_minor_frac: float = 0.20) -> Tuple[List[int], List[int], str, float, str]:
    if values.size == 0:
        return [], [], "no-call", 0.0, "no reads"
    vals = np.asarray(values, dtype=int)
    uniq, counts = np.unique(vals, return_counts=True)
    order = np.argsort(counts)[::-1]
    uniq, counts = uniq[order], counts[order]
    total = int(counts.sum())
    keep = counts >= min_support
    uniq, counts = uniq[keep], counts[keep]
    if uniq.size == 0:
        return [], [], "no-call", 0.0, f"<{min_support} supporting reads"

    top_frac = counts[0] / total
    if uniq.size == 1 or top_frac >= min_major_frac:
        conf = float(2 * top_frac - 1)
        return [int(uniq[0])], [int(counts[0])], "mono-allelic", conf, "peak-dominant"

    sep = abs(int(uniq[0]) - int(uniq[1]))
    f1, f2 = counts[0] / total, counts[1] / total
    if sep >= sep_min_units and f2 >= min_minor_frac:
        conf = float(min(f1, f2) / (f1 + f2))
        return [int(uniq[0]), int(uniq[1])], [int(counts[0]), int(counts[1])], "bi-allelic", conf, f"sep={sep}, f2={f2:.2f}"

    reason = []
    if sep < sep_min_units: reason.append(f"sep<{sep_min_units}")
    if f2 < min_minor_frac: reason.append(f"minor<{min_minor_frac}")
    return [int(uniq[0])], [int(counts[0])], "mono-allelic", float(2*top_frac-1), ";".join(reason) or "default"

def call_genotypes(reads_df: pd.DataFrame,
                   group_col: str = "group_id",
                   repeat_units_col: str = "repeat_units",
                   *,
                   min_support: int = 5,
                   sep_min_units: int = 1,
                   min_major_frac: float = 0.60,
                   min_minor_frac: float = 0.20) -> pd.DataFrame:
    if repeat_units_col not in reads_df.columns:
        return pd.DataFrame()
    ok = reads_df[repeat_units_col].notna() & (reads_df[repeat_units_col] > 0)
    df = reads_df.loc[ok, [group_col, repeat_units_col]]

    rows = []
    for gid, sub in df.groupby(group_col):
        units = sub[repeat_units_col].astype(int).to_numpy()
        alleles, supports, status, conf, reason = _peak_based_alleles(
            units,
            min_support=min_support,
            sep_min_units=sep_min_units,
            min_major_frac=min_major_frac,
            min_minor_frac=min_minor_frac,
        )
        n_reads = int(len(units))
        n_support = int(sum(supports))
        alleles, supports = alleles[:2], supports[:2]
        fracs = [s / n_reads if n_reads else 0.0 for s in supports]
        rows.append({
            "group_id": gid,
            "n_reads": n_reads,
            "n_support": n_support,
            "num_alleles": len(alleles),
            "allele1_units": alleles[0] if len(alleles) >= 1 else np.nan,
            "allele1_support": supports[0] if len(supports) >= 1 else 0,
            "allele1_fraction": fracs[0] if len(fracs) >= 1 else 0.0,
            "allele2_units": alleles[1] if len(alleles) >= 2 else np.nan,
            "allele2_support": supports[1] if len(supports) >= 2 else 0,
            "allele2_fraction": fracs[1] if len(fracs) >= 2 else 0.0,
            "genotype_str": ";".join([f"{a}({s})" for a, s in zip(alleles, supports)]),
            "model": "peak",
            "confidence": round(conf, 3),
            "status": status,
            "reason": reason,
        })
    return pd.DataFrame(rows).sort_values("group_id").reset_index(drop=True)
