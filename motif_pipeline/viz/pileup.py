"""
Enhanced pile-up visualiser (depth + indel + mismatch).

Four stacked tracks (identical to notebook version):
  1) indel-frequency fill + mismatch-frequency line
  2) reference-base colour bar
  3) insertion (+) & deletion (–) frequency bars
  4) scatter of indel lengths (size √n, colour by sign)

If *region* is None the whole BAM contig is plotted.
"""

from __future__ import annotations
import re, pathlib, shutil, subprocess, collections
from typing import Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import pysam
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter, MultipleLocator

# ──────────────────────────────────────────────────────────────────────────────
def _resolve_contig(bam: pysam.AlignmentFile, chrom: Optional[str]) -> str:
    """Return the contig present in *bam* that matches *chrom* (or first)."""
    refs = list(bam.references)
    if chrom is None:
        return refs[0]
    if chrom in refs:
        return chrom
    alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    if alt in refs:
        return alt
    matches = [r for r in refs if chrom in r]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f'contig "{chrom}" not in BAM (first refs: {refs[:5]} …)')


def _to_local_coords(contig: str, start: int, end: int) -> Tuple[int, int]:
    """If contig encodes slice (chr:start-end) convert genome coords → local."""
    m = re.match(r".*:(\d+)-(\d+)$", contig)
    if not m:                              # whole chromosome
        return start, end
    g_start = int(m.group(1))
    return max(1, start - g_start + 1), end - g_start + 1


# ──────────────────────────────────────────────────────────────────────────────
def plot_indel_profile(
        bam_path   : str | pathlib.Path,
        ref_fa     : str | pathlib.Path,
        chrom      : str | None = None,
        start      : int | None = None,
        end        : int | None = None,
        *,
        max_len    : int = 20,
        tick_every : int = 10,
        save_path  : str | pathlib.Path | None = None,
):
    """Render the four-track pile-up figure and optionally save to *save_path*."""
    bam_path = pathlib.Path(bam_path)
    bam = pysam.AlignmentFile(str(bam_path), "rb")
    fa  = pysam.FastaFile(str(ref_fa))

    contig = _resolve_contig(bam, chrom)
    ref_len = fa.get_reference_length(contig)

    # whole contig if no region supplied
    if start is None or end is None:
        start, end = 1, ref_len
    else:
        start, end = _to_local_coords(contig, int(start), int(end))

    # cache reference sequence for fast look-ups
    ref_seq = fa.fetch(contig, start - 1, end).upper()
    ref2color = {"A": "#b5ea8c", "C": "#377eb8", "G": "#ff7f00", "T": "#eac4d5"}
    ref_colors = [ref2color.get(b, "lightgrey") for b in ref_seq]

    pos_len_counter: dict[int, Counter[int]] = defaultdict(Counter)
    recs = []

    for col in bam.pileup(
            contig, start - 1, end,
            truncate=True, stepper="all",
            fastafile=fa, min_baseq=0, ignore_overlaps=False, max_depth=10_000):
        pos1 = col.reference_pos + 1       # 1-based
        cov  = col.nsegments
        ins = dels = mm = 0

        # reference base via cached string
        ref_idx = pos1 - start
        ref_base = ref_seq[ref_idx] if 0 <= ref_idx < len(ref_seq) else \
                   fa.fetch(contig, pos1 - 1, pos1).upper()

        for pr in col.pileups:
            # scatter support
            if pr.indel != 0:
                pos_len_counter[pos1][pr.indel] += 1
            # event tallies
            if pr.indel > 0:
                ins += 1
            if pr.is_del:
                dels += 1
            # mismatch tally
            if (not pr.is_del) and pr.indel == 0 and pr.query_position is not None:
                q_base = pr.alignment.query_sequence[pr.query_position].upper()
                if q_base != ref_base:
                    mm += 1

        if cov:
            recs.append((pos1, cov, ins / cov, dels / cov, mm / cov))

    df = pd.DataFrame(recs, columns=["pos", "cov", "ins_f", "del_f", "mm_f"])
    if df.empty:
        raise ValueError("no coverage in selected window")
    df["indel_f"] = df.ins_f + df.del_f

    # ────────────────────────── figure scaffold
    fig, axs = plt.subplots(
        3, 1, figsize=(18, 6),
        gridspec_kw={"height_ratios": [2, .1, 1]}, sharex=True)

    # track 1 ─ indel + mismatch overlay
    axs[0].fill_between(df.pos, df.indel_f, color="steelblue", alpha=.30)
    axs[0].plot(df.pos, df.indel_f, lw=.7, color="black", label="indel")

    axs[0].fill_between(df.pos, df.mm_f, color="goldenrod", alpha=.25)
    axs[0].plot(df.pos, df.mm_f, lw=.7, color="goldenrod", label="mismatch")

    axs[0].set_ylabel("Event frequency")
    axs[0].yaxis.set_major_formatter(PercentFormatter(1))
    axs[0].set_title(f"Per-position indel / mismatch rate – {bam_path.stem}")
    axs[0].legend(frameon=False, ncol=2, loc="upper right")

    # track 2 ─ reference-base colour bar
    for x, colc in zip(df.pos, ref_colors):
        axs[1].add_patch(Rectangle((x, 0), 1, 1, facecolor=colc, lw=0))
    axs[1].set_xlim(start, end)
    axs[1].set_yticks([])
    axs[1].set_ylabel("Ref\nbase")

    # track 3 ─ insertion (+) & deletion (–) bars
    axs[2].bar(df.pos,  df.ins_f,  color="#4daf4a", width=1, label="ins")
    axs[2].bar(df.pos, -df.del_f, color="#e41a1c", width=1, label="del")

    ymax = max(df.ins_f.max(), df.del_f.max()) * 1.2
    axs[2].set_ylim(-ymax, ymax)
    axs[2].yaxis.set_major_formatter(PercentFormatter(1))
    axs[2].set_ylabel("% ins / del")
    axs[2].axhline(0, color="grey", lw=.4)
    axs[2].legend(frameon=False, ncol=2, loc="upper right")

    # scatter of indel lengths (twin-y)
    ax_len = axs[2].twinx()
    for p, counter in pos_len_counter.items():
        for ln, n in counter.items():
            if abs(ln) > max_len:
                continue
            ax_len.scatter(
                p, ln, s=np.sqrt(n)*15,
                color="#b8e0d4" if ln > 0 else "#ffb3c6",
                alpha=.45, lw=0)
    ax_len.set_ylim(-max_len, max_len)
    ax_len.set_ylabel("Indel length (bp)")
    ax_len.yaxis.set_major_locator(MultipleLocator(4))
    ax_len.axhline(0, color="grey", lw=.4)

    # x-axis cosmetics
    axs[2].set_xlabel(f"{contig}:{start}-{end}")
    axs[2].xaxis.set_major_locator(MultipleLocator(tick_every))
    axs[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ saved pile-up figure → {save_path}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Mapping helper (unchanged from previous version)
# ──────────────────────────────────────────────────────────────────────────────
def pileup_run_pipeline(
    fastq_input : str | pathlib.Path,
    ref_fasta : str | pathlib.Path,
    out_dir   : str | pathlib.Path = "pileup_tmp",
    *,
    threads   : int = 1,
    overwrite : bool = False,
):
    """Map FASTQs → sorted & indexed BAM (minimap2 + samtools)."""
    fastq_input = pathlib.Path(fastq_input)
    if fastq_input.is_file():  # single file
       fastqs = [fastq_input]
    else: 
       fastqs = sorted(fastq_input.glob("*.fastq*"))

    if not fastqs:
        raise FileNotFoundError(f"No FASTQ files found in {fastq_input!s}")

    out_dir = pathlib.Path(out_dir)
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sam = out_dir / "aln.sam"
    bam = out_dir / "aln.bam"
    bam_sorted = out_dir / "aln.sorted.bam"

    print("▶ minimap2 …")
    subprocess.check_call(
        ["minimap2", "-a", "-x", "map-ont", "-t", str(threads),
         str(ref_fasta), *map(str, fastqs)],
        stdout=open(sam, "w"))
    print("▶ samtools sort …")
    subprocess.check_call(["samtools", "view", "-bS", str(sam)], stdout=open(bam, "wb"))
    subprocess.check_call(["samtools", "sort", str(bam), "-o", str(bam_sorted)])
    subprocess.check_call(["samtools", "index", str(bam_sorted)])
    sam.unlink(); bam.unlink()

    print("✅ ready:", bam_sorted)
    return bam_sorted
