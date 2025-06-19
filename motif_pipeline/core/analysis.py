import re
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner
from typing import Dict, Optional, Tuple
# ------------------------------------------------------------------
# robust edit-distance import  (C ext → pure-py fallback)
try:
    from Levenshtein import distance as lev
except ModuleNotFoundError:
    def lev(a: str, b: str) -> int:           # Wagner-Fischer O(len(a)*len(b))
        m, n = len(a), len(b)
        if m == 0: return n
        if n == 0: return m
        prev = list(range(n + 1))
        for i, ca in enumerate(a, 1):
            cur = [i] + [0] * n
            for j, cb in enumerate(b, 1):
                cur[j] = min(cur[j-1] + 1,
                              prev[j] + 1,
                              prev[j-1] + (ca != cb))
            prev = cur
        return prev[n]
# ------------------------------------------------------------------

# --------------------------------------------------------------------------
def find_longest_repeat_block(sequence: str,
                              motif: str,
                              min_units: int = 3) -> Optional[Tuple[int,int,int]]:
    """
    Return (start, end, units) of the longest *perfect* AAAG run in `sequence`.
    Units must be ≥ `min_units`; otherwise return None.
    """
    pattern = f"(?:{motif})+"
    hits = [(m.start(), m.end()) for m in re.finditer(pattern, sequence)]
    if not hits:
        return None
    start, end = max(hits, key=lambda x: x[1]-x[0])
    units = (end - start) // len(motif)
    if units < min_units:
        return None
    return start, end, units

# --------------------------------------------------------------------------
def locate_primer(seq: str,
                  primer: str,
                  aligner: PairwiseAligner,
                  min_similarity: float = .85
                  ) -> Optional[Tuple[int,int]]:
    """
    Local align `primer` to `seq` but **return exactly the primer length**:
    (start, start+len(primer)).  Avoids accidental 1-bp overhangs.
    """
    aln = aligner.align(seq, primer)
    if not aln:
        return None
    a = aln[0]
    ident = a.score / (aligner.match_score*len(primer))
    if ident < min_similarity:
        return None
    q_start = a.aligned[0][0][0]          # first matched position
    return q_start, q_start + len(primer) # enforce exact primer length


# ---------------------------------------------------------------------
# Utility: Hamming distance
import re
from Levenshtein import distance as lev  

def find_repeat_with_interruptions(seq,
                                   motif="AAAG",
                                   anchor_units=3,
                                   max_subs=1,
                                   max_indel=2):
    """
    Return (rep_start, rep_end, repeat_units, interruptions)
      interruptions = [{'unit_idx','seq','edit','type'}]
    """

    m_u, k = motif.upper(), len(motif)
    s_u = seq.upper()

    # -------- 1. locate every anchor of ≥ anchor_units perfect motifs
    anchor_pat = f'(?:{re.escape(m_u)})' + '{' + f'{anchor_units},' + '}'
    anchors = [(m.start(), m.end()) for m in re.finditer(anchor_pat, s_u)]
    if not anchors:
        return None

    def match_unit(pos, direction):
        # perfect
        span = (pos, pos + k) if direction > 0 else (pos - k, pos)
        s, e = span
        if 0 <= s < e <= len(seq) and s_u[s:e] == m_u:
            return (e if direction > 0 else s,
                    seq[s:e], 0, 'perfect')

        # indels
        for d in range(1, max_indel + 1):
            # insertion (+d)
            span = (pos, pos + k + d) if direction > 0 else (pos - (k + d), pos)
            s, e = span
            if 0 <= s < e <= len(seq) and lev(s_u[s:e], m_u) == d:
                return (e if direction > 0 else s,
                        seq[s:e], d, 'ins')
            # deletion (-d)
            span = (pos, pos + k - d) if direction > 0 else (pos - (k - d), pos)
            s, e = span
            if 0 <= s < e <= len(seq) and lev(s_u[s:e], m_u) == d:
                return (e if direction > 0 else s,
                        seq[s:e], d, 'del')

        # substitution
        span = (pos, pos + k) if direction > 0 else (pos - k, pos)
        s, e = span
        if 0 <= s < e <= len(seq):
            edits = lev(s_u[s:e], m_u)
            if 0 < edits <= max_subs:
                return (e if direction > 0 else s,
                        seq[s:e], edits, 'sub')
        return None

    # -------- 3. choose anchor that maximises PURE AAAG copies ----------
    best = None          # (units, rep_start, rep_end, pure_count)

    for a0, a1 in anchors:
        units = []

        # left extension
        pos = a0
        while True:
            res = match_unit(pos, -1)
            if res is None or res[0] < 0:
                break
            pos, u_seq, u_ed, u_tp = res
            units.insert(0, (u_seq, u_ed, u_tp, pos))

        # anchor perfect units
        for p in range(a0, a1, k):
            units.append((seq[p:p+k], 0, 'perfect', p))

        # right extension
        pos = a1
        while True:
            res = match_unit(pos, +1)
            if res is None or res[0] > len(seq):
                break
            pos, u_seq, u_ed, u_tp = res
            units.append((u_seq, u_ed, u_tp, pos - len(u_seq)))

        # summarise
        pure = sum(1 for _, _, t, _ in units if t == 'perfect')
        rep_start = units[0][3]
        rep_end   = units[-1][3] + len(units[-1][0])

        if best is None or pure > best[3] or (pure == best[3] and len(units) > len(best[0])):
            best = (units, rep_start, rep_end, pure)

    # -------- 4. final output ------------------------------------------
    units, rep_start, rep_end, pure_count = best
    interruptions = [
        dict(unit_idx=i, seq=sequ, edit=ed, type=typ)
        for i, (sequ, ed, typ, _) in enumerate(units) if typ != 'perfect'
    ]
    return rep_start, rep_end, sum(1 for _, _, t, _ in units if t == 'perfect'), interruptions


# ------------- per-read analysis ----------
def analyse_read(seq_raw: str,
                 primer_fwd: str,
                 primer_rev: str,
                 motif: str = "AAAG",
                 max_subs: int = 1,
                 anchor_units: int = 3,
                 min_similarity: float = 0.85,
                 aligner: Optional[PairwiseAligner] = None) -> dict:
    """
    Analyse one read end-to-end.  Returns a dict with keys:
      valid, strand, repeat_units, repeat_len, flank1_len, flank2_len,
      amplicon_len, interruptions, seq_flank1, seq_repeat, seq_flank2
    """
    if aligner is None:                 # create a strict local aligner
        aligner = PairwiseAligner(mode="local")
        aligner.match_score, aligner.mismatch_score = 2, -1
        aligner.open_gap_score, aligner.extend_gap_score = -0.5, -0.2

    rev_rc = str(Seq(primer_rev).reverse_complement())

    # ---------- try forward orientation ---------------------------------
    fwd = locate_primer(seq_raw, primer_fwd, aligner, min_similarity)
    rev = locate_primer(seq_raw, rev_rc,     aligner, min_similarity)
    if fwd and rev and fwd[0] < rev[0]:
        seq, strand, f_pos, r_pos = seq_raw, "Forward", fwd, rev
    else:
        # ---------- try reverse-complement orientation ------------------
        seq_rc = str(Seq(seq_raw).reverse_complement())
        fwd = locate_primer(seq_rc, primer_fwd, aligner, min_similarity)
        rev = locate_primer(seq_rc, rev_rc,     aligner, min_similarity)
        if fwd and rev and fwd[0] < rev[0]:
            seq, strand, f_pos, r_pos = seq_rc, "Reverted", fwd, rev
        else:
            return {"valid": False, "strand": "Unmatched"}

    # ---------- repeat detection inside the amplicon window -------------
    window = seq[f_pos[1]:r_pos[0]]
    rep = find_repeat_with_interruptions(window,
                                         motif=motif,
                                         anchor_units=anchor_units,
                                         max_subs=max_subs)
    if rep is None:
        return {"valid": False, "strand": strand}

    rep_start_local, rep_end_local, pure_units, intr = rep
    rep_start = f_pos[1] + rep_start_local
    rep_end   = f_pos[1] + rep_end_local

    return {
        "valid": True,
        "strand": strand,
        "repeat_units": pure_units,
        "repeat_len": rep_end - rep_start,
        "flank1_len": rep_start - f_pos[1],
        "flank2_len": r_pos[0] - rep_end,
        "amplicon_len": r_pos[1] - f_pos[0],
        "read_class": "pure_repeat" if not intr else "repeat_interruption",
        "interruptions": intr,
        "seq_flank1": seq[f_pos[1]:rep_start],
        "seq_repeat": seq[rep_start:rep_end],
        "seq_flank2": seq[rep_end:r_pos[0]],
    }
