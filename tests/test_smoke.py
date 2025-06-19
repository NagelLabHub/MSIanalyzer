import pathlib, tempfile, shutil
from Bio.SeqIO.QualityIO import FastqGeneralIterator
from motif_pipeline.pipeline.driver import run_marker_pipeline
import random, string

def _random_qual(n: int, qmin: int = 33, qmax: int = 73) -> str:
    return "".join(chr(random.randint(qmin, qmax)) for _ in range(n))

def _make_fastq(path: pathlib.Path):
    with open(path, "w") as f:
        for i in range(30):
            flank1 = "ACGT" * 5        # 20 nt
            motif  = "AAAG" * 5        # 20 nt
            flank2 = "TGCA" * 5        # 20 nt
            seq  = flank1 + motif + flank2
            qual = _random_qual(len(seq))
            f.write(f"@A_read{i}\n{seq}\n+\n{qual}\n")

# ----------------------------------------------------------------------
def test_run_marker_pipeline(tmp_path: pathlib.Path):
    # 1. synth FASTQs
    f_a = tmp_path / "A.fastq"
    f_b = tmp_path / "B.fastq"
    _make_fastq(f_a); _make_fastq(f_b)

    # 2. minimal config dict   (pass explicit fastq list → no globbing)
    cfg = {
        "seq1": "ACGTACGTACGT",
        "seq2": "TGCATGCATGCA",
        "motif": "AAAG",
        "fastq_files": [str(f_a), str(f_b)],
    }
    group_map = {"A": "WT", "B": "KO"}

    # 3. run (hierarchical tests OFF for speed)
    run_marker_pipeline("TEST", cfg, group_map, do_hierarchical_tests=False)
