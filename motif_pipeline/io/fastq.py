# top of file, before any function defs
from __future__ import annotations
from datetime import datetime
import os, itertools
from collections import OrderedDict
import numpy as np
import pandas as pd
from Bio import SeqIO
from joblib import Parallel, delayed

from ..qc   import apply_read_support_filter, filter_valid_reads, map_group_labels
from ..utils.summary import compute_motif_summary, compute_summary_statistics
from ..core.analysis import analyse_read

def get_output_dir(marker):
    """
    Generates and creates a dated output directory for a given marker.
    """
    # Format current date as MM-DD-YY
    date_str = datetime.today().strftime("%m-%d-%y")

    # Create and return the directory
    output_dir = f"./output/{marker}_{date_str}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def extract_group_id(read_id):
    return read_id.split('_', 1)[1] if '_' in read_id else read_id

def process_fastq_file(file, seq1, seq2, motif, match_threshold, batch_size=500, num_workers=10):
    reads = SeqIO.parse(file, "fastq")
    batched_reads = iter(lambda: list(itertools.islice(reads, batch_size)), [])
    
    results = Parallel(n_jobs=num_workers)(
        delayed(process_batch)(batch, seq1, seq2, motif, match_threshold) for batch in batched_reads
    )

    return list(itertools.chain(*results))

def process_fastq_parallel(fastq_files, seq1, seq2, motif, match_threshold, num_workers=10, batch_size=500):
    all_processed_results = []
    for file in fastq_files:
        print(f"🚀 Processing {file}...")
        processed_results = process_fastq_file(
            file, seq1, seq2, motif, match_threshold, batch_size, num_workers
        )
        all_processed_results.extend(processed_results)
    return all_processed_results


# ------------- batch wrapper -------------
def process_batch(batch,
                  seq1,
                  seq2,
                  motif="AAAG",
                  max_subs=1,
                  anchor_units=3,
                  min_similarity=0.90):
    """
    Apply analyse_read() to a list of SeqRecord objects.
    Keeps id, group_id, quality in the output rows.
    """
    results = []
    for rec in batch:
        seq_str = str(rec.seq)
        rec_id  = getattr(rec, "id", None)
        group_id = extract_group_id(rec_id)
        phred   = rec.letter_annotations.get("phred_quality", [])

        stats = analyse_read(seq_str,
                             seq1,
                             seq2,
                             motif=motif,
                             max_subs=max_subs,
                             anchor_units=anchor_units,
                             min_similarity=min_similarity)
        # build ordered dict with 'id' and 'group_id' first
        ordered = OrderedDict()
        ordered["id"] = rec_id
        ordered["group_id"] = group_id
        ordered.update(stats)
        ordered["quality"] = phred

        results.append(ordered)
    return results
