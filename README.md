# MSIanalyzer  

**MSIanalyzer** is a flexible pipeline for high-resolution analysis and visualization of Microsatellite Instability (MSI) and tandem repeat regions from sequencing reads.

---

## Key Features

- **Universal Repeat Region Analysis** – Supports user-defined primers targeting **any repetitive motif region**.
- **Per-Read Repeat Count Quantification** – High-resolution repeat unit counting for accurate MSI profiling.
- **Detailed Indel Characterization** – Precise detection and annotation of interruptions in repeat sequences.
- **Customized Pileup Visualizations** – Clear visual summaries of coverage, repeats, indels, and genomic context.
- **Optimized for Nanopore Data** – Designed to tolerate higher error rates and fully leverage ONT long-read sequencing.
- **Cluster-Aware Statistical Analysis** – Incorporates read clustering to enhance detection of sample-level differences.

---

## Installation

```bash
pip install git+https://github.com/NagelLabHub/MSIanalyzer.git
```

## Quick Usage Example
From the repo root:

```bash
# Run analysis on a single marker
motif run-marker BAT25 example_marker.json

# Run batch analysis on all markers in the JSON
motif run-batch example_marker.json

# Generate pileup visualization for a single FASTQ file
motif-pileup pileup fastq_example/BVSBWG_2.fastq hg38/chr4_BAT25.fa
```

## Citation

If you use this software, please cite: 

Ting Zhai, Daniel J. Laverty, Zachary D. Nagel (2025). Nanopore sequencing enables accurate characterization of allelic diversity in MSI tandem repeats. 