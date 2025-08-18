# MSIanalyzer  

**MSIanalyzer** is a flexible pipeline for high-resolution analysis and visualization of Microsatellite Instability (MSI) and tandem repeat regions from sequencing reads.



## Key Features

- **Universal Repeat Region Analysis** – Supports user-defined primers targeting **any repetitive motif region**.
- **Per-Read Repeat Count Quantification** – High-resolution repeat unit counting for accurate MSI profiling.
- **Detailed Indel Characterization** – Precise detection and annotation of interruptions in repeat sequences.
- **Customized Pileup Visualizations** – Clear visual summaries of coverage, repeats, indels, and genomic context.
- **Optimized for Nanopore Data** – Designed to tolerate higher error rates and fully leverage ONT long-read sequencing.
- **Cluster-Aware Statistical Analysis** – Incorporates read clustering to enhance detection of sample-level differences.



## Installation

```bash
pip install git+https://github.com/NagelLabHub/MSIanalyzer.git
```

## Quick Usage Example
A ready-to-run Google Colab notebook is available [here](https://colab.research.google.com/drive/13PjP7rVajoGOFAizytyXdfj6Souv2cts?usp=sharing) to demonstrate an example run of `MSIanalyzer` using the built-in example data.

To run `MSIanalyzer` via the command line, use the following examples from the `example` folder:

```bash
# Run analysis on a single marker (without or with sample comparison)
msianalyzer run-marker BAT25 example_marker.json
msianalyzer run-marker BAT25 example_marker.json --run-tests

# Run batch analysis on all markers in the JSON (can use '--threads' for parrallel processing)
msianalyzer run-batch example_marker.json

# Generate pileup visualization for FASTQ file(s)
msianalyzer pileup fastq_example/BVSBWG_3_500x.fastq hg38/chr4_BAT25.fa
```

## Citation

If you use this software, please cite: 

Ting Zhai, Daniel J. Laverty, Zachary D. Nagel (2025). MSIanalyzer: Targeted Nanopore Sequencing Enables Single Nucleotide Resolution Analysis of Microsatellite Instability Diversity. *bioRxiv* 2025.06.26.661510. https://doi.org/10.1101/2025.06.26.661510 