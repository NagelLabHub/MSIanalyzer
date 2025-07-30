"""
Typer CLI wrappers for Motif-Pipeline.
"""
from __future__ import annotations
import json
import pathlib
from typing import Optional

import typer

from .driver import run_marker_pipeline, run_batch_pipeline
from ..viz.pileup import pileup_run_pipeline, plot_indel_profile

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _load_json(path: pathlib.Path) -> dict:
    """Load JSON, falling back to json5/yaml if installed, with a clear error."""
    txt = path.read_text()
    try:
        return json.loads(txt)
    except json.JSONDecodeError as e:  # pragma: no cover
        try:
            import json5
            return json5.loads(txt)
        except Exception:
            typer.secho(f"❌ JSON parse error in {path} – {e}", fg=typer.colors.RED)
            raise typer.Exit(1)


def _extract_marker_cfg(marker: str, cfg: dict) -> dict:
    """
    Accept three layouts:

    1. flat:        {"seq1": "..."}
    2. wrapped:     {"BAT25": {...}}
    3. manifest:    {"markers": {"BAT25": {...}}, ...globals }
    """
    if "markers" in cfg and marker in cfg["markers"]:
        return cfg["markers"][marker]
    if marker in cfg and isinstance(cfg[marker], dict):
        return cfg[marker]
    # already flat
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Typer app
# ──────────────────────────────────────────────────────────────────────────────
app = typer.Typer(add_completion=False, help="Motif-Pipeline CLI")

# ------------------------------------------------------------------ run-marker
@app.command("run-marker")
def run_marker(
    marker: str,
    config: pathlib.Path,
    group_map: Optional[pathlib.Path] = typer.Option(
        None,
        help="JSON mapping sample-prefix → group label "
        "(overrides any mapping inside CONFIG).",
    ),
    outcomes: str = typer.Option(
        "repeat_units",
        "--outcomes",
        help="Comma-sep outcome list: repeat_units,amplicon_len,interruptions…",
    ),
    skip_tests: bool = typer.Option(
        True, "--skip-tests/--run-tests", help="Skip hierarchical tests"
    ),
    skip_variant_summary: bool = typer.Option(
        False, "--skip-variant-summary", help="Skip variant-signature summary/plots"
    ),
    threads: int = typer.Option(1, "--threads", help="CPU cores (default 1)"),
):
    """Process one marker."""
    cfg_all = _load_json(config)
    cfg     = _extract_marker_cfg(marker, cfg_all)
    cfg["outcome_cols"] = [c.strip() for c in outcomes.split(",") if c.strip()]

    gmap = json.loads(group_map.read_text()) if group_map else None

    run_marker_pipeline(
        marker, cfg, gmap,
        do_hierarchical_tests=not skip_tests,
        do_variant_summary=not skip_variant_summary, 
        num_workers=threads,
    )

# ------------------------------------------------------------------ run-batch
@app.command("run-batch")
def run_batch(
    manifest: pathlib.Path,
    skip_tests: bool = typer.Option(
        True, "--skip-tests/--run-tests", help="Skip hierarchical tests"
    ),
    skip_variant_summary: bool = typer.Option(
        False, "--skip-variant-summary", help="Skip variant-signature summary/plots"
    ),
    threads: int = typer.Option(1, "--threads", help="CPU cores per marker"),
):
    """Run every marker in a manifest JSON (see README for schema)."""
    run_batch_pipeline(manifest, skip_tests=skip_tests, skip_variant_summary=skip_variant_summary,global_threads=threads)

# ------------------------------------------------------------------ pile-up
@app.command("pileup")
def pileup(
    fastq_dir: pathlib.Path,
    reference: pathlib.Path,
    region: Optional[str] = typer.Option(
        None, help="chr:start-end. If omitted, the whole contig is plotted."),
    threads: int = typer.Option(1, "--threads"),
):
    """Align FASTQs → BAM and plot indel spectrum."""
    bam = pileup_run_pipeline(fastq_dir, reference, threads=threads)
    out_png = bam.with_suffix(".png")           # e.g. pileup_tmp/aln.sorted.png

    if region:
        chrom, coords = region.split(":")
        start, end = map(int, coords.split("-"))
        plot_indel_profile(
            bam, reference, chrom, start, end,
            save_path=out_png)
    else:
        plot_indel_profile(
            bam, reference,
            save_path=out_png)

    typer.echo("Done – figure saved → {}".format(out_png))


if __name__ == "__main__":
    app()
