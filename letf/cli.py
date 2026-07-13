"""LETF command-line interface."""

import click

from letf.analyzer import analyze_run, format_analysis
from letf.comparator import compare_runs, format_comparison
from letf.runner import run_experiment
from letf.tracker import list_runs


@click.group()
def cli():
    """Lightweight Experiment Training Framework."""
    pass


@cli.command("run")
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--root", default="experiments", show_default=True)
@click.option("--device", default="cpu", show_default=True)
def run_cmd(config: str, root: str, device: str):
    """Run an experiment from a YAML config."""
    paths = run_experiment(config, root=root, device=device)
    click.echo(f"Run completed: {paths.run_id}")
    click.echo(f"Directory: {paths.run_dir}")


@cli.command("list")
@click.option("--root", default="experiments", show_default=True)
def list_cmd(root: str):
    """List experiment run IDs."""
    runs = list_runs(root)
    if not runs:
        click.echo("No runs found.")
        return
    for run_id in runs:
        click.echo(run_id)


@cli.command("analyze")
@click.argument("run_id")
@click.option("--root", default="experiments", show_default=True)
def analyze_cmd(run_id: str, root: str):
    """Summarize a single run."""
    click.echo(format_analysis(analyze_run(run_id, root=root)))


@cli.command("compare")
@click.argument("run_ids", nargs=-1, required=True)
@click.option("--root", default="experiments", show_default=True)
def compare_cmd(run_ids: tuple[str, ...], root: str):
    """Compare two or more runs."""
    click.echo(format_comparison(compare_runs(list(run_ids), root=root)))


if __name__ == "__main__":
    cli()