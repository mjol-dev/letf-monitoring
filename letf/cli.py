"""LETF command-line interface."""
import click
@click.group()
def cli():
    """Lightweight Experiment Training Framework."""
    pass
    
if __name__ == "__main__":
    cli()