import typer
from rich import print as rprint

from ..config import settings

app = typer.Typer(name="models", help="Manage LLMs. (set)")


@app.command("set")
def set_model(model: str = typer.Argument(None, help="Name of the LLM")):
    """Set the default model for all agents"""
    if not model:
        model = typer.prompt("Model")
    settings.model = model
    settings.save()
    rprint(f"Set default model to {model}")
