import typer
from rich import print as rprint
from rich.table import Table

from ..config import settings

app = typer.Typer(name="keys", help="Manage API keys. (list/add/remove)")


@app.command("list")
def list_keys():
    """List all registered API keys"""
    table = Table()
    table.add_column("Provider", justify="left")
    table.add_column("Key", justify="left")

    for provider, key in settings.keys.items():
        table.add_row(provider, f"{key[:6]}...{key[-6:]}")

    rprint(table)


@app.command("add")
def add_key(
    provider: str = typer.Argument(None, help="Name of the LLM provider"),
    key: str = typer.Option(None, hide_input=True, help="API Key"),
):
    """Add or replace the API key for a specific provider"""
    if not provider:
        provider = typer.prompt("Provider")
    if not key:
        key = typer.prompt("API Key", hide_input=True)
    confirm = True
    if provider in settings.keys:
        confirm = typer.confirm(f"Provider {provider} already exists. Overwrite it?")
    if confirm:
        settings.keys[provider] = key
        settings.save()
        rprint(f"Added API key for {provider}")


@app.command("remove")
def remove_key(provider: str = typer.Argument(None, help="Name of the LLM provider")):
    """Delete the API key for a specific provider"""
    if not provider:
        provider = typer.prompt("Provider")
    if provider in settings.keys:
        del settings.keys[provider]
        settings.save()
        rprint(f"Deleted API key for {provider}")
    else:
        rprint(f"No API key found for {provider}")
