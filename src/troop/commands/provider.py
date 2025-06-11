import typer
from rich import print as rprint
from rich.table import Table

from ..config import settings

app = typer.Typer(name="provider", help="Manage API keys for LLM providers. (list/add/remove)")


@app.command("list")
def list_keys():
    """List all registered API keys"""
    table = Table()
    table.add_column("Provider", justify="left")
    table.add_column("Key", justify="left")

    for provider, key in settings.providers.items():
        table.add_row(provider, f"{key[:6]}...{key[-6:]}")

    rprint(table)


@app.command("add")
def add_key(
    provider: str = typer.Argument(None, help="Provider name/ID (e.g., openai, anthropic)"),
    key: str = typer.Option(None, "--api-key", hide_input=True, help="API Key"),
):
    """Add or replace the API key for a specific provider"""
    if not provider:
        provider = typer.prompt("Enter provider name/ID")
    if not key:
        key = typer.prompt("Enter API key", hide_input=True)
    confirm = True
    if provider in settings.providers:
        confirm = typer.confirm(f"Provider {provider} already exists. Overwrite it?")
    if confirm:
        settings.providers[provider] = key
        settings.save()
        rprint(f"Added API key for {provider}")


@app.command("remove")
def remove_key(provider: str = typer.Argument(None, help="Name of the LLM provider")):
    """Delete the API key for a specific provider"""
    if not provider:
        provider = typer.prompt("Provider")
    if provider in settings.providers:
        del settings.providers[provider]
        settings.save()
        rprint(f"Deleted API key for {provider}")
    else:
        rprint(f"No API key found for {provider}")
