import shlex

import typer
from rich import print as rprint
from rich.table import Table

from ..config import settings

app = typer.Typer(name="mcp", help="Manage MCP servers that provide tools. (list/add/remove)")


@app.command("list")
def list_servers():
    """List all available servers"""
    table = Table()
    table.add_column("Name", justify="left")
    table.add_column("Command", justify="left")

    for name, params in settings.mcps.items():
        table.add_row(name, " ".join(params["command"]))

    rprint(table)


@app.command("add")
def add_server(name: str = typer.Argument(None, help="Name of the MCP server")):
    """Add a new server"""
    if not name:
        name = typer.prompt("Enter name")
    confirm = True
    if name in settings.mcps:
        confirm = typer.confirm(f"Server {name} already exists. Overwrite it?")
    if confirm:
        command = typer.prompt("Enter command")
        env = {}
        while True:
            env_var = typer.prompt(
                "Enter env var (KEY_NAME=KEY_VALUE, leave empty to finish)", default=""
            )
            if not env_var:
                break
            if "=" not in env_var:
                rprint("[red]Invalid format. Use KEY_NAME=KEY_VALUE[/red]")
                continue
            key, value = env_var.split("=", 1)
            env[key] = value
    if confirm:
        settings.mcps[name] = {
            "command": shlex.split(command),
            "env": env,
        }
        settings.save()
        rprint(f"Added MCP server {name}")


@app.command("remove")
def remove_server(name: str = typer.Argument(None, help="Name of the MCP server")):
    """Remove an existing Server"""
    if not name:
        name = typer.prompt("Enter name")
    if name in settings.mcps:
        del settings.mcps[name]
        settings.save()
        rprint(f"Deleted MCP server {name}")
    else:
        rprint(f"No MCP server found with name {name}")
