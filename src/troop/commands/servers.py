import shlex

import typer
from rich import print as rprint
from rich.table import Table

from ..config import settings

app = typer.Typer(name="servers", help="Manage MCP Servers. (list/add/remove)")


@app.command("list")
def list_servers():
    """List all available servers"""
    table = Table()
    table.add_column("Name", justify="left")
    table.add_column("Command", justify="left")

    for name, params in settings.servers.items():
        table.add_row(name, " ".join(params["command"]))

    rprint(table)


@app.command("add")
def add_server(name: str = typer.Argument(None, help="Name of the MCP server")):
    """Add a new server"""
    if not name:
        name = typer.prompt("Server Name")
    confirm = True
    if name in settings.servers:
        confirm = typer.confirm(f"Server {name} already exists. Overwrite it?")
    if confirm:
        command = typer.prompt("Command (including args)")
    env = {}
    while confirm:
        key = typer.prompt(
            "Environment variable key (leave empty to finish)", default=""
        )
        if not key:
            break
        value = typer.prompt(f"Environment Value for {key}")
        env[key] = value
    if confirm:
        settings.servers[name] = {
            "command": shlex.split(command),
            "env": env,
        }
        settings.save()
        rprint(f"Added MCP server {name}")


@app.command("remove")
def remove_server(name: str = typer.Argument(None, help="Name of the MCP server")):
    """Remove an existing Server"""
    if not name:
        name = typer.prompt("Server Name")
    if name in settings.servers:
        del settings.servers[name]
        settings.save()
        rprint(f"Deleted MCP server {name}")
    else:
        rprint(f"No MCP server found with name {name}")
