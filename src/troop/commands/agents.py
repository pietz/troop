import typer
from rich import print as rprint
from rich.table import Table

from ..config import settings

app = typer.Typer(name="agents", help="Manage troop Agents. (list/add/remove)")


@app.command("list")
def list_agents():
    """List all available agents"""
    table = Table()
    table.add_column("Name", justify="left")
    table.add_column("Instructions", justify="left")
    table.add_column("Servers", justify="left")

    for name, agent in settings.agents.items():
        table.add_row(
            name,
            agent["instructions"][:20] + "...",
            ", ".join(agent["servers"]),
        )

    rprint(table)


@app.command("add")
def add_agent(name: str = typer.Argument(None, help="Name of the agent")):
    """Add a new agent"""
    if not name:
        name = typer.prompt("Name")
    confirm = True
    if name in settings.agents:
        confirm = typer.confirm(f"Agent {name} already exists. Overwrite it?")
    instructions = typer.prompt("Instructions")
    servers = []
    while confirm:
        server = typer.prompt("Server name (leave empty to finish)", default="")
        if not server:
            break
        if server not in settings.servers:
            rprint(f"Server {server} does not exist")
            continue
        servers.append(server)
    if confirm:
        settings.agents[name] = {
            "instructions": instructions,
            "servers": servers,
        }
        settings.save()
        rprint(f"Added agent {name}")


@app.command("remove")
def remove_agent(name: str = typer.Argument(None, help="Name of the Agent")):
    """Remove an existing agent"""
    if not name:
        name = typer.prompt("Name")
    if name in settings.agents:
        del settings.agents[name]
        settings.save()
        rprint(f"Deleted agent {name}")
    else:
        rprint(f"No agent found with name {name}")


@app.command("set")
def set_agent(name: str = typer.Argument(None, help="Name of the agent")):
    """Set the default agent for all servers"""
    if not name:
        name = typer.prompt("Agent")
    if name not in settings.agents:
        rprint(f"Agent {name} does not exist")
        return
    settings.agent = name
    settings.save()
    rprint(f"Set default agent to {name}")
