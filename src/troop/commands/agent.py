import typer
from rich import print as rprint
from rich.table import Table

from ..config import settings, RESERVED_NAMES

app = typer.Typer(
    name="agent",
    help="Manage AI agents with their instructions and tools. (list/add/remove/set)",
)


@app.command("list")
def list_agents():
    """List all available agents"""
    table = Table()
    table.add_column("Name", justify="left")
    table.add_column("Model", justify="left")
    table.add_column("Instructions", justify="left")
    table.add_column("Servers", justify="left")

    for name, agent in settings.agents.items():
        table.add_row(
            name,
            agent.get("model", "Not set"),
            agent["instructions"][:30] + "...",
            ", ".join(agent["servers"]),
        )

    rprint(table)


@app.command("add")
def add_agent(name: str = typer.Argument(None, help="Name of the agent")):
    """Add a new agent"""
    if not name:
        name = typer.prompt("Enter name")

    # Validate agent name
    if name in RESERVED_NAMES:
        rprint(
            f"[red]Error:[/red] '{name}' is a reserved command name and cannot be used as an agent name"
        )
        rprint("\nReserved names: " + ", ".join(sorted(RESERVED_NAMES)))
        return

    confirm = True
    if name in settings.agents:
        confirm = typer.confirm(f"Agent {name} already exists. Overwrite it?")

    if confirm:
        model = typer.prompt("Enter model (e.g., openai:gpt-4o)")
        instructions = typer.prompt("Enter instructions")
        servers = []
        while True:
            server = typer.prompt(
                "Enter MCP servers (leave empty to finish)", default=""
            )
            if not server:
                break
            if server not in settings.mcps:
                rprint(f"Server {server} does not exist")
                continue
            servers.append(server)

        settings.agents[name] = {
            "model": model,
            "instructions": instructions,
            "servers": servers,
        }
        settings.save()
        rprint(f"Added agent {name}")


@app.command("remove")
def remove_agent(name: str = typer.Argument(None, help="Name of the Agent")):
    """Remove an existing agent"""
    if not name:
        name = typer.prompt("Enter name")
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
        name = typer.prompt("Enter name")
    if name not in settings.agents:
        rprint(f"Agent {name} does not exist")
        return
    settings.default_agent = name
    settings.save()
    rprint(f"Set default agent to {name}")
