import typer
import json
from rich import print as rprint
from rich.table import Table

from ..config import settings, RESERVED_NAMES

app = typer.Typer(
    name="agent",
    help="Manage AI agents with their instructions and tools. (list/add/edit/remove/set)",
)


def _parse_setting_value(value: str):
    """Coerce a string value into bool/int/float/dict/list/str.

    - true/false/null/none are coerced to bool/None
    - integers and floats are coerced numerically
    - JSON objects/arrays (starting with { or [) are parsed
    - otherwise returned as string
    """
    v = value.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    if v.lower() in {"null", "none"}:
        return None
    # int
    try:
        if v and (v[0].isdigit() or (v[0] in "+-" and len(v) > 1 and v[1].isdigit())) and "." not in v:
            return int(v)
    except Exception:
        pass
    # float
    try:
        if any(ch in v for ch in [".", "e", "E"]):
            return float(v)
    except Exception:
        pass
    # json object/array
    if v.startswith("{") or v.startswith("["):
        try:
            return json.loads(v)
        except Exception:
            return value
    return value


@app.command("list")
def list_agents():
    """List all available agents"""
    table = Table()
    table.add_column("Name", justify="left")
    table.add_column("Model", justify="left")
    table.add_column("Instructions", justify="left")
    table.add_column("Servers", justify="left")
    table.add_column("Settings", justify="left")

    for name, agent in settings.agents.items():
        settings_preview = agent.get("model_settings") or {}
        if settings_preview:
            items = list(settings_preview.items())[:3]
            preview = ", ".join(f"{k}={v}" for k, v in items)
            if len(settings_preview) > 3:
                preview += ", ..."
        else:
            preview = "-"

        table.add_row(
            name,
            agent.get("model", "Not set"),
            agent["instructions"][:30] + "...",
            ", ".join(agent["servers"]),
            preview,
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

        # Optional model settings (ask right after model)
        model_settings: dict[str, object] = {}
        rprint("\n[dim]Enter model settings as key=value per line (leave empty to finish)[/dim]")
        while True:
            line = typer.prompt("Model setting", default="")
            if not line:
                break
            if "=" not in line:
                rprint("[red]Invalid format. Use KEY=VALUE[/red]")
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                rprint("[red]Invalid key[/red]")
                continue
            model_settings[key] = _parse_setting_value(value)

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
            "model_settings": model_settings,
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


@app.command("edit")
def edit_agent(name: str = typer.Argument(None, help="Name of the agent to edit")):
    """Edit an existing agent"""
    if not name:
        name = typer.prompt("Enter name")
    
    if name not in settings.agents:
        rprint(f"[red]Error:[/red] Agent '{name}' does not exist")
        return
    
    # Get current agent configuration
    current_agent = settings.agents[name]
    
    rprint(f"\n[bold]Editing agent: {name}[/bold]")
    rprint("[dim]Press Enter to keep current value[/dim]\n")
    
    # Edit model
    new_model = typer.prompt(
        "Enter model", 
        default=current_agent.get("model", ""),
        show_default=True
    )
    
    # Edit instructions
    rprint(f"\n[dim]Current instructions:[/dim] {current_agent['instructions']}")
    new_instructions = typer.prompt(
        "Enter instructions",
        default=current_agent["instructions"],
        show_default=False  # Don't show default for long text
    )
    
    # Edit servers
    rprint(f"\n[dim]Current servers:[/dim] {', '.join(current_agent['servers']) if current_agent['servers'] else 'None'}")
    rprint("[dim]Enter MCP servers (leave empty to keep current, 'none' to clear all)[/dim]")
    
    servers_input = typer.prompt(
        "Enter MCP servers (comma-separated)",
        default=", ".join(current_agent['servers']) if current_agent['servers'] else "",
        show_default=False
    )
    
    if servers_input.strip().lower() == 'none':
        new_servers = []
    elif servers_input.strip() == "":
        new_servers = current_agent['servers']
    else:
        # Parse comma-separated list
        new_servers = [s.strip() for s in servers_input.split(",") if s.strip()]
        
        # Validate servers exist
        invalid_servers = [s for s in new_servers if s not in settings.mcps]
        if invalid_servers:
            rprint(f"[yellow]Warning:[/yellow] These servers don't exist: {', '.join(invalid_servers)}")
            if not typer.confirm("Continue anyway?", default=False):
                return
    
    # Edit model settings (MCP-style key=value lines)
    current_settings = current_agent.get("model_settings", {}) or {}
    # Compact preview of current settings
    if current_settings:
        preview_items = ", ".join(
            f"{k}={current_settings[k]}" for k in list(current_settings.keys())[:4]
        )
        rprint(f"\n[dim]Current model settings:[/dim] {preview_items}"
               + (" ..." if len(current_settings) > 4 else ""))
    else:
        rprint("\n[dim]Current model settings:[/dim] None")

    rprint(
        "[dim]Enter model settings (KEY=VALUE). Leave empty to keep current, 'none' to clear all.[/dim]"
    )
    new_settings: dict[str, object] | None = None
    received_any = False
    while True:
        line = typer.prompt("Model setting", default="")
        if not line:
            break
        if line.strip().lower() == "none":
            new_settings = {}
            received_any = True
            break
        if "=" not in line:
            rprint("[red]Invalid format. Use KEY=VALUE[/red]")
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            rprint("[red]Invalid key[/red]")
            continue
        if new_settings is None:
            new_settings = {}
        new_settings[key] = _parse_setting_value(value)
        received_any = True

    # Decide final settings
    if not received_any:
        final_settings = current_settings
    else:
        final_settings = new_settings or {}

    # Update agent configuration
    settings.agents[name] = {
        "model": new_model,
        "instructions": new_instructions,
        "servers": new_servers,
        "model_settings": final_settings,
    }
    settings.save()
    rprint(f"\n[green]✓[/green] Agent '{name}' updated successfully")


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
