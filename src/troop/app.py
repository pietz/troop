import typer
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from pydantic_ai import Agent

from .commands import provider_app, mcp_app, agent_app
from .utils import run_async, get_servers, setup_provider_env
from .config import settings, RESERVED_NAMES

console = Console()


async def stream_response(result, agent_name: str):
    """Stream response chunks to console"""
    console.print(
        f"[bold green]{agent_name.capitalize()}:[/bold green] ", sep="", end=""
    )
    async for chunk in result.stream_text(delta=True):
        console.print(chunk, sep="", end="")
    console.print()  # Newline after full response
    console.print()  # Add blank line after agent response


async def stream_message(result, sender: str = "Agent"):
    """Stream a message character by character."""
    panel = Panel("", title=sender.capitalize(), border_style="blue")
    with Live(panel, console=console, refresh_per_second=10) as live:
        async for chunk in result.stream_text(delta=False):
            panel.renderable = chunk
            live.update(panel)
    console.print()


def create_agent_command(agent_name: str):
    """Create a command function for a specific agent"""

    @run_async
    async def agent_command(
        prompt: str = typer.Option(
            None, "-p", "--prompt", help="Single prompt to send to the agent"
        ),
        model: str = typer.Option(
            None, "-m", "--model", help="Override the default model"
        ),
    ):
        # Use provided model or agent's default
        agent_config = settings.agents[agent_name]
        model = model or agent_config.get("model")

        if not model:
            console.print(
                f"[red]Error:[/red] No model specified for agent '{agent_name}'"
            )
            return

        # Set up provider API key environment variable
        provider = setup_provider_env(model, settings.providers)
        if provider:
            console.print(f"[dim]Using API key for {provider}[/dim]")

        # Create agent
        llm = Agent(
            model=model,
            system_prompt=agent_config["instructions"],
            mcp_servers=get_servers(settings, agent_name),
        )

        try:
            if prompt:
                async with llm.run_mcp_servers():
                    async with llm.run_stream(prompt) as result:
                        await stream_message(result, agent_name)
            else:
                messages = []

                console.print(f"[bold]Starting chat with {agent_name}[/bold]")

                async with llm.run_mcp_servers():
                    while True:
                        console.print("[bold blue]User:[/bold blue]", sep="", end="")
                        message = typer.prompt("", type=str, prompt_suffix="")

                        console.print()
                        async with llm.run_stream(
                            message, message_history=messages
                        ) as result:
                            await stream_message(result, agent_name)
                            messages += result.new_messages()
        except Exception as e:
            console.print(
                f"\n[red]Error:[/red] Failed to connect to MCP server: {str(e)}"
            )
            console.print(
                "[dim]Check that the MCP server command is correct and the server is installed.[/dim]"
            )
            raise typer.Exit(1)

    return agent_command


app = typer.Typer()

# Add static command groups
app.add_typer(provider_app, name="provider")
app.add_typer(mcp_app, name="mcp")
app.add_typer(agent_app, name="agent")

# Dynamically add agent commands at startup
for agent_name in settings.agents:
    if agent_name not in RESERVED_NAMES:
        app.command(name=agent_name)(create_agent_command(agent_name))
