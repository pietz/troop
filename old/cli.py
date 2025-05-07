from typing import Annotated

import typer
from dotenv import load_dotenv
from mcp.client.stdio import stdio_client
from mcp import ClientSession
from pydantic_ai import Agent
from rich.console import Console
from rich.table import Table

from .mcptools import get_tools, get_prompt
from .registry import (
    add_agent,
    add_key,
    get_server_params,
    list_agents,
    list_keys,
    remove_agent,
    remove_key,
)
from .utils import run_async, setup_api_key_env

load_dotenv()

console = Console()
app = typer.Typer(help="Troop - A simple CLI for LLM chat completions")
agent_app = typer.Typer(help="Manage agents")
key_app = typer.Typer(help="Manage API keys")

app.add_typer(agent_app, name="agent")
app.add_typer(key_app, name="key")


@app.command()
@run_async
async def prompt(
    message: str = typer.Argument(..., help="Message to send to the LLM"),
    agent_name: Annotated[str, typer.Option("--agent", "-a")] = "researcher",
    model: Annotated[str, typer.Option("--model", "-m")] = "openai:gpt-4o",
) -> None:
    """
    Send a single message to an LLM and get a response.
    """
    # Set up API key from registry if needed
    setup_api_key_env(model)

    # Get agent configuration
    server_params = get_server_params(agent_name)
    if not server_params:
        typer.echo(f"Error: Agent '{agent_name}' not found", err=True)
        raise typer.Exit(1)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await get_tools(session)
            prompt_text = await get_prompt(session)

            agent = Agent(
                model=model,
                system_prompt=prompt_text,
                tools=tools,
            )

            async with agent.run_stream(message) as result:
                async for chunk in result.stream_text(delta=True):
                    typer.echo(chunk, nl=False)
                typer.echo()


@app.command()
@run_async
async def chat(
    agent_name: Annotated[str, typer.Option("--agent", "-a")] = "researcher",
    model: Annotated[str, typer.Option("--model", "-m")] = "openai:gpt-4o",
) -> None:
    """
    Start an interactive chat session with an LLM.
    """
    # Set up API key from registry if needed
    setup_api_key_env(model)

    # Get agent configuration
    server_params = get_server_params(agent_name)
    if not server_params:
        typer.echo(f"Error: Agent '{agent_name}' not found", err=True)
        raise typer.Exit(1)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await get_tools(session)
            prompt_text = await get_prompt(session)

            agent = Agent(
                model=model,
                system_prompt=prompt_text,
                tools=tools,
            )

            messages = []

            while True:
                message = typer.prompt("Human")
                async with agent.run_stream(
                    message, message_history=messages
                ) as result:
                    typer.echo("Agent: ", nl=False)
                    async for chunk in result.stream_text(delta=True):
                        typer.echo(chunk, nl=False)
                    typer.echo()
                    messages += result.new_messages()


@agent_app.command("list")
def list_agents_cmd() -> None:
    """List all registered agents"""
    agents = list_agents()

    if not agents:
        console.print("No agents registered.")
        return

    table = Table(title="Registered Agents")
    table.add_column("Name", style="cyan")
    table.add_column("Command", style="green")
    table.add_column("Args", style="yellow")
    table.add_column("Description")

    for name, config in agents.items():
        table.add_row(
            name,
            config["command"],
            " ".join(config["args"]),
            config.get("description", ""),
        )

    console.print(table)


@agent_app.command("add")
def add_agent_cmd(
    name: Annotated[str, typer.Argument(help="Name of the agent")],
    command: Annotated[
        str, typer.Option("--command", "-c", help="Command to run the agent")
    ],
    args: Annotated[
        list[str],
        typer.Option(
            "--arg",
            "-a",
            help="Arguments to pass to the command (can be specified multiple times)",
        ),
    ],
    description: Annotated[
        str, typer.Option("--description", "-d", help="Description of the agent")
    ] = "",
) -> None:
    """Add or update an agent in the registry"""
    add_agent(name, command, args, description)
    console.print(f"Agent '[cyan]{name}[/cyan]' added successfully.")


@agent_app.command("remove")
def remove_agent_cmd(
    name: Annotated[str, typer.Argument(help="Name of the agent to remove")],
) -> None:
    """Remove an agent from the registry"""
    if remove_agent(name):
        console.print(f"Agent '[cyan]{name}[/cyan]' removed successfully.")
    else:
        console.print(f"Agent '[cyan]{name}[/cyan]' not found.", style="red")
        raise typer.Exit(1)


@key_app.command("list")
def list_keys_cmd() -> None:
    """List all registered API keys (provider names only)"""
    keys = list_keys()

    if not keys:
        console.print("No API keys registered.")
        return

    table = Table(title="Registered API Keys")
    table.add_column("Provider", style="cyan")
    table.add_column("Status")

    for provider in keys:
        table.add_row(provider, "Set")

    console.print(table)


@key_app.command("add")
def add_key_cmd(
    provider: Annotated[
        str, typer.Argument(help="Provider name (e.g., openai, anthropic)")
    ],
    key: Annotated[
        str, typer.Option("--key", "-k", help="API key", prompt=True, hide_input=True)
    ],
) -> None:
    """Add or update an API key for a provider"""
    add_key(provider.lower(), key)
    console.print(f"API key for '[cyan]{provider}[/cyan]' added successfully.")


@key_app.command("remove")
def remove_key_cmd(
    provider: Annotated[
        str, typer.Argument(help="Provider name to remove API key for")
    ],
) -> None:
    """Remove an API key for a provider"""
    if remove_key(provider.lower()):
        console.print(f"API key for '[cyan]{provider}[/cyan]' removed successfully.")
    else:
        console.print(f"API key for '[cyan]{provider}[/cyan]' not found.", style="red")
        raise typer.Exit(1)
