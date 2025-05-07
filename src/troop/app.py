import typer
from rich.console import Console
from pydantic_ai import Agent

from .commands import keys_app, servers_app, agents_app, models_app
from .utils import run_async, get_servers
from .config import settings

app = typer.Typer()
console = Console()

app.add_typer(keys_app, name="keys")
app.add_typer(servers_app, name="servers")
app.add_typer(agents_app, name="agents")
app.add_typer(models_app, name="models")


@app.command()
@run_async
async def prompt(
    message: str = typer.Argument(help="The message to send to the agent"),
    agent: str = typer.Option(None, help="The name of the agent"),
    model: str = typer.Option(None, help="The LLM model to use"),
):
    """Run a single prompt against an agent."""
    if not agent:
        agent = settings.agent
    if not model:
        model = settings.model
    llm = Agent(model=model, mcp_servers=get_servers(settings, agent))
    async with llm.run_mcp_servers():
        async with llm.run_stream(message) as result:
            async for chunk in result.stream_text(delta=True):
                console.print(chunk, sep="", end="")


@app.command()
@run_async
async def chat(
    agent: str = typer.Option(None, help="The name of the agent"),
    model: str = typer.Option(None, help="The LLM model to use"),
):
    """Start an interactive chat session."""
    agent = agent if agent else settings.agent
    model = model if model else settings.model
    llm = Agent(
        model=model,
        system_prompt=settings.agents[agent]["instructions"],
        mcp_servers=get_servers(settings, agent),
    )
    messages = []

    while True:
        console.print("[bold blue]User:[/bold blue] ", sep="", end="")
        message = typer.prompt("", type=str, prompt_suffix="")

        # await stream_with_tools(llm, message, messages)

        async with llm.run_mcp_servers():
            async with llm.run_stream(message, message_history=messages) as result:
                console.print("[bold green]Agent:[/bold green] ", sep="", end="")
                async for chunk in result.stream_text(delta=True):
                    console.print(chunk, sep="", end="")
                console.print()
                messages += result.new_messages()