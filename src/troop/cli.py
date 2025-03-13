import asyncio
from functools import wraps

import typer
from pydantic_ai import Agent
from dotenv import load_dotenv
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

from .mcptools import mcptools, mcpprompt

load_dotenv()

app = typer.Typer(help="Troop - A simple CLI for LLM chat completions")


def typer_async(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


@app.command()
@typer_async
async def prompt(
    message: str = typer.Argument(..., help="Message to send to the LLM"),
) -> None:
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "researcher.py"],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await mcptools(session)
            prompt = await mcpprompt(session)

            agent = Agent(
                model="openai:gpt-4o",
                system_prompt=prompt,
                tools=tools,
            )

            async with agent.run_stream(message) as result:
                async for chunk in result.stream_text(delta=True):
                    typer.echo(chunk, nl=False)
                typer.echo()


@app.command()
@typer_async
async def chat() -> None:
    params = StdioServerParameters(
        command="uv",
        args=["run", "researcher.py"],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await mcptools(session)
            prompt = await mcpprompt(session)

            agent = Agent(
                model="openai:gpt-4o",
                system_prompt=prompt,
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
