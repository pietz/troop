import typer
from rich.console import Console
from pydantic_ai import Agent

from .commands import provider_app, mcp_app, agent_app
from .utils import run_async, get_servers, setup_provider_env
from .config import settings, RESERVED_NAMES
from .display import MessageDisplay

console = Console()
display = MessageDisplay(console)


async def run_agent_with_tools(
    agent: Agent, prompt: str, agent_name: str, show_tools: bool, message_history=None
):
    """Run agent using iter() to capture tool calls and results"""
    async with agent.iter(prompt, message_history=message_history) as agent_run:
        async for node in agent_run:
            if Agent.is_user_prompt_node(node):
                # User prompt already displayed, skip
                pass
            elif Agent.is_model_request_node(node):
                # Stream the model's response
                async with node.stream(agent_run.ctx) as request_stream:
                    await display.handle_streaming_events(request_stream, agent_name)
            elif Agent.is_call_tools_node(node):
                # Handle tool calls and results
                async with node.stream(agent_run.ctx) as tool_stream:
                    await display.handle_tool_events(tool_stream, show_tools)
            elif Agent.is_end_node(node):
                # End of agent run
                pass

    return agent_run.result


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
        show_tools: bool = typer.Option(
            False, "-t", "--show-tools", help="Show tool calls and results"
        ),
    ):
        agent_config = settings.agents[agent_name]
        model = model or agent_config.get("model")

        if not model:
            display.print_error(f"No model specified for agent '{agent_name}'")
            return

        # Set up provider API key environment variable
        provider = setup_provider_env(model, settings.providers)

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
                        await display.stream_simple_response(result)
            else:
                messages = []
                async with llm.run_mcp_servers():
                    console.print()  # Line break before first user prompt
                    while True:
                        message = display.prompt_user_input()

                        # Run agent with tool display support
                        result = await run_agent_with_tools(
                            llm,
                            message,
                            agent_name,
                            show_tools,
                            message_history=messages,
                        )
                        messages += result.new_messages()
        except Exception as e:
            display.print_error(f"Failed to connect to MCP server: {str(e)}")
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
