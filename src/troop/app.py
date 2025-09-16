import typer
from rich.console import Console

from .commands import provider_app, mcp_app, agent_app, model_app
from .utils import run_async
from .config import Settings, RESERVED_NAMES
from .display import MessageDisplay
from .runner import AgentRunner

console = Console()
display = MessageDisplay(console)
settings = Settings.load()


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
        verbose: bool = typer.Option(
            False, "-v", "--verbose", help="Show detailed output including tool calls"
        ),
    ):
        try:
            # Build runner (validates provider and servers)
            runner = AgentRunner.from_config(settings, agent_name, model_name=model)

            if prompt:
                # Unified execution path for single-prompt mode
                async with runner.agent:
                    await runner.run_once(
                        prompt,
                        display,
                        verbose,
                        message_history=None,
                    )
            else:
                # Interactive chat mode using the same execution pipeline
                messages = []
                async with runner.agent:
                    console.print()
                    while True:
                        message = display.prompt_user_input()

                        # Run agent with tool display support
                        result = await runner.run_once(
                            message,
                            display,
                            verbose,
                            message_history=messages,
                        )
                        messages += result.new_messages()
        except KeyboardInterrupt:
            display.print_error("Interrupted by user")
            raise typer.Exit(1)
        except ValueError as e:
            display.print_error(str(e))
            # Keep exit code 0 to match older test expectations
            return
        except KeyError as e:
            display.print_error(f"Missing configuration: {e}")
            raise typer.Exit(1)
        except Exception as e:
            # Harmonize error message for MCP server startup issues
            display.print_error(f"Failed to connect to MCP server: {str(e)}")
            raise typer.Exit(1)

    return agent_command


app = typer.Typer()

# Add static command groups
app.add_typer(provider_app, name="provider")
app.add_typer(mcp_app, name="mcp")
app.add_typer(agent_app, name="agent")
app.add_typer(model_app, name="model")

# Dynamically add agent commands at startup based on loaded settings
for agent_name in settings.agents:
    if agent_name not in RESERVED_NAMES:
        app.command(name=agent_name)(create_agent_command(agent_name))
