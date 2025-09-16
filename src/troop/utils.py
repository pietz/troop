import os
import asyncio
from functools import wraps
from contextlib import asynccontextmanager
from typing import Optional

from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models import Model, infer_model
from pydantic_ai.settings import ModelSettings

from .config import Settings


def run_async(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


class QuietMCPServer(MCPServerStdio):
    """A version of ``MCPServerStdio`` that suppresses *all* output coming from the
    MCP server's **stderr** stream.

    We can't just redirect the server's *stdout* because that is where the JSON‑RPC
    protocol messages are sent.  Instead we override ``client_streams`` so we can
    hand our own ``errlog`` (``os.devnull``) to ``mcp.client.stdio.stdio_client``.
    """

    @asynccontextmanager
    async def client_streams(self):  # type: ignore[override]
        """Start the subprocess exactly like the parent class but silence *stderr*."""
        # Local import to avoid cycles
        from mcp.client.stdio import StdioServerParameters, stdio_client
        import logging

        server_params = StdioServerParameters(
            command=self.command,
            args=list(self.args),
            env=self.env or os.environ,
        )

        # Open ``/dev/null`` for the lifetime of the subprocess so anything the
        # server writes to *stderr* is discarded.
        #
        # This is to help with noisy MCP's that have options for verbosity
        with open(os.devnull, "w", encoding=server_params.encoding) as devnull:
            try:
                async with stdio_client(server=server_params, errlog=devnull) as (
                    read_stream,
                    write_stream,
                ):
                    yield read_stream, write_stream
            except GeneratorExit:
                # Silently handle generator cleanup
                pass
            except Exception as e:
                # Log the error but don't re-raise during cleanup
                logging.debug(f"Error during MCP server cleanup: {e}")
                raise


def get_tools(agent_name: str, settings: Settings) -> list:
    tools = []
    mcps = settings.agents[agent_name].get("mcps") or []
    missing = [mcp for mcp in mcps if mcp not in settings.mcps]
    if missing:
        raise KeyError(f"Unknown MCP servers: {', '.join(missing)}")
    for mcp in mcps:
        env = os.environ.copy()
        if "env" in mcp:
            env.update(mcp["env"])

        tools.append(
            QuietMCPServer(
                command=mcp["command"][0],
                args=mcp["command"][1:],
                env=env,
            )
        )
    return tools

def get_model(model_name: str, settings) -> Model:
    if model_name in settings.models:
        model = infer_model(settings.models[model_name].get("model"))
        model._settings = ModelSettings(
            **settings.models[model_name].get("settings", {})
        )
    else:
        model = infer_model(model_name)
    return model


# Provider to environment variable mapping
PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def setup_provider_env(model: Model, providers: dict) -> Optional[str]:
    """Set the correct API key env var based on provider.
    Returns the provider if an env var was set.
    """
    provider = model.system.lower()
    if provider in providers and provider in PROVIDER_ENV_VARS:
        os.environ[PROVIDER_ENV_VARS[provider]] = providers[provider]
        return provider
    return None
