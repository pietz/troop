import os
import asyncio
from functools import wraps
from contextlib import asynccontextmanager
from typing import Optional

from pydantic_ai.mcp import MCPServerStdio


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


def get_servers(settings, agent_name: str):
    servers = []
    for s in settings.agents[agent_name]["servers"]:
        mcp_config = settings.mcps[s]
        # Create environment with MCP-specific variables
        env = os.environ.copy()
        if "env" in mcp_config:
            env.update(mcp_config["env"])

        servers.append(
            QuietMCPServer(
                command=mcp_config["command"][0],
                args=mcp_config["command"][1:],
                env=env,
            )
        )
    return servers


# Provider to environment variable mapping
PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def setup_provider_env(model: str, providers: dict) -> Optional[str]:
    """Extract provider from model string and set API key environment variable.

    Args:
        model: Model string like "openai:gpt-4o" or "anthropic:claude-3-5-sonnet"
        providers: Dictionary of provider names to API keys

    Returns:
        The provider name if API key was set, None otherwise
    """
    # Extract provider from model string
    if ":" in model:
        provider = model.split(":")[0].lower()
    else:
        raise ValueError("Model name must be provided as 'provider:model'.")

    # Set API key if we have it
    if provider in providers and provider in PROVIDER_ENV_VARS:
        os.environ[PROVIDER_ENV_VARS[provider]] = providers[provider]
        return provider

    return None
