from mcp import StdioServerParameters
from typing import Any

from .config import config


def list_agents() -> dict[str, dict[str, Any]]:
    """List all registered agents"""
    return config.agents.list()


def add_agent(name: str, command: str, args: list[str], description: str = "") -> None:
    """Add or update an agent in the registry"""
    config.agents.set(name, {
        "command": command,
        "args": args,
        "description": description
    })


def remove_agent(name: str) -> bool:
    """Remove an agent from the registry"""
    return config.agents.remove(name)


def get_agent(name: str) -> dict[str, Any] | None:
    """Get an agent configuration by name"""
    return config.agents.get(name)


def get_server_params(agent_name: str) -> StdioServerParameters | None:
    """Get StdioServerParameters for a registered agent"""
    agent = get_agent(agent_name)
    if not agent:
        return None
    
    return StdioServerParameters(
        command=agent["command"],
        args=agent["args"]
    )


def list_keys() -> list[str]:
    """List all registered API keys (provider names only)"""
    return list(config.keys.list().keys())


def add_key(provider: str, key: str) -> None:
    """Add or update an API key for a provider"""
    config.keys.set(provider, key)


def remove_key(provider: str) -> bool:
    """Remove an API key for a provider"""
    return config.keys.remove(provider)


def get_key(provider: str) -> str | None:
    """Get an API key by provider name"""
    return config.keys.get(provider)
