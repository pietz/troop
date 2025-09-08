from __future__ import annotations

from typing import Any

from pydantic_ai import Agent

from .utils import get_servers, setup_provider_env
from .display import MessageDisplay


class AgentRunner:
    """Thin wrapper to build and run a pydantic_ai Agent from config."""

    def __init__(self, agent: Agent, name: str):
        self.agent = agent
        self.name = name

    @classmethod
    def from_config(
        cls,
        settings: Any,
        agent_name: str,
        model_override: str | None = None,
    ) -> "AgentRunner":
        agent_cfg = settings.agents[agent_name]
        model = model_override or agent_cfg.get("model")
        if not model:
            raise ValueError(f"No model specified for agent '{agent_name}'")

        # Set provider env and validate MCP servers
        setup_provider_env(model, settings.providers)

        server_names = agent_cfg.get("servers") or []
        missing = [s for s in server_names if s not in settings.mcps]
        if missing:
            raise KeyError(f"Unknown MCP servers: {', '.join(missing)}")

        kwargs: dict[str, Any] = {
            "model": model,
            "system_prompt": agent_cfg["instructions"],
            "toolsets": get_servers(settings, agent_name),
        }
        ms = agent_cfg.get("model_settings") or None
        if ms:
            kwargs["model_settings"] = ms

        return cls(agent=Agent(**kwargs), name=agent_name)

    async def run_once(
        self,
        prompt: str,
        display: MessageDisplay,
        verbose: bool,
        message_history: list | None = None,
    ):
        """Run agent using iter() to capture tool calls and streaming output."""
        async with self.agent.iter(prompt, message_history=message_history) as run:
            async for node in run:
                if Agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as req_stream:
                        await display.handle_streaming_events(req_stream, self.name)
                elif Agent.is_call_tools_node(node):
                    async with node.stream(run.ctx) as tool_stream:
                        await display.handle_tool_events(tool_stream, verbose)
        return run.result
