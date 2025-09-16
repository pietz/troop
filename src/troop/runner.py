from __future__ import annotations
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import infer_model
from pydantic_ai.settings import ModelSettings

from .config import Settings
from .utils import get_tools, get_model, setup_provider_env
from .display import MessageDisplay


class AgentRunner:
    """Thin wrapper to build and run a pydantic_ai Agent from config."""

    def __init__(self, agent: Agent, name: str):
        self.agent = agent
        self.name = name

    @classmethod
    def from_config(
        cls,
        settings: Settings,
        agent_name: str,
        model_name: str | None = None,
    ) -> "AgentRunner":
        if agent_name not in settings.agents:
            raise KeyError(f"Unknown agent: {agent_name}")
        agent_cfg = settings.agents[agent_name]
        model = get_model(model_name or agent_cfg.get("model"), settings)
        tools = get_tools(agent_name, settings)
        setup_provider_env(model, settings.providers)

        kwargs: dict[str, Any] = {
            "model": model,
            "instructions": agent_cfg["instructions"],
            "tools": tools,
        }

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
