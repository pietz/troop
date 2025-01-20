from collections import defaultdict

from anthropic import AsyncAnthropic
from openai.types.chat import ChatCompletionMessage

from .base import BaseClient
from ..types import Agent
from ..tools import functions_to_tools


class AnthropicClient(BaseClient):
    def __init__(self):
        self.client = AsyncAnthropic()

    async def aget_chat_completion(
        self,
        agent: Agent,
        history: list,
        context_variables: dict,
        model_override: str,
        stream: bool,
        debug: bool,
    ) -> ChatCompletionMessage:
        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )
        return await self.client.messages.create(
            model=model_override or agent.model,
            system=instructions,
            messages=history,
            tools=functions_to_tools(agent.functions),
            stream=stream,
        )
