from collections import defaultdict

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage

from .base import BaseClient
from ..types import Agent
from ..tools import functions_to_tools


class OpenAIClient(BaseClient):
    def __init__(self):
        self.client = AsyncOpenAI()

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

        messages = [{"role": "system", "content": instructions}] + history

        return await self.client.chat.completions.create(
            model=model_override or agent.model,
            messages=messages,
            tools=functions_to_tools(agent.functions),
            stream=stream,
            tool_choice=agent.tool_choice,
        )
