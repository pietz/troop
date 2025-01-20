from abc import ABC, abstractmethod
from typing import AsyncGenerator

from openai.types.chat import ChatCompletionMessage

from ..types import Agent
from ..util import run_sync


class BaseClient(ABC):
    @abstractmethod
    async def aget_chat_completion(
        self,
        agent: Agent,
        history: list,
        context_variables: dict,
        model_override: str,
        stream: bool,
        debug: bool,
    ) -> ChatCompletionMessage | AsyncGenerator:
        pass

    def get_chat_completion(
        self,
        agent: Agent,
        history: list,
        context_variables: dict,
        model_override: str,
        stream: bool,
        debug: bool,
    ) -> ChatCompletionMessage:
        return run_sync(
            self.aget_chat_completion(
                agent=agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
                debug=debug,
            )
        )
