import copy
import json
from collections import defaultdict
from typing import AsyncGenerator

from .util import merge_chunk, run_sync, debug_print
from .clients import BaseClient, OpenAIClient
from .tools import handle_tool_calls
from .types import (
    Agent,
    Response,
    Function,
    ChatCompletionMessageToolCall,
)


class Troop:
    def __init__(self, client=None):
        self.client: BaseClient = OpenAIClient() if not client else client

    async def arun_and_stream(
        self,
        agent: Agent,
        messages: list,
        context_variables: dict = {},
        model_override: str = None,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ):
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns:
            message = {
                "content": "",
                "sender": agent.name,
                "role": "assistant",
                "function_call": None,
                "tool_calls": defaultdict(
                    lambda: {
                        "function": {"arguments": "", "name": ""},
                        "id": "",
                        "type": "",
                    }
                ),
            }

            # get completion with current history, agent
            completion = await self.client.aget_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=True,
                debug=debug,
            )

            yield {"delim": "start"}
            async for chunk in completion:
                delta = json.loads(chunk.choices[0].delta.model_dump_json())
                if delta["role"] == "assistant":
                    delta["sender"] = active_agent.name
                yield delta
                delta.pop("role", None)
                delta.pop("sender", None)
                merge_chunk(message, delta)
            yield {"delim": "end"}

            message["tool_calls"] = list(message.get("tool_calls", {}).values())
            if not message["tool_calls"]:
                message["tool_calls"] = None
            debug_print(debug, "Received completion:", message)
            history.append(message)

            if not message["tool_calls"] or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # convert tool_calls to objects
            tool_calls = []
            for tool_call in message["tool_calls"]:
                function = Function(
                    arguments=tool_call["function"]["arguments"],
                    name=tool_call["function"]["name"],
                )
                tool_call_object = ChatCompletionMessageToolCall(
                    id=tool_call["id"], function=function, type=tool_call["type"]
                )
                tool_calls.append(tool_call_object)

            # handle function calls, updating context_variables, and switching agents
            partial_response = await handle_tool_calls(
                tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        yield {
            "response": Response(
                messages=history[init_len:],
                agent=active_agent,
                context_variables=context_variables,
            )
        }

    async def arun(
        self,
        agent: Agent,
        messages: list,
        context_variables: dict = {},
        model_override: str = None,
        stream: bool = False,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response | AsyncGenerator:
        """
        Run the agent asynchronously.
        
        Args:
            agent: The agent to run
            messages: The message history
            context_variables: Variables available to the agent
            model_override: Override the agent's model
            stream: If True, return an AsyncGenerator that yields chunks
            debug: Enable debug logging
            max_turns: Maximum number of turns
            execute_tools: Whether to execute tool calls
            
        Returns:
            Response if stream=False
            AsyncGenerator if stream=True
        """
        if stream:
            return self.arun_and_stream(
                agent=agent,
                messages=messages,
                context_variables=context_variables,
                model_override=model_override,
                debug=debug,
                max_turns=max_turns,
                execute_tools=execute_tools,
            )
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns and active_agent:
            # get completion with current history, agent
            completion = await self.client.aget_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
                debug=debug,
            )
            message = completion.choices[0].message
            debug_print(debug, "Received completion:", message)
            message.sender = active_agent.name
            history.append(
                json.loads(message.model_dump_json())
            )  # to avoid OpenAI types (?)

            if not message.tool_calls or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # handle function calls, updating context_variables, and switching agents
            partial_response = await handle_tool_calls(
                message.tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )

    def run_and_stream(
        self,
        agent: Agent,
        messages: list,
        context_variables: dict = {},
        model_override: str = None,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ):
        """Synchronous version of arun_and_stream"""
        async_gen = self.arun_and_stream(
            agent=agent,
            messages=messages,
            context_variables=context_variables,
            model_override=model_override,
            debug=debug,
            max_turns=max_turns,
            execute_tools=execute_tools,
        )

        # Convert async generator to sync generator
        while True:
            try:
                yield run_sync(async_gen.__anext__())
            except StopAsyncIteration:
                break

    def run(
        self,
        agent: Agent,
        messages: list,
        context_variables: dict = {},
        model_override: str = None,
        stream: bool = False,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        """Synchronous version of arun"""
        if stream:
            return self.run_and_stream(
                agent=agent,
                messages=messages,
                context_variables=context_variables,
                model_override=model_override,
                debug=debug,
                max_turns=max_turns,
                execute_tools=execute_tools,
            )
        return run_sync(
            self.arun(
                agent=agent,
                messages=messages,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
                debug=debug,
                max_turns=max_turns,
                execute_tools=execute_tools,
            )
        )
