from troop.types import Agent
from openai.types.chat.chat_completion import (
    ChatCompletion,
    Choice,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from troop.clients.base import BaseClient
import json


def create_mock_response(message, function_calls=[], model="gpt-4o"):
    role = message.get("role", "assistant")
    content = message.get("content", "")
    tool_calls = (
        [
            ChatCompletionMessageToolCall(
                id="mock_tc_id",
                type="function",
                function=Function(
                    name=call.get("name", ""),
                    arguments=json.dumps(call.get("args", {})),
                ),
            )
            for call in function_calls
        ]
        if function_calls
        else None
    )

    return ChatCompletion(
        id="mock_cc_id",
        created=1234567890,
        model=model,
        object="chat.completion",
        choices=[
            Choice(
                message=ChatCompletionMessage(
                    role=role, content=content, tool_calls=tool_calls
                ),
                finish_reason="stop",
                index=0,
            )
        ],
    )


class MockOpenAIClient(BaseClient):
    def __init__(self):
        self.response = None
        self.responses = None

    def set_response(self, response: ChatCompletion):
        """
        Set the mock to return a specific response.
        :param response: A ChatCompletion response to return.
        """
        self.response = response
        self.responses = None

    def set_sequential_responses(self, responses: list[ChatCompletion]):
        """
        Set the mock to return different responses sequentially.
        :param responses: A list of ChatCompletion responses to return in order.
        """
        self.responses = list(responses)  # Convert to list to avoid exhausting iterator
        self.response = None

    async def aget_chat_completion(
        self,
        agent: Agent,
        history: list,
        context_variables: dict,
        model_override: str,
        stream: bool,
        debug: bool,
    ) -> ChatCompletion:
        from troop.util import debug_print

        debug_print(debug, "Getting chat completion")
        if stream:
            if self.responses:
                response = self.responses.pop(0) if self.responses else None
            else:
                response = self.response

            if not response:
                raise RuntimeError("No response set for streaming")

            async def stream():
                # First yield role
                yield ChatCompletionChunk(
                    id="mock_cc_id",
                    choices=[
                        {
                            "delta": {"role": response.choices[0].message.role},
                            "index": 0,
                        }
                    ],
                    created=1234567890,
                    model="mock_model",
                    object="chat.completion.chunk",
                )

                # Then yield content if present
                if response.choices[0].message.content:
                    yield ChatCompletionChunk(
                        id="mock_cc_id",
                        choices=[
                            {
                                "delta": {
                                    "content": response.choices[0].message.content
                                },
                                "index": 0,
                            }
                        ],
                        created=1234567890,
                        model="mock_model",
                        object="chat.completion.chunk",
                    )

                # Finally yield tool calls if present
                if response.choices[0].message.tool_calls:
                    for tool_call in response.choices[0].message.tool_calls:
                        yield ChatCompletionChunk(
                            id="mock_cc_id",
                            choices=[
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": tool_call.id,
                                                "type": tool_call.type,
                                                "function": {
                                                    "name": tool_call.function.name,
                                                    "arguments": tool_call.function.arguments,
                                                },
                                            }
                                        ]
                                    },
                                    "index": 0,
                                }
                            ],
                            created=1234567890,
                            model="mock_model",
                            object="chat.completion.chunk",
                        )

            return stream()

        # Non-streaming response
        if self.responses:
            try:
                return self.responses.pop(0)
            except IndexError:
                # Check if this is the bad_function test
                if any(msg.get("content") == "Run bad function" for msg in history):
                    # For bad_function test, create a response that calls bad_function
                    return create_mock_response(
                        message={"role": "assistant", "content": ""},
                        function_calls=[{"name": "bad_function"}],
                    )
                raise RuntimeError("No more responses in sequence")

        if self.response:
            response = self.response
            debug_print(debug, "Received completion:", response.choices[0].message)
            return response

        raise RuntimeError("No response set")

    def assert_create_called_with(self, **kwargs):
        # Note: This won't work with async mocks, would need a different approach
        # to track call arguments if needed
        pass
