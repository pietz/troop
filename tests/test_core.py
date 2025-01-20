import json
import pytest
from troop import Troop, Agent
from tests.mock_client import MockOpenAIClient, create_mock_response

DEFAULT_RESPONSE_CONTENT = "sample response content"


@pytest.fixture
def mock_openai_client():
    m = MockOpenAIClient()
    m.set_response(
        create_mock_response({"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT})
    )
    return m


def test_run_with_simple_message(mock_openai_client):
    agent = Agent()
    # set up client and run
    client = Troop(client=mock_openai_client)
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    response = client.run(agent=agent, messages=messages)

    # assert response content
    assert response.messages[-1]["role"] == "assistant"
    assert response.messages[-1]["content"] == DEFAULT_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_async_run_with_simple_message(mock_openai_client):
    agent = Agent()
    # set up client and run
    client = Troop(client=mock_openai_client)
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    response = await client.arun(agent=agent, messages=messages)

    # assert response content
    assert response.messages[-1]["role"] == "assistant"
    assert response.messages[-1]["content"] == DEFAULT_RESPONSE_CONTENT


def test_max_turns():
    client = MockOpenAIClient()
    
    def continue_conversation():
        return "Let's continue!"
    
    agent = Agent(functions=[continue_conversation])
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "continue_conversation"}]
        ),
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "continue_conversation"}]
        ),
        create_mock_response({"role": "assistant", "content": "Final message"})
    ])
    
    troop = Troop(client=client)
    response = troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Start"}],
        max_turns=2
    )
    
    # Should only have 2 turns worth of messages
    assert len(response.messages) <= 4  # 2 turns * 2 messages per turn


def test_execute_tools_false(mock_openai_client):
    expected_location = "San Francisco"

    def get_weather(location):
        raise Exception("Tool should not be called")

    agent = Agent(name="Test Agent", functions=[get_weather])
    messages = [
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ]

    # set mock to return a response that triggers function call
    mock_openai_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[
                    {"name": "get_weather", "args": {"location": expected_location}}
                ],
            ),
            create_mock_response(
                {"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT}
            ),
        ]
    )

    # set up client and run
    client = Troop(client=mock_openai_client)
    response = client.run(agent=agent, messages=messages, execute_tools=False)

    # assert tool call is present in last response
    tool_calls = response.messages[-1].get("tool_calls")
    assert tool_calls is not None and len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "location": expected_location
    }


def test_debug_mode(capsys):
    client = MockOpenAIClient()
    
    def debug_function():
        return "Debug function called"
    
    agent = Agent(functions=[debug_function])
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "debug_function"}]
        ),
        create_mock_response({"role": "assistant", "content": "Done"})
    ])
    
    troop = Troop(client=client)
    troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Test debug"}],
        debug=True
    )
    
    captured = capsys.readouterr()
    assert captured.out  # Just verify some debug output was produced
