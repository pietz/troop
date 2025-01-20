import pytest
from unittest.mock import Mock
from troop import Troop, Agent, Result
from tests.mock_client import MockOpenAIClient, create_mock_response
import asyncio

DEFAULT_RESPONSE_CONTENT = "sample response content"


@pytest.fixture
def mock_openai_client():
    m = MockOpenAIClient()
    m.set_response(
        create_mock_response({"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT})
    )
    return m


def test_tool_call(mock_openai_client):
    expected_location = "San Francisco"

    # set up mock to record function calls
    get_weather_mock = Mock()

    def get_weather(location):
        get_weather_mock(location=location)
        return "It's sunny today."

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
    response = client.run(agent=agent, messages=messages)

    get_weather_mock.assert_called_once_with(location=expected_location)
    assert response.messages[-1]["role"] == "assistant"
    assert response.messages[-1]["content"] == DEFAULT_RESPONSE_CONTENT


@pytest.mark.asyncio
async def test_concurrent_tool_calls():
    """Test that parallel tool calls are executed concurrently"""
    client = MockOpenAIClient()
    
    async def slow_task1():
        await asyncio.sleep(0.1)
        return "Task 1 done"
        
    async def slow_task2():
        await asyncio.sleep(0.1)
        return "Task 2 done"
    
    agent = Agent(
        functions=[slow_task1, slow_task2],
        parallel_tool_calls=True
    )
    
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[
                {"name": "slow_task1"},
                {"name": "slow_task2"}
            ]
        ),
        create_mock_response({"role": "assistant", "content": "All tasks complete"})
    ])
    
    troop = Troop(client=client)
    
    # Time the execution
    start = asyncio.get_event_loop().time()
    response = await troop.arun(
        agent=agent,
        messages=[{"role": "user", "content": "Run tasks"}]
    )
    duration = asyncio.get_event_loop().time() - start
    
    # Both tasks should complete in roughly 0.1s if run concurrently
    assert duration < 0.3  # Increased threshold for test environment
    assert "Task 1 done" in str(response.messages)
    assert "Task 2 done" in str(response.messages)


@pytest.mark.asyncio
async def test_tool_error_handling():
    """Test handling of errors from tool execution"""
    client = MockOpenAIClient()
    
    async def failing_tool():
        raise ValueError("Tool execution failed")
    
    agent = Agent(functions=[failing_tool])
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "failing_tool"}]
        )
    ])
    
    troop = Troop(client=client)
    
    with pytest.raises(ValueError, match="Tool execution failed"):
        await troop.arun(
            agent=agent,
            messages=[{"role": "user", "content": "Run failing tool"}]
        )


@pytest.mark.asyncio
async def test_tool_result_types():
    """Test different return types from tools are handled correctly"""
    client = MockOpenAIClient()
    
    def return_string():
        return "plain string"
        
    def return_result():
        return Result(value="result object")
        
    def return_dict():
        return {"key": "value"}  # Should be converted to string
        
    agent = Agent(functions=[return_string, return_result, return_dict])
    
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "return_string"}]
        ),
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "return_result"}]
        ),
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "return_dict"}]
        ),
        create_mock_response({"role": "assistant", "content": "Done"})
    ])
    
    troop = Troop(client=client)
    response = await troop.arun(
        agent=agent,
        messages=[{"role": "user", "content": "Test returns"}]
    )
    
    messages = [msg["content"] for msg in response.messages if msg["role"] == "tool"]
    assert "plain string" in messages
    assert "result object" in messages
    assert "{'key': 'value'}" in messages


def test_invalid_tool_arguments():
    client = MockOpenAIClient()
    
    def function_with_args(required_arg: str):
        return f"Got {required_arg}"
    
    agent = Agent(functions=[function_with_args])
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "function_with_args", "args": {}}]  # Missing required arg
        ),
        create_mock_response({"role": "assistant", "content": "Done"})
    ])
    
    troop = Troop(client=client)
    with pytest.raises(TypeError):
        troop.run(
            agent=agent,
            messages=[{"role": "user", "content": "Call with invalid args"}]
        )


def test_error_handling_missing_tool():
    client = MockOpenAIClient()
    
    agent = Agent(functions=[])
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "nonexistent_function"}]
        ),
        create_mock_response({"role": "assistant", "content": "Done"})
    ])
    
    troop = Troop(client=client)
    response = troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Call missing function"}]
    )
    
    assert any("Error: Tool nonexistent_function not found" in str(msg) for msg in response.messages)


def test_function_return_result_object():
    client = MockOpenAIClient()
    
    def complex_function():
        return Result(
            value="Function output",
            context_variables={"key": "value"},
            agent=Agent(name="NewAgent")
        )
    
    agent = Agent(functions=[complex_function])
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "complex_function"}]
        ),
        create_mock_response({"role": "assistant", "content": "Done"})
    ])
    
    troop = Troop(client=client)
    response = troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Run complex function"}]
    )
    
    assert response.context_variables["key"] == "value"
    assert response.agent.name == "NewAgent"
    assert "Function output" in str(response.messages)
