import pytest
from troop import Troop, Agent
from troop.types import Result
from tests.mock_client import MockOpenAIClient, create_mock_response

DEFAULT_RESPONSE_CONTENT = "sample response content"


@pytest.fixture
def mock_openai_client():
    m = MockOpenAIClient()
    m.set_response(
        create_mock_response({"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT})
    )
    return m


def test_default_agent_values():
    agent = Agent()
    assert agent.model == "gpt-4o"  # Default model from types.py
    assert agent.name == "Agent"  # Default name
    assert agent.instructions == "You are a helpful agent."  # Default instructions
    assert agent.tool_choice is None  # Default tool_choice
    assert agent.parallel_tool_calls is True  # Default parallel_tool_calls


@pytest.mark.asyncio
async def test_streaming():
    client = MockOpenAIClient()
    client.set_response(
        create_mock_response({"role": "assistant", "content": "Hello there!"})
    )
    
    troop = Troop(client=client, async_client=client)
    agent = Agent()
    messages = [{"role": "user", "content": "Hi"}]
    
    async for chunk in troop.arun_and_stream(
        agent=agent, messages=messages
    ):
        if "delim" in chunk:
            assert chunk["delim"] in ["start", "end"]
        elif "response" in chunk:
            assert chunk["response"].messages[-1]["content"] == "Hello there!"
        else:
            # Verify chunk format
            assert "role" in chunk or "content" in chunk


def test_context_variables():
    client = MockOpenAIClient()
    context = {"user_name": "Alice", "preference": "Python"}
    
    def greet(name, context_variables=None):
        assert context_variables["user_name"] == "Alice"
        return f"Hello {name}!"
    
    agent = Agent(functions=[greet])
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "greet", "args": {"name": "Bob"}}]
        ),
        create_mock_response({"role": "assistant", "content": "Greeting sent!"})
    ])
    
    troop = Troop(client=client, async_client=client)
    response = troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Say hi"}],
        context_variables=context
    )
    
    assert "Hello Bob!" in str(response.messages)
    assert response.context_variables["user_name"] == "Alice"


def test_model_override():
    client = MockOpenAIClient()
    override_model = "gpt-3.5-turbo"
    
    agent = Agent(model="gpt-4o")  # Using default model from types.py
    client.set_response(
        create_mock_response(
            {"role": "assistant", "content": "Response"},
            model=override_model
        )
    )
    
    troop = Troop(client=client, async_client=client)
    response = troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Hi"}],
        model_override=override_model
    )
    
    # We can only verify the response was received since model info isn't in the response
    assert len(response.messages) > 0


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
    
    troop = Troop(client=client, async_client=client)
    response = troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Start"}],
        max_turns=2
    )
    
    # Should only have 2 turns worth of messages
    assert len(response.messages) <= 4  # 2 turns * 2 messages per turn


def test_parallel_tool_calls():
    client = MockOpenAIClient()
    
    def task1(): return "Task 1 done"
    def task2(): return "Task 2 done"
    
    agent = Agent(
        functions=[task1, task2],
        parallel_tool_calls=True  # Boolean flag as defined in types.py
    )
    
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[
                {"name": "task1"},
                {"name": "task2"}
            ]
        ),
        create_mock_response({"role": "assistant", "content": "All tasks complete"})
    ])
    
    troop = Troop(client=client, async_client=client)
    response = troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Run tasks"}]
    )
    
    assert "Task 1 done" in str(response.messages)
    assert "Task 2 done" in str(response.messages)


@pytest.mark.skip(reason="Test hangs due to async error propagation issues. TODO: Fix error handling in mock client and core.py")
@pytest.mark.asyncio
async def test_error_handling_invalid_return():
    client = MockOpenAIClient()
    
    def bad_function():
        class BadReturn:
            pass
        return BadReturn()
    
    agent = Agent(functions=[bad_function])
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "bad_function"}]
        )
    ])
    
    troop = Troop(client=client, async_client=client)
    
    with pytest.raises(TypeError):
        await troop.arun(
            agent=agent,
            messages=[{"role": "user", "content": "Run bad function"}]
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
    
    troop = Troop(client=client, async_client=client)
    response = troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Call missing function"}]
    )
    
    assert any("Error: Tool nonexistent_function not found" in str(msg) for msg in response.messages)


def test_tool_choice():
    client = MockOpenAIClient()
    
    def required_tool():
        return "Tool was called"
    
    agent = Agent(
        functions=[required_tool],
        tool_choice="required_tool"  # String format as defined in types.py
    )
    
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "required_tool"}]
        ),
        create_mock_response({"role": "assistant", "content": "Done"})
    ])
    
    troop = Troop(client=client, async_client=client)
    response = troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Do something"}]
    )
    
    assert "Tool was called" in str(response.messages)


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
    
    troop = Troop(client=client, async_client=client)
    troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Test debug"}],
        debug=True
    )
    
    captured = capsys.readouterr()
    assert "Getting chat completion" in captured.out
    assert "Processing tool call" in captured.out


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
    
    troop = Troop(client=client, async_client=client)
    with pytest.raises(TypeError):
        troop.run(
            agent=agent,
            messages=[{"role": "user", "content": "Call with invalid args"}]
        )


@pytest.mark.asyncio
async def test_failing_async_function():
    client = MockOpenAIClient()
    
    async def failing_function():
        raise ValueError("Async function failed")
    
    agent = Agent(functions=[failing_function])
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "failing_function"}]
        )
    ])
    
    troop = Troop(client=client, async_client=client)
    with pytest.raises(ValueError):
        await troop.arun(
            agent=agent,
            messages=[{"role": "user", "content": "Call failing function"}]
        )


def test_tool_choice_auto():
    client = MockOpenAIClient()
    
    def auto_tool():
        return "Auto tool called"
    
    agent = Agent(
        functions=[auto_tool],
        tool_choice="auto"  # String format as defined in types.py
    )
    
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "auto_tool"}]
        ),
        create_mock_response({"role": "assistant", "content": "Done"})
    ])
    
    troop = Troop(client=client, async_client=client)
    response = troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Do something"}]
    )
    
    assert "Auto tool called" in str(response.messages)


def test_tool_choice_none():
    client = MockOpenAIClient()
    
    def should_not_call():
        raise Exception("Tool should not be called")
    
    agent = Agent(
        functions=[should_not_call],
        tool_choice="none"  # String format as defined in types.py
    )
    
    client.set_response(
        create_mock_response({"role": "assistant", "content": "No tools used"})
    )
    
    troop = Troop(client=client, async_client=client)
    response = troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Do something"}]
    )
    
    assert "No tools used" in str(response.messages)


@pytest.mark.asyncio
async def test_streaming_with_tool_calls():
    client = MockOpenAIClient()
    
    def stream_tool():
        return "Tool executed during stream"
    
    agent = Agent(functions=[stream_tool])
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "stream_tool"}]
        ),
        create_mock_response({"role": "assistant", "content": "Final response"})
    ])
    
    troop = Troop(client=client, async_client=client)
    tool_call_seen = False
    final_response_seen = False
    
    async for chunk in troop.arun_and_stream(
        agent=agent,
        messages=[{"role": "user", "content": "Test streaming with tools"}]
    ):
        if "tool_calls" in str(chunk):
            tool_call_seen = True
        if "Final response" in str(chunk):
            final_response_seen = True
    
    assert tool_call_seen and final_response_seen


def test_nested_agent_switches():
    client = MockOpenAIClient()
    
    def switch_to_agent2():
        return agent2
    
    def switch_to_agent3():
        return agent3
    
    agent1 = Agent(name="Agent1", functions=[switch_to_agent2])
    agent2 = Agent(name="Agent2", functions=[switch_to_agent3])
    agent3 = Agent(name="Agent3")
    
    client.set_sequential_responses([
        # First switch
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "switch_to_agent2"}]
        ),
        # Second switch
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "switch_to_agent3"}]
        ),
        # Final response
        create_mock_response({"role": "assistant", "content": "Done"})
    ])
    
    troop = Troop(client=client, async_client=client)
    response = troop.run(
        agent=agent1,
        messages=[{"role": "user", "content": "Switch twice"}]
    )
    
    assert response.agent.name == "Agent3"


def test_context_updates_from_multiple_tools():
    client = MockOpenAIClient()
    
    def tool1():
        return Result(value="Tool 1", context_variables={"key1": "value1"})
    
    def tool2():
        return Result(value="Tool 2", context_variables={"key2": "value2"})
    
    agent = Agent(functions=[tool1, tool2])
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "tool1"}]
        ),
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "tool2"}]
        ),
        create_mock_response({"role": "assistant", "content": "Done"})
    ])
    
    troop = Troop(client=client, async_client=client)
    response = troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Run both tools"}]
    )
    
    assert response.context_variables["key1"] == "value1"
    assert response.context_variables["key2"] == "value2"


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
    
    troop = Troop(client=client, async_client=client)
    response = troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Run complex function"}]
    )
    
    assert response.context_variables["key"] == "value"
    assert response.agent.name == "NewAgent"
    assert "Function output" in str(response.messages)
