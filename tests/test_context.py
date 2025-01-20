import pytest
from troop import Troop, Agent, Result
from tests.mock_client import MockOpenAIClient, create_mock_response

DEFAULT_RESPONSE_CONTENT = "sample response content"


@pytest.fixture
def mock_openai_client():
    m = MockOpenAIClient()
    m.set_response(
        create_mock_response({"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT})
    )
    return m


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
    
    troop = Troop(client=client)
    response = troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Say hi"}],
        context_variables=context
    )
    
    assert "Hello Bob!" in str(response.messages)
    assert response.context_variables["user_name"] == "Alice"


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
    
    troop = Troop(client=client)
    response = troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Run both tools"}]
    )
    
    assert response.context_variables["key1"] == "value1"
    assert response.context_variables["key2"] == "value2"


def test_context_persistence_across_turns():
    client = MockOpenAIClient()
    initial_context = {"session_id": "123"}
    
    def update_context():
        return Result(
            value="Updated",
            context_variables={"turn_count": 1}
        )
    
    agent = Agent(functions=[update_context])
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "update_context"}]
        ),
        create_mock_response({"role": "assistant", "content": "First turn"}),
        create_mock_response({"role": "assistant", "content": "Second turn"})
    ])
    
    troop = Troop(client=client)
    
    # First turn
    response = troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Turn 1"}],
        context_variables=initial_context
    )
    
    # Second turn - context should persist
    response = troop.run(
        agent=agent,
        messages=response.messages + [{"role": "user", "content": "Turn 2"}],
        context_variables=response.context_variables
    )
    
    assert response.context_variables["session_id"] == "123"
    assert response.context_variables["turn_count"] == 1


def test_context_with_agent_switch():
    client = MockOpenAIClient()
    initial_context = {"user_id": "user123"}
    
    def switch_agent():
        return Result(
            value="Switching",
            context_variables={"agent_switch_count": 1},
            agent=Agent(name="NewAgent")
        )
    
    agent = Agent(functions=[switch_agent])
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "switch_agent"}]
        ),
        create_mock_response({"role": "assistant", "content": "Switched"})
    ])
    
    troop = Troop(client=client)
    response = troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Switch agent"}],
        context_variables=initial_context
    )
    
    assert response.agent.name == "NewAgent"
    assert response.context_variables["user_id"] == "user123"
    assert response.context_variables["agent_switch_count"] == 1


def test_context_in_nested_tool_calls():
    client = MockOpenAIClient()
    initial_context = {"level": 0}
    
    def nested_tool():
        return Result(
            value="Nested call",
            context_variables={"level": 1}
        )
    
    def outer_tool():
        return Result(
            value="Outer call",
            context_variables={"outer": True}
        )
    
    agent = Agent(functions=[outer_tool, nested_tool])
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "outer_tool"}]
        ),
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "nested_tool"}]
        ),
        create_mock_response({"role": "assistant", "content": "Done"})
    ])
    
    troop = Troop(client=client)
    response = troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Run nested tools"}],
        context_variables=initial_context
    )
    
    assert response.context_variables["level"] == 1
    assert response.context_variables["outer"] is True


def test_context_with_parallel_tools():
    client = MockOpenAIClient()
    
    def tool1():
        return Result(value="Tool 1", context_variables={"source": "tool1"})
    
    def tool2():
        return Result(value="Tool 2", context_variables={"source": "tool2"})
    
    agent = Agent(
        functions=[tool1, tool2],
        parallel_tool_calls=True
    )
    
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[
                {"name": "tool1"},
                {"name": "tool2"}
            ]
        ),
        create_mock_response({"role": "assistant", "content": "Done"})
    ])
    
    troop = Troop(client=client)
    response = troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Run parallel tools"}]
    )
    
    # The last tool's context should win in case of conflicts
    assert response.context_variables["source"] == "tool2"
