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


def test_default_agent_values():
    agent = Agent()
    assert agent.model == "gpt-4o"  # Default model from types.py
    assert agent.name == "Agent"  # Default name
    assert agent.instructions == "You are a helpful agent."  # Default instructions
    assert agent.tool_choice is None  # Default tool_choice
    assert agent.parallel_tool_calls is True  # Default parallel_tool_calls


def test_model_override(mock_openai_client):
    override_model = "gpt-3.5-turbo"
    
    agent = Agent(model="gpt-4o")  # Using default model from types.py
    mock_openai_client.set_response(
        create_mock_response(
            {"role": "assistant", "content": "Response"},
            model=override_model
        )
    )
    
    troop = Troop(client=mock_openai_client)
    response = troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Hi"}],
        model_override=override_model
    )
    
    assert len(response.messages) > 0


def test_tool_choice():
    client = MockOpenAIClient()
    
    def required_tool():
        return "Tool was called"
    
    agent = Agent(
        functions=[required_tool],
        tool_choice="required_tool"
    )
    
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "required_tool"}]
        ),
        create_mock_response({"role": "assistant", "content": "Done"})
    ])
    
    troop = Troop(client=client)
    response = troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Do something"}]
    )
    
    assert "Tool was called" in str(response.messages)


def test_tool_choice_auto():
    client = MockOpenAIClient()
    
    def auto_tool():
        return "Auto tool called"
    
    agent = Agent(
        functions=[auto_tool],
        tool_choice="auto"
    )
    
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "auto_tool"}]
        ),
        create_mock_response({"role": "assistant", "content": "Done"})
    ])
    
    troop = Troop(client=client)
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
        tool_choice="none"
    )
    
    client.set_response(
        create_mock_response({"role": "assistant", "content": "No tools used"})
    )
    
    troop = Troop(client=client)
    response = troop.run(
        agent=agent,
        messages=[{"role": "user", "content": "Do something"}]
    )
    
    assert "No tools used" in str(response.messages)


@pytest.mark.asyncio
async def test_callable_instructions():
    """Test that callable instructions work with context variables"""
    client = MockOpenAIClient()
    
    def dynamic_instructions(context_variables):
        return f"You are helping {context_variables['user_name']}"
    
    agent = Agent(instructions=dynamic_instructions)
    context = {"user_name": "Alice"}
    
    client.set_response(
        create_mock_response({"role": "assistant", "content": "Hello Alice!"})
    )
    
    troop = Troop(client=client)
    response = await troop.arun(
        agent=agent,
        messages=[{"role": "user", "content": "Hi"}],
        context_variables=context
    )
    
    assert response.messages[-1]["content"] == "Hello Alice!"


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
    
    troop = Troop(client=client)
    response = troop.run(
        agent=agent1,
        messages=[{"role": "user", "content": "Switch twice"}]
    )
    
    assert response.agent.name == "Agent3"
