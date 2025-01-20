import pytest
from troop import Troop, Agent, Result
from tests.mock_client import MockOpenAIClient, create_mock_response


@pytest.fixture
def mock_openai_client():
    m = MockOpenAIClient()
    m.set_response(
        create_mock_response({"role": "assistant", "content": "Hello there!"})
    )
    return m


@pytest.mark.asyncio
async def test_streaming():
    client = MockOpenAIClient()
    client.set_response(
        create_mock_response({"role": "assistant", "content": "Hello there!"})
    )
    
    troop = Troop(client=client)
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
    
    troop = Troop(client=client)
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


@pytest.mark.asyncio
async def test_streaming_with_context_updates():
    client = MockOpenAIClient()
    
    def update_context():
        return Result(
            value="Context updated",
            context_variables={"key": "value"}
        )
    
    agent = Agent(functions=[update_context])
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "update_context"}]
        ),
        create_mock_response({"role": "assistant", "content": "Final response"})
    ])
    
    troop = Troop(client=client)
    context_updated = False
    final_chunk_seen = False
    
    async for chunk in troop.arun_and_stream(
        agent=agent,
        messages=[{"role": "user", "content": "Test streaming with context"}]
    ):
        if isinstance(chunk, dict) and "response" in chunk:
            if "key" in chunk["response"].context_variables:
                context_updated = True
        if "Final response" in str(chunk):
            final_chunk_seen = True
    
    assert context_updated and final_chunk_seen


@pytest.mark.asyncio
async def test_streaming_with_agent_switch():
    client = MockOpenAIClient()
    
    def switch_agent():
        return Agent(name="NewAgent")
    
    agent = Agent(functions=[switch_agent])
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "switch_agent"}]
        ),
        create_mock_response({"role": "assistant", "content": "Response from new agent"})
    ])
    
    troop = Troop(client=client)
    agent_switched = False
    final_response_seen = False
    
    async for chunk in troop.arun_and_stream(
        agent=agent,
        messages=[{"role": "user", "content": "Test streaming with agent switch"}]
    ):
        if isinstance(chunk, dict) and "response" in chunk:
            if chunk["response"].agent.name == "NewAgent":
                agent_switched = True
        if "Response from new agent" in str(chunk):
            final_response_seen = True
    
    assert agent_switched and final_response_seen


@pytest.mark.asyncio
async def test_streaming_error_handling():
    client = MockOpenAIClient()
    
    async def failing_tool():
        raise ValueError("Stream error")
    
    agent = Agent(functions=[failing_tool])
    client.set_sequential_responses([
        create_mock_response(
            message={"role": "assistant", "content": ""},
            function_calls=[{"name": "failing_tool"}]
        )
    ])
    
    troop = Troop(client=client)
    error_seen = False
    
    try:
        async for chunk in troop.arun_and_stream(
            agent=agent,
            messages=[{"role": "user", "content": "Test streaming error"}]
        ):
            pass
    except ValueError as e:
        if str(e) == "Stream error":
            error_seen = True
    
    assert error_seen
