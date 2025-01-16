# Troop

A lightweight framework for multi-agent orchestration, forked from OpenAI's Swarm library.

## Install

Requires Python 3.10+

```shell
pip install git+https://github.com/pietz/troop.git
```

## Usage

```python
from troop import Troop, Agent

client = Troop()

def transfer_to_agent_b():
    return agent_b

agent_a = Agent(
    name="Agent A",
    instructions="You are a helpful agent.",
    functions=[transfer_to_agent_b],
)

agent_b = Agent(
    name="Agent B",
    instructions="Only speak in Haikus.",
)

response = client.run(
    agent=agent_a,
    messages=[{"role": "user", "content": "I want to talk to agent B."}],
)

print(response.messages[-1]["content"])
```

## Overview

Troop focuses on lightweight agent coordination and execution through two core abstractions:

1. `Agent`: Encapsulates instructions and tools
2. **Handoffs**: Allows agents to transfer control to other agents

These primitives enable building scalable multi-agent systems while maintaining simplicity.

## Documentation

### Running Troop

Initialize a client:

```python
from troop import Troop
client = Troop()
```

### Client.run() Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|----------|
| **agent** | `Agent` | Initial agent to be called | (required) |
| **messages** | `List` | List of message objects | (required) |
| **context_variables** | `dict` | Additional context variables | `{}` |
| **max_turns** | `int` | Maximum conversation turns | `float("inf")` |
| **model_override** | `str` | Override agent's model | `None` |
| **execute_tools** | `bool` | Execute tool calls | `True` |
| **stream** | `bool` | Enable streaming responses | `False` |
| **debug** | `bool` | Enable debug logging | `False` |

### Agent Configuration

| Field | Type | Description | Default |
|-------|------|-------------|----------|
| **name** | `str` | Agent name | `"Agent"` |
| **model** | `str` | Model to use | `"gpt-4"` |
| **instructions** | `str`/`func` | Agent instructions | `"You are a helpful agent."` |
| **functions** | `List` | Available functions | `[]` |
| **tool_choice** | `str` | Tool choice override | `None` |

### Functions

Functions can:
- Return strings (or values castable to strings)
- Return other agents for handoffs
- Access context variables
- Update context through Result objects

Example with context variables:

```python
def greet(context_variables, language):
    user_name = context_variables["user_name"]
    greeting = "Hola" if language.lower() == "spanish" else "Hello"
    return f"{greeting}, {user_name}!"

agent = Agent(functions=[greet])
```

### Streaming

```python
stream = client.run(agent, messages, stream=True)
for chunk in stream:
    print(chunk)
```

## Testing

Use the built-in REPL for testing:

```python
from troop.repl import run_demo_loop
run_demo_loop(agent, stream=True)
