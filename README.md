# troop

Build your own troop of agents right from the CLI.

## Intro

troop is a lightweight command line tool based on [PydanticAI](https://ai.pydantic.dev/), the [Model Context Protocol](https://modelcontextprotocol.io) (MCP) and [Typer](https://typer.tiangolo.com). It allows you to setup and configure your own agents in minutes, while using industry standard frameworks under the hood.

In Troop we define agents as:

> Agent = LLM + Instructions + Tools

## Quickstart

We recommend using `uv` to install troop.

```bash
uv tool install troop
```

Add an API key for your favorite LLM provider. Since `troop` is based on PydanticAI, we support all of [these](https://ai.pydantic.dev/models/).

```bash
troop key add openai <your-api-key>
```

The `troop` default agent has access to tools for searching the web and loading content.

> The first time you run troop, it will ask you to set a default model. You can also do this manually by running `troop model set openai:gpt-4o`.

```bash
troop "What's the weather like in San Fransisco?"

# Assistant: It's a sunny day with low winds at 78°F.
```

Instead of stopping after a single iteration, we can also continue in an interactive chat mode.

```bash
troop chat

# > User: _
```

## Servers

We use MCP servers to provide tools to our agents. You can write your own or use existing ones and run them both locally or remotely.

```bash
troop server add

# Name: _

# Command: <enter> to skip

# Env name & value: <enter> to skip
```

Registered servers can be removed like this:

```bash
troop server remove <name>
```

### Agents

An agent in `troop` is defined by a name, an instruction text (system prompt) and a list of registered servers. The agent will be able to access all of the provided MCP servers and their tools at runtime.

```bash
troop agent add

# Enter name: _

# Enter model: openai:gpt-4o

# Enter instructions: <enter> to skip

# Add a server: <enter> to skip
```

This is how you run troop with a specific agent:

```bash
troop --agent <name> <query>
```

This is how you remove registered agents.

```bash
troop agent remove <name>
```

This is how you set an agent as the default choice:

```bash
troop agent set <name>
```

## Settings

Troop stores a global config YAML file in the user directory. On macOS it will be stored under ~/.troop/config.yaml and it looks like this:

```yaml
keys:
  openai: sk-proj-vBAU...
  anthropic: IRvjU...
  google: AIzaS...
servers:
  web_tools:
    command:
    - mcp-proxy
    - https://mcp-tools.up.railway.app/sse
    env: {}
agents:
  assistant:
    instructions: You're a helpful assistant with access to tools.
    servers:
    - web_tools
defaults:
  model: openai:gpt-4o
  agent: assistant
    
```

## Best Practices

### System Instructions vs. Tool Descriptions

When defining agents and their tools, you might come across the question:

> What should I describe in the instructions and what should I put in the tool description?

A pattern that works well is this:

- Tool Description: Explain what the tools does, what it returns and **HOW** it needs to be used on a technical level. There shouldn't be any mentions of other tools or servers.
- System Instructions: Explain **WHEN** and in what situation a tool should be used or favored over another. Focus on the overall process the agent will go through.