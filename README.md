# Troop

Build and manage AI agents from the CLI with PydanticAI and Model Context Protocol.

## Introduction

Troop is a lightweight CLI wrapper around [PydanticAI](https://ai.pydantic.dev/) and the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP). It enables users to easily create, customize, manage, and interact with AI agents through a simple yet powerful command line interface.

```bash
troop researcher -p "Any music events in Paris tonight?"

Researcher: There are multiple music events in Pari...
```

Think of troop as a generalized version of Claude Code, that allows you to build and use all kinds of useful agents based on any LLM you like.

In troop, an AI agent consists of 3 parts:

1. **Model**: Choose any LLM from any supported provider as the backbone of your agent.
2. **Tools**: Every tool in troop is provided through an MCP server. Mix and match as many as you like, remote or locally.
3. **Instructions**: Tie everything together and let the agent know how to act and which tools to use in what situation.

> Agent = Model + Tools + Instructions

## Features

- **Async-First Architecture**: Built with asyncio for efficient concurrent operations
- **PydanticAI**: As the core engine to manage requests and responses with different providers
- **MCP**: As a single interface for tool usage
- **Modern CLI Experience**: Beautiful terminal interface using Typer and Rich
- **REPL**: Nicely formatted display of Human, Agent, and Tool messages
- **Agent Registry**: Add and manage agents to your needs

## Installation

We recommend using `uv` to install troop:

```bash
uv tool install troop
```

Or with pip:

```bash
pip install troop
```

## Quick Start

Let's build and run our first custom agent.

### 1. Add a Provider

```bash
troop provider add
# Enter provider name/ID: openai
# Enter API key: sk-abcd...
```

This adds OpenAI as a model provider making all their models available in troop. We support all [PydanticAI providers](https://ai.pydantic.dev/models/).

### 2. Add an MCP Server

```bash
troop mcp add
# Enter name: web-tools
# Enter command: uvx mcp-web-tools
# Enter env var: BRAVE_SEARCH_API_KEY=abc...
```

This adds a predefined MCP server called mcp-web-tools with its search and fetch tools using uv, and accesses it directly from PyPI.

### 3. Create an Agent

```bash
troop agent add
# Enter name: researcher
# Enter model: openai:gpt-4o
# Enter MCP servers: web-tools
# Enter instructions: You're a helpful researcher agent with access to the web...
```

This defines the actual agent with the model, tools and instructions.

### 4. Use Your Agent

```bash
troop researcher

> What's the first headline on the Guardian?

> "Climate Summit Reaches Historic Agreement on Emissions"
```

This launches the interactive REPL for the researcher agent we just created.

## Commands Overview

### Agent Invocation

```bash
# Interactive chat mode (default)
troop researcher

# Single prompt mode
troop researcher -p "What's the weather in Paris?"

# Override model
troop researcher -m openai:gpt-4o-mini
```

### Provider Management

```bash
# List providers
troop provider list

# Add provider
troop provider add openai --api-key sk-abc123...

# Remove provider
troop provider remove openai
```

### MCP Server Management

```bash
# List servers
troop mcp list

# Add server
troop mcp add

# Remove server
troop mcp remove web-tools
```

### Agent Management

```bash
# List agents
troop agent list

# Add agent
troop agent add

# Remove agent
troop agent remove researcher

# Set default agent
troop agent set researcher
```

### Model Management

```bash
# Set default model
troop model set openai:gpt-4o
```

## Configuration

Troop stores a global config YAML file in the user directory:
- macOS: `~/.troop/config.yaml`
- Linux: `~/.config/troop/config.yaml`
- Windows: `%APPDATA%/troop/config.yaml`

Example configuration:

```yaml
providers:
  openai: sk-proj-vBAU...
  anthropic: sk-ant-api...
servers:
  web_tools:
    command:
    - uvx
    - mcp-web-tools
    env:
      BRAVE_SEARCH_API_KEY: BSA-abc123...
agents:
  researcher:
    model: openai:gpt-4o
    instructions: You're a helpful researcher with access to web tools...
    servers:
    - web_tools
defaults:
  model: openai:gpt-4o
  agent: researcher
```

## REPL Experience

Troop provides a rich interactive experience in the terminal, with clear formatting for different message types:

```
┌─ User ────────────────────────────────────────────────────────────────┐
│ What is the Model Context Protocol?                                   │
└───────────────────────────────────────────────────────────────────────┘

┌─ Tool Call: web_tools-search_web ─────────────────────────────────────┐
│ {                                                                     │
│   "query": "Model Context Protocol",                                  │
│ }                                                                     │
└───────────────────────────────────────────────────────────────────────┘

┌─ Researcher ──────────────────────────────────────────────────────────┐
│ The Model Context Protocol (MCP) is an open standard created by       │
│ Anthropic that enables secure, two-way connections between AI systems │
│ and external data sources or tools.                                   │
└───────────────────────────────────────────────────────────────────────┘
```

## Best Practices

### System Instructions vs. Tool Descriptions

When defining agents and their tools, consider:

- **Tool Description**: Explain what the tool does, what it returns and HOW it needs to be used on a technical level. There shouldn't be any mentions of other tools or servers.
- **System Instructions**: Explain WHEN and in what situation a tool should be used or favored over another. Focus on the overall process the agent will go through.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Links

- [GitHub Repository](https://github.com/pietz/troop)
- [Issue Tracker](https://github.com/pietz/troop/issues)
- [PydanticAI Documentation](https://ai.pydantic.dev/)
- [Model Context Protocol](https://modelcontextprotocol.io/)