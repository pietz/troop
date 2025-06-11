# Simple Yet Powerful CLI Tool for building AI Agents

## Introduction

Troop is a modern LLM agent framework built on top of PydanticAI and the Model Context Protocol (MCP). It enables users to easily create, customize, manage, and interact with AI agents through a simple yet powerful CLI.

```bash
troop researcher -p "Any music events in Paris tonight?"

> There are multiple music events in Paris tonight depending...
```

Think of troop as a generalized version of Claude Code, that allows you to build and use all kinds of useful agents based on any LLM you like.

In troop, an AI agent consists of 3 parts:

1. **Model**: Choose any LLM from any supported provider as the backbone of your agent.
2. **Tools**: Every tool in troop is provided through an MCP server. Mix and match as many as you like, remote or locally.
3. **Instructions**: Tie everything together and let the agent know how to act and which tools to use in what situation.

> Agent = Model + Tools + Instructions

## Quick Start

Let's build and run our first custom agent.

```bash
troop provider add
# Enter provider name/ID: openai
# Enter API key: sk-abcd...
```

This adds OpenAI as a model provider making all their models available in troop.

```bash
troop mcp add
# Enter name: web-tools
# Enter command: uvx mcp-web-tools
# Enter env var: BRAVE_SEARCH_API_KEY=abc...
```

This adds a predefined MCP server called mcp-web-tools with its search and fetch tools using uv, and accesses it directly from PyPi. You can also use any MCP server: Local, remote, custom, Python, JavaScript, ...

```bash
troop agent add
# Enter name: researcher
# Enter model: openai:gpt-4o
# Enter MCP servers: web-tools
# Enter instructions: You're a helpful researcher agent with access to the web...
```

This defines the actual agent with the model, tools and instructions. The MCP servers prompt accepts a list of MCP server names that must have been registered in troop beforehand.

```bash
troop researcher

> What's the first headline on the Guardian?

> "Trump does something stupid, yet again"
```

This launches the interactive REPL for the researcher agent we just created. Through the tools it can interact with content on the web. It will run as many LLM requests necessary until it delivers the final response to the user.

This is just a simple example. A coding agent could include tools to edit local files and run bash commands. You could set up a deep research agent with sequential thinking and access to the web.

It's in your hands.

## Core Principles

- **Async-First Architecture**: Built with asyncio for efficient concurrent operations
- **PydanticAI**: As the core engine to manage requests and responses with different providers.
- **MCP**: As a single interface for tool usage.
- **Modern CLI Experience**: Beautiful terminal interface using Typer and Rich
- **REPL**: Nicely formatted display of Human, Agent, and Tool messages
- **Agent Registry**: Add and manage agents to your needs

## REPL

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

┌─ Tool Result: web_tools-search_web ───────────────────────────────────┐
│ [                                                                     │
│   {                                                                   │
│     "title": "Introducing the Model Context Protocol",                │
│     "url": "https://www.anthropic.com/news/model-context-protocol",   │
│     "description": "An open standard for connecting AI systems..."    │
│   },                                                                  │
│   ...                                                                 │
│ ]                                                                     │
└───────────────────────────────────────────────────────────────────────┘

┌─ Tool Call: web_tools-load_page ──────────────────────────────────────┐
│ {                                                                     │
│   "url": "https://www.anthropic.com/news/model-context-protocol"      │
│ }                                                                     │
└───────────────────────────────────────────────────────────────────────┘

┌─ Tool Result: web_tools-load_page ────────────────────────────────────┐
│ # Introduction                                                        │
│ Get Started with the Model Context Protocol (MCP)                     │
│                                                                       │
│ MCP is an open protocol that standardizes how applications provide    │
│ context to LLMs. Think of MCP like a USB-C port for AI applications.  │
│ ...                                                                   │
└───────────────────────────────────────────────────────────────────────┘

┌─ Researcher ──────────────────────────────────────────────────────────┐
│ The Model Context Protocol (MCP) is an open standard created by       │
│ Anthropic that enables secure, two-way connections between AI systems │
│ and external data sources or tools.                                   │
│                                                                       │
│ Key features of MCP include:                                          │
│                                                                       │
│ 1. Standardized communication between LLMs and data sources           │
│ 2. Support for resources, tools, and prompts                          │
│ 3. Security and access control built into the protocol                │
│                                                                       │
│ [Source: https://www.anthropic.com/news/model-context-protocol]       │
└───────────────────────────────────────────────────────────────────────┘
```


## API Reference

Overview of all commands in troop to create, use, update and delete entities and settings.

### Agent Invocation

The primary way to use troop is by calling agents directly:

#### Direct Agent Chat (Default)
```bash
troop researcher
# Starts an interactive chat session with the researcher agent
# Type 'exit' or 'quit' to end the conversation
```

#### Single Prompt Mode
```bash
troop researcher -p "What's the weather in Paris?"
# or
troop researcher --prompt "What's the weather in Paris?"
```

#### Model Override
```bash
troop researcher -m openai:gpt-4o-mini
# Uses a different model than the default
```

### Provider Management

Manage API keys for LLM providers.

#### `troop provider list`
List all registered API keys.

```bash
troop provider list
```

#### `troop provider add`
Add or replace an API key for a provider.

```bash
troop provider add
# Enter provider name/ID: openai
# Enter API key: sk-abc123...

# Or non-interactively:
troop provider add openai --api-key sk-abc123...
```

#### `troop provider remove`
Delete an API key for a provider.

```bash
troop provider remove openai
```

### MCP Server Management

Manage MCP servers that provide tools to agents.

#### `troop mcp list`
List all available MCP servers.

```bash
troop mcp list
```

#### `troop mcp add`
Add a new MCP server.

```bash
troop mcp add
# Enter name: web-tools
# Enter command: uvx mcp-web-tools
# Enter env var (leave empty to finish): BRAVE_SEARCH_API_KEY
# Enter env var: BRAVE_SEARCH_API_KEY: abc123...
```

#### `troop mcp remove`
Remove an existing MCP server.

```bash
troop mcp remove web-tools
```

### Model Management

Manage default LLM model settings.

#### `troop model set`
Set the default model for all agents.

```bash
troop model set openai:gpt-4o
```

### Agent Management

Manage AI agents with their instructions and tool configurations.

#### `troop agent list`
List all available agents.

```bash
troop agent list
```

#### `troop agent add`
Add a new agent.

```bash
troop agent add
# Enter name: researcher
# Enter instructions: You're a helpful researcher with access to web tools...
# Enter MCP servers (leave empty to finish): web-tools
# Enter MCP servers (leave empty to finish): 
```

**Note:** Agent names cannot be any of the reserved command names: `provider`, `mcp`, `agent`, `model`, `help`, `version`.

#### `troop agent remove`
Remove an existing agent.

```bash
troop agent remove researcher
```

#### `troop agent set`
Set the default agent.

```bash
troop agent set researcher
```

