# Troop

A simple CLI for interacting with LLMs through chat completions.

## Installation

You can install Troop directly from the repository:

```bash
pip install git+https://github.com/username/troop.git
```

Or if you've cloned the repository:

```bash
pip install -e .
```

## Configuration

Troop requires an OpenAI API key to function. You can provide it in two ways:

1. Set the `OPENAI_API_KEY` environment variable:
   ```bash
   export OPENAI_API_KEY=your-api-key-here
   ```

2. Pass it as an option when running commands:
   ```bash
   troop --api-key=your-api-key-here chat "Hello there"
   ```

## Usage

### Single Message

Send a single message to the LLM and get a response:

```bash
troop chat "What is the capital of France?"
```

Options:
- `--temperature`: Control randomness (0.0 = deterministic, 1.0 = creative)
- `--system-prompt`: Custom system prompt to use
- `--model`: LLM model to use (default: "gpt-4o")
- `--api-key`: OpenAI API key (if not set in environment)

### Interactive Chat

Start an interactive chat session:

```bash
troop interactive
```

The interactive mode maintains conversation history, allowing for more contextual interactions. Type 'exit' or 'quit' to end the session, or press Ctrl+C.

Options:
- `--temperature`: Control randomness for responses
- `--system-prompt`: Custom system prompt to use
- `--model`: LLM model to use (default: "gpt-4o")
- `--api-key`: OpenAI API key (if not set in environment)

### Process Files

Process a file with instructions:

```bash
troop file path/to/code.py "Explain what this code does"
```

Options:
- `--temperature`: Control randomness for the response
- `--system-prompt`: Custom system prompt to use
- `--model`: LLM model to use (default: "gpt-4o")
- `--api-key`: OpenAI API key (if not set in environment)

## Examples

```bash
# Get a concise answer with low temperature
troop chat --temperature=0.2 "Summarize the key points of quantum computing"

# Use a different model
troop --model="gpt-3.5-turbo" chat "Write a haiku about coding"

# Custom system prompt for specific tasks
troop chat --system-prompt="You are a Python expert" "How do I use decorators?"

# Interactive session with a coding assistant
troop interactive --system-prompt="You are an expert programmer. Provide code examples when appropriate."

# Analyze a file
troop file app.js "Identify potential bugs and suggest improvements"
```

## Using as a Library

You can use Troop as a library in your Python code:

```python
from troop.llm import ChatClient

# Initialize the client
client = ChatClient(model="gpt-4o", api_key="your-api-key-here")

# Send a message
response = client.send_message("Hello, how are you?")
print(response)

# Send multiple messages in a conversation
response1 = client.send_message("What is machine learning?")
print(response1)

response2 = client.send_message("Can you provide a simple example?")
print(response2)  # The client maintains conversation history

# Clear conversation history
client.reset_history()

# Use a custom system prompt
response = client.send_message(
    "Explain quantum computing",
    system_prompt="Explain concepts as if to a 10-year-old child",
    temperature=0.3
)
print(response)
