"""
Pytest configuration and shared fixtures for troop tests.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import asyncio
from troop.config import Settings


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config_file(temp_config_dir):
    """Mock the CONFIG_FILE to use a temporary location."""
    config_file = temp_config_dir / "config.yaml"
    with patch('troop.config.CONFIG_FILE', config_file):
        yield config_file


@pytest.fixture
def sample_settings():
    """Create sample settings for testing."""
    return Settings(
        api_keys={
            "openai": "sk-test123",
            "anthropic": "ak-test456"
        },
        mcp_servers={
            "web-tools": {
                "command": ["uvx", "mcp-web-tools"],
                "env": {"BRAVE_API_KEY": "test_key"}
            },
            "filesystem": {
                "command": ["node", "/path/to/filesystem-server.js"],
                "env": {}
            }
        },
        agents={
            "researcher": {
                "instructions": "You are a helpful research assistant with web access.",
                "model": "openai:gpt-4",
                "mcp_servers": ["web-tools"]
            },
            "coder": {
                "instructions": "You are a coding assistant with filesystem access.",
                "model": "anthropic:claude-3-opus",
                "mcp_servers": ["filesystem"]
            }
        },
        default_model="openai:gpt-4",
        default_agent="researcher"
    )


@pytest.fixture
def mock_mcp_server():
    """Create a mock MCP server."""
    server = MagicMock()
    server.run = MagicMock(return_value=asyncio.Future())
    server.run.return_value.set_result(MagicMock())
    return server


@pytest.fixture
def mock_agent():
    """Create a mock PydanticAI agent."""
    agent = MagicMock()
    result = MagicMock()
    result.data = "Mock response"
    agent.run = MagicMock(return_value=asyncio.Future())
    agent.run.return_value.set_result(result)
    return agent


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_console():
    """Mock Rich console for testing output."""
    with patch('troop.app.Console') as mock_console_class:
        console = MagicMock()
        mock_console_class.return_value = console
        yield console


# Test data fixtures
@pytest.fixture
def test_commands():
    """Sample command strings for testing parsing."""
    return {
        "simple": "ls -la",
        "quoted": 'echo "hello world"',
        "single_quoted": "echo 'single quotes'",
        "mixed_quotes": '''python -c "print('hello')"''',
        "escaped": 'echo "test\\"quote"',
        "complex": 'uvx mcp-server --arg="value with spaces" --flag',
        "empty": "",
        "whitespace": "   ",
    }


@pytest.fixture
def test_api_keys():
    """Sample API keys for testing."""
    return {
        "openai": "sk-proj-abcdefghijklmnopqrstuvwxyz123456",
        "anthropic": "ak-ant-api03-abcdefghijklmnopqrstuvwxyz",
        "google": "AIzaSyAbcdefghijklmnopqrstuvwxyz123456",
        "short": "key123",
        "empty": ""
    }


# Async test utilities
@pytest.fixture
async def async_mock_server():
    """Create an async mock MCP server."""
    server = MagicMock()
    
    async def mock_run():
        return MagicMock()
    
    server.run = mock_run
    return server


# Environment setup
@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    """Isolate test environment from system environment."""
    # Clear any existing API keys from environment
    env_vars_to_clear = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "BRAVE_SEARCH_API_KEY",
    ]
    for var in env_vars_to_clear:
        monkeypatch.delenv(var, raising=False)
    
    # Set test-specific environment variables if needed
    monkeypatch.setenv("TROOP_TEST_MODE", "1")


# CLI testing helpers
@pytest.fixture
def cli_invoke():
    """Helper function to invoke CLI commands with proper error handling."""
    from typer.testing import CliRunner
    runner = CliRunner()
    
    def invoke(app, args, **kwargs):
        result = runner.invoke(app, args, **kwargs)
        if result.exception and not isinstance(result.exception, SystemExit):
            raise result.exception
        return result
    
    return invoke