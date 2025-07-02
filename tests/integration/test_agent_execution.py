import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, call
from typer.testing import CliRunner
from troop.config import Settings
from pydantic_ai import Agent
from rich.console import Console


class TestAgentExecution:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def mock_settings_with_agent(self):
        """Create settings with a test agent configured."""
        return Settings(
            api_keys={"openai": "sk-test"},
            mcp_servers={
                "test-server": {
                    "command": ["echo", "test"],
                    "env": {"TEST": "true"}
                }
            },
            agents={
                "test-agent": {
                    "instructions": "You are a test agent",
                    "model": "openai:gpt-4",
                    "mcp_servers": ["test-server"]
                }
            },
            default_model="openai:gpt-4"
        )

    @patch('troop.config.Settings.load')
    @patch('troop.app.Agent')
    @patch('troop.app.get_servers')
    async def test_run_agent_single_prompt(self, mock_get_servers, mock_agent_class, mock_load, mock_settings_with_agent, runner):
        """Test running an agent with a single prompt."""
        mock_load.return_value = mock_settings_with_agent
        
        # Mock MCP server
        mock_server = AsyncMock()
        mock_get_servers.return_value = {"test-server": mock_server}
        
        # Mock agent
        mock_agent = AsyncMock()
        mock_result = MagicMock()
        mock_result.data = "Test response"
        mock_agent.run.return_value = mock_result
        mock_agent_class.return_value = mock_agent
        
        # Import app after settings are mocked
        from troop.app import app
        
        # Run the agent command
        result = runner.invoke(app, ["test-agent", "-p", "Test prompt"])
        
        assert result.exit_code == 0
        
        # Verify agent was created correctly
        mock_agent_class.assert_called_once_with(
            "openai:gpt-4",
            system_prompt="You are a test agent"
        )
        
        # Verify MCP server setup
        mock_agent.tool.assert_called()
        
        # Verify agent was run with the prompt
        mock_agent.run.assert_called_once_with("Test prompt")

    @patch('troop.config.Settings.load')
    def test_main_no_agents(self, mock_load, runner):
        """Test main app when no agents are configured."""
        mock_load.return_value = Settings()  # No agents
        
        # Import app after mocking
        from troop.app import app
        
        result = runner.invoke(app, ["--help"])
        
        # Should show help but no agent commands
        assert result.exit_code == 0
        assert "provider" in result.stdout
        assert "agent" in result.stdout

    @patch('troop.config.Settings.load')
    def test_main_agent_not_found(self, mock_load, runner, mock_settings_with_agent):
        """Test main app when specified agent doesn't exist."""
        mock_load.return_value = mock_settings_with_agent
        
        from troop.app import app
        
        result = runner.invoke(app, ["nonexistent-agent"])
        
        # Command not found, typer shows error
        assert result.exit_code == 2
        assert "No such option" in result.stdout or "Invalid value" in result.stdout

    @patch('troop.config.Settings.load')
    @patch('troop.app.Agent')
    @patch('troop.app.get_servers')
    def test_main_with_prompt_flag(self, mock_get_servers, mock_agent_class, mock_load, runner, mock_settings_with_agent):
        """Test running agent with --prompt flag."""
        mock_load.return_value = mock_settings_with_agent
        
        # Mock agent
        mock_agent = AsyncMock()
        mock_result = MagicMock()
        mock_result.data = "Test response"
        mock_stream = AsyncMock()
        mock_stream.__aenter__.return_value = mock_result
        mock_agent.run_stream.return_value = mock_stream
        mock_agent_class.return_value = mock_agent
        
        # Mock async context managers
        mock_mcp_ctx = AsyncMock()
        mock_mcp_ctx.__aenter__.return_value = None
        mock_mcp_ctx.__aexit__.return_value = None
        mock_agent.run_mcp_servers.return_value = mock_mcp_ctx
        
        # Mock stream_text method
        async def mock_stream_text(delta=True):
            yield "Test response"
        mock_result.stream_text = mock_stream_text
        
        from troop.app import app
        
        result = runner.invoke(app, ["test-agent", "--prompt", "Test prompt"])
        
        assert result.exit_code == 0

    @patch('troop.config.Settings.load')
    @patch('troop.app.Agent')
    @patch('troop.app.get_servers')
    def test_main_with_model_override(self, mock_get_servers, mock_agent_class, mock_load, runner, mock_settings_with_agent):
        """Test running agent with --model flag."""
        mock_load.return_value = mock_settings_with_agent
        
        # Mock agent
        mock_agent = AsyncMock()
        mock_result = MagicMock()
        mock_result.data = "Test response"
        mock_stream = AsyncMock()
        mock_stream.__aenter__.return_value = mock_result
        mock_agent.run_stream.return_value = mock_stream
        mock_agent_class.return_value = mock_agent
        
        # Mock async context managers
        mock_mcp_ctx = AsyncMock()
        mock_mcp_ctx.__aenter__.return_value = None
        mock_mcp_ctx.__aexit__.return_value = None
        mock_agent.run_mcp_servers.return_value = mock_mcp_ctx
        
        # Mock stream_text method
        async def mock_stream_text(delta=True):
            yield "Test response"
        mock_result.stream_text = mock_stream_text
        
        from troop.app import app
        
        result = runner.invoke(app, ["test-agent", "-p", "Test", "-m", "gpt-3.5-turbo"])
        
        assert result.exit_code == 0
        # Verify the agent was created with the override model
        mock_agent_class.assert_called_once_with(
            model="gpt-3.5-turbo",
            system_prompt="You are a test agent",
            mcp_servers=mock_get_servers.return_value
        )

    @patch('troop.config.Settings.load')
    @patch('troop.app.Agent')
    @patch('troop.app.get_servers')
    @patch('troop.app.typer.prompt')
    async def test_run_agent_interactive_mode(self, mock_prompt, mock_get_servers, mock_agent_class, mock_load, runner):
        """Test running agent in interactive mode (REPL)."""
        settings = Settings(
            api_keys={"openai": "sk-test"},
            agents={
                "chat": {
                    "instructions": "You are a chat agent",
                    "model": "gpt-4",
                    "mcp_servers": []
                }
            }
        )
        mock_load.return_value = settings
        
        # No need to mock console anymore
        
        # Mock agent
        mock_agent = AsyncMock()
        mock_agent_class.return_value = mock_agent
        
        # Mock user inputs
        mock_prompt.side_effect = ["Hello", "exit"]
        
        # Mock agent responses
        mock_result1 = MagicMock()
        mock_result1.data = "Hi there!"
        mock_agent.run.side_effect = [mock_result1]
        
        # Mock async context managers
        mock_mcp_ctx = AsyncMock()
        mock_mcp_ctx.__aenter__.return_value = None
        mock_mcp_ctx.__aexit__.return_value = None
        mock_agent.run_mcp_servers.return_value = mock_mcp_ctx
        
        # Mock stream result
        mock_stream = AsyncMock()
        mock_stream.__aenter__.return_value = mock_result1
        mock_agent.run_stream.return_value = mock_stream
        
        # Mock stream_text method
        async def mock_stream_text(delta=True):
            yield "Hi there!"
        mock_result1.stream_text = mock_stream_text
        mock_result1.new_messages.return_value = []
        
        from troop.app import app
        
        result = runner.invoke(app, ["chat"])
        
        # Verify interactive prompts
        assert mock_prompt.call_count == 2
        
        # Verify agent was called with user input
        mock_agent.run_stream.assert_called_once_with("Hello", message_history=[])

    @patch('troop.config.Settings.load')
    @patch('troop.app.Agent')
    @patch('troop.app.get_servers')
    async def test_run_agent_with_streaming(self, mock_get_servers, mock_agent_class, mock_load, runner):
        """Test agent execution with streaming responses."""
        settings = Settings(
            api_keys={"openai": "sk-test"},
            agents={
                "stream-test": {
                    "instructions": "Test streaming",
                    "model": "gpt-4",
                    "mcp_servers": []
                }
            }
        )
        mock_load.return_value = settings
        
        # Mock agent with streaming
        mock_agent = AsyncMock()
        mock_agent_class.return_value = mock_agent
        
        # Create async generator for streaming
        async def mock_stream():
            messages = [
                {"type": "text", "content": "Part 1"},
                {"type": "text", "content": " Part 2"},
                {"type": "tool_call", "tool": "test_tool", "args": {"arg": "value"}},
                {"type": "tool_result", "tool": "test_tool", "result": "Tool output"},
                {"type": "text", "content": " Final part"}
            ]
            for msg in messages:
                yield msg
        
        mock_result = MagicMock()
        mock_result.stream.return_value = mock_stream()
        mock_result.data = "Part 1 Part 2 Final part"
        mock_agent.run.return_value = mock_result
        
        # Mock async context managers
        mock_mcp_ctx = AsyncMock()
        mock_mcp_ctx.__aenter__.return_value = None
        mock_mcp_ctx.__aexit__.return_value = None
        mock_agent.run_mcp_servers.return_value = mock_mcp_ctx
        
        # Mock stream result
        mock_stream = AsyncMock()
        mock_stream.__aenter__.return_value = mock_result
        mock_agent.run_stream.return_value = mock_stream
        
        # Mock stream_text method
        async def mock_stream_text(delta=True):
            yield "Part 1 "
            yield "Part 2 "
            yield "Final part"
        mock_result.stream_text = mock_stream_text
        
        from troop.app import app
        
        result = runner.invoke(app, ["stream-test", "-p", "Test streaming"])
        
        assert result.exit_code == 0
        # Check the streaming happened
        mock_agent.run_stream.assert_called_once_with("Test streaming")

    @patch('troop.config.Settings.load')
    @patch('troop.app.typer.prompt')
    def test_main_keyboard_interrupt(self, mock_prompt, mock_load, runner, mock_settings_with_agent):
        """Test handling KeyboardInterrupt in interactive mode."""
        mock_load.return_value = mock_settings_with_agent
        
        # Simulate KeyboardInterrupt
        mock_prompt.side_effect = KeyboardInterrupt()
        
        from troop.app import app
        
        result = runner.invoke(app, ["test-agent"])
        
        # Should exit gracefully with error code
        assert result.exit_code == 1

    @patch('troop.config.Settings.load')
    @patch('troop.app.Agent')
    @patch('troop.app.get_servers')
    async def test_run_agent_with_tool_errors(self, mock_get_servers, mock_agent_class, mock_load, runner):
        """Test agent execution when tools throw errors."""
        settings = Settings(
            api_keys={"openai": "sk-test"},
            agents={
                "error-test": {
                    "instructions": "Test error handling",
                    "model": "gpt-4",
                    "mcp_servers": ["error-server"]
                }
            },
            mcp_servers={
                "error-server": {
                    "command": ["error"],
                    "env": {}
                }
            }
        )
        mock_load.return_value = settings
        
        # Mock MCP server that fails
        mock_server = AsyncMock()
        mock_server.run.side_effect = Exception("Server failed to start")
        mock_get_servers.return_value = {"error-server": mock_server}
        
        # Mock agent
        mock_agent = AsyncMock()
        mock_agent_class.return_value = mock_agent
        
        # Mock MCP context manager that raises error
        mock_mcp_ctx = AsyncMock()
        mock_mcp_ctx.__aenter__.side_effect = Exception("Server failed to start")
        mock_agent.run_mcp_servers.return_value = mock_mcp_ctx
        
        from troop.app import app
        
        result = runner.invoke(app, ["error-test", "-p", "Test"])
        
        # Should exit with error
        assert result.exit_code == 1
        assert "Failed to connect to MCP server" in result.stdout

    @patch('troop.config.Settings.load')
    def test_dynamic_command_creation(self, mock_load, runner):
        """Test that agent commands are dynamically created."""
        settings = Settings(
            api_keys={"openai": "sk-test"},
            agents={
                "dynamic1": {
                    "instructions": "First dynamic agent",
                    "model": "gpt-4",
                    "mcp_servers": []
                },
                "dynamic2": {
                    "instructions": "Second dynamic agent",
                    "model": "gpt-4",
                    "mcp_servers": []
                }
            }
        )
        mock_load.return_value = settings
        
        # Import app after settings are mocked
        from troop.app import app
        
        # Check help to see if commands were created
        result = runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "dynamic1" in result.stdout
        assert "dynamic2" in result.stdout