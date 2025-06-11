import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, call
from typer.testing import CliRunner
from troop.app import app
from troop import main
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

    @patch('troop.app.Settings.load')
    @patch('troop.app.Agent')
    @patch('troop.app.get_servers')
    async def test_run_agent_single_prompt(self, mock_get_servers, mock_agent_class, mock_load, mock_settings_with_agent):
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
        
        # Import after mocking to get the mocked version
        from troop.app import run_agent
        
        await run_agent("test-agent", "Test prompt", model_override=None)
        
        # Verify agent was created correctly
        mock_agent_class.assert_called_once_with(
            "openai:gpt-4",
            system_prompt="You are a test agent"
        )
        
        # Verify MCP server setup
        mock_agent.tool.assert_called()
        
        # Verify agent was run with the prompt
        mock_agent.run.assert_called_once_with("Test prompt")

    @patch('troop.app.Settings.load')
    @patch('troop.app.Console')
    def test_main_no_agents(self, mock_console_class, mock_load, runner):
        """Test main app when no agents are configured."""
        mock_load.return_value = Settings()  # No agents
        
        result = runner.invoke(app, ["test-agent"])
        
        assert result.exit_code == 1
        assert "No agents configured" in result.stdout

    @patch('troop.app.Settings.load')
    def test_main_agent_not_found(self, mock_load, runner, mock_settings_with_agent):
        """Test main app when specified agent doesn't exist."""
        mock_load.return_value = mock_settings_with_agent
        
        result = runner.invoke(app, ["nonexistent-agent"])
        
        assert result.exit_code == 1
        assert "Agent 'nonexistent-agent' not found" in result.stdout

    @patch('troop.app.Settings.load')
    @patch('troop.app.run_agent')
    def test_main_with_prompt_flag(self, mock_run_agent, mock_load, runner, mock_settings_with_agent):
        """Test running agent with --prompt flag."""
        mock_load.return_value = mock_settings_with_agent
        mock_run_agent.return_value = asyncio.Future()
        mock_run_agent.return_value.set_result(None)
        
        result = runner.invoke(app, ["test-agent", "--prompt", "Test prompt"])
        
        assert result.exit_code == 0
        mock_run_agent.assert_called_once_with("test-agent", "Test prompt", model_override=None)

    @patch('troop.app.Settings.load')
    @patch('troop.app.run_agent')
    def test_main_with_model_override(self, mock_run_agent, mock_load, runner, mock_settings_with_agent):
        """Test running agent with --model flag."""
        mock_load.return_value = mock_settings_with_agent
        mock_run_agent.return_value = asyncio.Future()
        mock_run_agent.return_value.set_result(None)
        
        result = runner.invoke(app, ["test-agent", "-p", "Test", "-m", "gpt-3.5-turbo"])
        
        assert result.exit_code == 0
        mock_run_agent.assert_called_once_with("test-agent", "Test", model_override="gpt-3.5-turbo")

    @patch('troop.app.Settings.load')
    @patch('troop.app.Agent')
    @patch('troop.app.get_servers')
    @patch('troop.app.Prompt.ask')
    @patch('troop.app.Console')
    async def test_run_agent_interactive_mode(self, mock_console_class, mock_prompt, mock_get_servers, mock_agent_class, mock_load):
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
        
        # Mock console
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        # Mock agent
        mock_agent = AsyncMock()
        mock_agent_class.return_value = mock_agent
        
        # Mock user inputs
        mock_prompt.side_effect = ["Hello", "exit"]
        
        # Mock agent responses
        mock_result1 = MagicMock()
        mock_result1.data = "Hi there!"
        mock_agent.run.side_effect = [mock_result1]
        
        from troop.app import run_agent
        await run_agent("chat", None, None)
        
        # Verify interactive prompts
        assert mock_prompt.call_count == 2
        
        # Verify agent was called with user input
        mock_agent.run.assert_called_once_with("Hello")

    @patch('troop.app.Settings.load')
    @patch('troop.app.Agent')
    @patch('troop.app.get_servers')
    async def test_run_agent_with_streaming(self, mock_get_servers, mock_agent_class, mock_load):
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
        
        from troop.app import run_agent
        await run_agent("stream-test", "Test streaming", None)
        
        # Verify streaming was used
        mock_result.stream.assert_called_once()

    @patch('troop.app.Settings.load')
    def test_main_keyboard_interrupt(self, mock_load, runner, mock_settings_with_agent):
        """Test handling KeyboardInterrupt in interactive mode."""
        mock_load.return_value = mock_settings_with_agent
        
        with patch('troop.app.run_agent') as mock_run_agent:
            mock_run_agent.side_effect = KeyboardInterrupt()
            
            result = runner.invoke(app, ["test-agent"])
            
            # Should exit gracefully
            assert result.exit_code == 0

    @patch('troop.app.Settings.load')
    @patch('troop.app.Agent')
    @patch('troop.app.get_servers')
    async def test_run_agent_with_tool_errors(self, mock_get_servers, mock_agent_class, mock_load):
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
        
        from troop.app import run_agent
        
        # Should handle server errors gracefully
        with pytest.raises(Exception, match="Server failed to start"):
            await run_agent("error-test", "Test", None)

    @patch('troop.app.Settings.load')
    def test_dynamic_command_creation(self, mock_load, runner):
        """Test that agent commands are dynamically created."""
        settings = Settings(
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
        
        # Import main to trigger dynamic command creation
        from troop import main
        
        # Check help to see if commands were created
        result = runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "dynamic1" in result.stdout
        assert "dynamic2" in result.stdout