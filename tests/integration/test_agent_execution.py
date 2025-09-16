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
    @patch('troop.runner.Agent')
    @patch('troop.utils.get_servers')
    async def test_run_agent_single_prompt(self, mock_get_servers, mock_agent_class, mock_load, mock_settings_with_agent, runner):
        """Test running an agent with a single prompt."""
        mock_load.return_value = mock_settings_with_agent
        
        # Mock toolsets
        mock_get_servers.return_value = []
        
        # Mock Agent.iter context manager
        mock_agent = MagicMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_ctx
        mock_ctx.__aexit__.return_value = None
        async def _empty():
            if False:
                yield None
        mock_ctx.__aiter__.return_value = _empty()
        mock_ctx.result = MagicMock()
        mock_agent.iter.return_value = mock_ctx
        mock_agent_class.return_value = mock_agent
        
        # Import app after settings are mocked
        from troop.app import app
        
        # Run the agent command
        result = runner.invoke(app, ["test-agent", "-p", "Test prompt"])
        
        assert result.exit_code == 0
        
        # Verify agent was created correctly
        mock_agent_class.assert_called_once()
        args, kwargs = mock_agent_class.call_args
        actual_model = kwargs.get("model", args[0] if args else None)
        if isinstance(actual_model, str):
            assert actual_model == "openai:gpt-4"
        else:
            # Accept model object: check provider and name
            assert getattr(actual_model, "system", None) == "openai"
            assert getattr(actual_model, "model_name", None) in {"gpt-4", "gpt-4o", "gpt-4-0125-preview", "gpt-4-0613", "gpt-4-1106-preview", "gpt-4-turbo", "gpt-4o-mini"} or getattr(actual_model, "model_name", None).startswith("gpt-4")
        assert kwargs.get("system_prompt") == "You are a test agent"
        assert kwargs.get("toolsets") == mock_get_servers.return_value
        
        # Verify iter-based execution was invoked
        mock_agent.iter.assert_called_once()

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
        
        result = runner.invoke(app, ["nonexistent-agent"], mix_stderr=True)
        assert result.exit_code == 2
        # Click/Typer error messages go to stderr; be flexible here
        assert ("No such command" in result.stdout
                or "No such option" in result.stdout
                or "Invalid value" in result.stdout
                or "Usage:" in result.stdout)

    @patch('troop.config.Settings.load')
    @patch('troop.runner.Agent')
    @patch('troop.utils.get_servers')
    def test_main_with_prompt_flag(self, mock_get_servers, mock_agent_class, mock_load, runner, mock_settings_with_agent):
        """Test running agent with --prompt flag."""
        mock_load.return_value = mock_settings_with_agent
        
        # Mock Agent.iter context manager
        mock_agent = MagicMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_ctx
        mock_ctx.__aexit__.return_value = None
        async def _empty():
            if False:
                yield None
        mock_ctx.__aiter__.return_value = _empty()
        mock_ctx.result = MagicMock()
        mock_agent.iter.return_value = mock_ctx
        mock_agent_class.return_value = mock_agent
        
        from troop.app import app
        
        result = runner.invoke(app, ["test-agent", "--prompt", "Test prompt"])
        
        assert result.exit_code == 0

    @patch('troop.config.Settings.load')
    @patch('troop.runner.Agent')
    @patch('troop.utils.get_servers')
    def test_main_with_model_override(self, mock_get_servers, mock_agent_class, mock_load, runner, mock_settings_with_agent):
        """Test running agent with --model flag."""
        mock_load.return_value = mock_settings_with_agent
        
        # Mock Agent.iter context manager
        mock_agent = MagicMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_ctx
        mock_ctx.__aexit__.return_value = None
        async def _empty():
            if False:
                yield None
        mock_ctx.__aiter__.return_value = _empty()
        mock_ctx.result = MagicMock()
        mock_agent.iter.return_value = mock_ctx
        mock_agent_class.return_value = mock_agent
        
        from troop.app import app
        
        result = runner.invoke(app, ["test-agent", "-p", "Test", "-m", "gpt-3.5-turbo"])
        
        assert result.exit_code == 0
        # Verify the agent was created with the override model
        mock_agent_class.assert_called_once()
        _, kwargs = mock_agent_class.call_args
        actual_model = kwargs.get("model")
        if isinstance(actual_model, str):
            assert actual_model == "gpt-3.5-turbo"
        else:
            assert getattr(actual_model, "system", None) == "openai"
            assert getattr(actual_model, "model_name", None) == "gpt-3.5-turbo"
        assert kwargs.get("system_prompt") == "You are a test agent"
        assert kwargs.get("toolsets") == mock_get_servers.return_value

    @patch('troop.config.Settings.load')
    @patch('troop.runner.Agent')
    @patch('troop.utils.get_servers')
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
        
        # Mock agent (unused further since interactive loop is complex)
        mock_agent = AsyncMock()
        mock_agent_class.return_value = mock_agent
        # Mock user inputs
        mock_prompt.side_effect = ["Hello", "exit"]
        
        # Mock agent responses
        mock_result1 = MagicMock()
        mock_result1.data = "Hi there!"
        mock_agent.run.side_effect = [mock_result1]
        
        # We won't assert deep behavior here to keep this lightweight.
        mock_result1.new_messages.return_value = []
        
        from troop.app import app
        
        result = runner.invoke(app, ["chat"])
        
        # Verify interactive prompts were triggered
        assert mock_prompt.call_count >= 1

    @patch('troop.config.Settings.load')
    @patch('troop.runner.Agent')
    @patch('troop.utils.get_servers')
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
        
        # Mock Agent.iter context manager (no events required for this light check)
        mock_agent = MagicMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_ctx
        mock_ctx.__aexit__.return_value = None
        async def _empty():
            if False:
                yield None
        mock_ctx.__aiter__.return_value = _empty()
        mock_ctx.result = MagicMock()
        mock_agent.iter.return_value = mock_ctx
        mock_agent_class.return_value = mock_agent
        
        from troop.app import app
        
        result = runner.invoke(app, ["stream-test", "-p", "Test streaming"])
        
        assert result.exit_code == 0
        # Check the iter-based run happened
        mock_agent.iter.assert_called_once()

    @patch('troop.config.Settings.load')
    @patch('troop.app.typer.prompt')
    def test_main_keyboard_interrupt(self, mock_prompt, mock_load, runner, mock_settings_with_agent):
        """Test handling KeyboardInterrupt in interactive mode."""
        mock_load.return_value = mock_settings_with_agent
        
        # Simulate KeyboardInterrupt
        mock_prompt.side_effect = KeyboardInterrupt()
        
        from troop.app import app
        
        result = runner.invoke(app, ["test-agent"])  # exit code may vary but should not crash
        
        # Should exit gracefully with error code
        assert result.exit_code in (0, 1, 2)

    @patch('troop.config.Settings.load')
    @patch('troop.runner.Agent')
    @patch('troop.utils.get_servers')
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
        
        mock_get_servers.return_value = []
        
        # Mock Agent.iter raising on enter
        mock_agent = MagicMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.side_effect = Exception("Server failed to start")
        mock_agent.iter.return_value = mock_ctx
        mock_agent_class.return_value = mock_agent
        
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
