import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from typer.testing import CliRunner
from troop.config import Settings


class TestAgentExecutionSimple:
    """Simplified integration tests that test core functionality."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_help_shows_static_commands(self, runner):
        """Test that help command shows the static commands."""
        from troop.app import app
        
        result = runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "provider" in result.stdout
        assert "mcp" in result.stdout
        assert "agent" in result.stdout
    
    def test_provider_list_command(self, runner):
        """Test that provider list command works."""
        from troop.app import app
        
        with patch('troop.config.Settings.load') as mock_load:
            mock_load.return_value = Settings(providers={"openai": "sk-test"})
            
            result = runner.invoke(app, ["provider", "list"])
            
            assert result.exit_code == 0
            assert "openai" in result.stdout
    
    def test_mcp_list_command(self, runner):
        """Test that mcp list command works."""
        from troop.app import app
        # Patch module-level settings directly to avoid import-order issues
        with patch('troop.commands.mcp.settings') as mock_settings:
            mock_settings.mcps = {
                "test-server": {
                    "command": ["echo", "test"],
                    "env": {}
                }
            }
            result = runner.invoke(app, ["mcp", "list"])
            
            assert result.exit_code == 0
            assert "test-server" in result.stdout
    
    def test_agent_list_command(self, runner):
        """Test that agent list command works."""
        from troop.app import app
        # Patch module-level settings directly
        with patch('troop.commands.agent.settings') as mock_settings:
            mock_settings.agents = {
                "test-agent": {
                    "instructions": "Test agent",
                    "model": "gpt-4",
                    "servers": []
                }
            }
            result = runner.invoke(app, ["agent", "list"])
            
            assert result.exit_code == 0
            assert "test-agent" in result.stdout
    
    @patch('troop.runner.Agent')
    @patch('troop.utils.get_servers')
    def test_agent_execution_flow(self, mock_get_servers, mock_agent_class):
        """Test the basic flow of agent execution."""
        # Mock Agent.iter context manager to yield no events and a result
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
        mock_get_servers.return_value = []
        
        # Test the create_agent_command function directly
        from troop.app import create_agent_command
        
        # Create a test command
        test_command = create_agent_command("test-agent")
        
        # Mock the settings for this agent
        with patch('troop.app.settings') as mock_settings:
            mock_settings.agents = {
                "test-agent": {
                    "instructions": "Test instructions",
                    "model": "gpt-4",
                    "mcp_servers": []
                }
            }
            mock_settings.providers = {"openai": "sk-test"}
            
            # Test with prompt
            runner = CliRunner()
            result = runner.invoke(test_command, ["--prompt", "Test prompt"])
            
            assert result.exit_code == 0
            mock_agent_class.assert_called_once()
            _, kwargs = mock_agent_class.call_args
            actual_model = kwargs.get("model")
            if isinstance(actual_model, str):
                assert actual_model == "gpt-4"
            else:
                assert getattr(actual_model, "system", None) == "openai"
                assert getattr(actual_model, "model_name", None) in {"gpt-4", "gpt-4o", "gpt-4-0125-preview", "gpt-4-0613", "gpt-4-1106-preview", "gpt-4-turbo", "gpt-4o-mini"} or getattr(actual_model, "model_name", None).startswith("gpt-4")
            assert kwargs.get("system_prompt") == "Test instructions"
            assert kwargs.get("toolsets") == mock_get_servers.return_value
            mock_agent.iter.assert_called_once()
    
    def test_error_handling_no_model(self):
        """Test error handling when no model is specified."""
        from troop.app import create_agent_command
        
        test_command = create_agent_command("test-agent")
        
        with patch('troop.app.settings') as mock_settings:
            mock_settings.agents = {
                "test-agent": {
                    "instructions": "Test",
                    "mcp_servers": []
                    # No model specified
                }
            }
            
            runner = CliRunner()
            result = runner.invoke(test_command, ["--prompt", "Test"])
            
            assert result.exit_code == 0  # Typer doesn't exit with error by default
            assert "No model specified" in result.stdout
    
    @patch('troop.runner.Agent')
    def test_mcp_server_error_handling(self, mock_agent_class):
        """Test error handling when MCP server fails."""
        # Mock agent that raises error on context enter
        mock_agent = AsyncMock()
        mock_agent.__aenter__.side_effect = Exception("MCP server failed")
        mock_agent_class.return_value = mock_agent
        
        from troop.app import create_agent_command
        
        test_command = create_agent_command("test-agent")
        
        with patch('troop.app.settings') as mock_settings:
            mock_settings.agents = {
                "test-agent": {
                    "instructions": "Test",
                    "model": "gpt-4",
                    "mcp_servers": ["failing-server"]
                }
            }
            mock_settings.providers = {"openai": "sk-test"}
            
            with patch('troop.app.get_servers') as mock_get_servers:
                mock_get_servers.return_value = []
                
                runner = CliRunner()
                result = runner.invoke(test_command, ["--prompt", "Test"])
                
                assert result.exit_code == 1
                assert "Failed to connect to MCP server" in result.stdout
