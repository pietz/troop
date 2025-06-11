import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from troop.commands.mcp import app
from troop.config import Settings
import shlex


class TestMCPCommand:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def mock_settings(self):
        """Create a mock settings object with MCP servers."""
        settings = Settings(
            mcps={
                "web-tools": {
                    "command": ["uvx", "mcp-web-tools"],
                    "env": {"BRAVE_API_KEY": "test123"}
                },
                "filesystem": {
                    "command": ["node", "/path/to/server.js"],
                    "env": {}
                }
            }
        )
        return settings

    @patch('troop.commands.mcp.settings')
    def test_list_mcp_servers(self, mock_settings, runner):
        """Test listing MCP servers."""
        mock_settings.mcps = {
            "web-tools": {
                "command": ["uvx", "mcp-web-tools"],
                "env": {"BRAVE_API_KEY": "test123"}
            },
            "filesystem": {
                "command": ["node", "/path/to/server.js"],
                "env": {}
            }
        }
        
        result = runner.invoke(app, ["list"])
        
        assert result.exit_code == 0
        assert "web-tools" in result.stdout
        assert "filesystem" in result.stdout
        assert "uvx mcp-web-tools" in result.stdout
        assert "node /path/to/server.js" in result.stdout

    @patch('troop.commands.mcp.settings')
    def test_list_mcp_servers_empty(self, mock_settings, runner):
        """Test listing MCP servers when none exist."""
        mock_settings.mcps = {}
        
        result = runner.invoke(app, ["list"])
        
        assert result.exit_code == 0
        # Empty table should still be shown

    @patch('troop.commands.mcp.settings')
    @patch('troop.commands.mcp.typer.prompt')
    def test_add_mcp_server_interactive(self, mock_prompt, mock_settings, runner):
        """Test adding an MCP server interactively."""
        mock_settings.mcps = {}
        mock_settings.save = MagicMock()
        
        # Mock interactive prompts
        mock_prompt.side_effect = [
            "test-server",  # name
            "uvx mcp-test --arg value",  # command
            "API_KEY=test-key",  # first env var
            "",  # no more env vars
        ]
        
        result = runner.invoke(app, ["add"])
        
        assert result.exit_code == 0
        assert "test-server" in mock_settings.mcps
        server = mock_settings.mcps["test-server"]
        assert server["command"] == ["uvx", "mcp-test", "--arg", "value"]
        assert server["env"] == {"API_KEY": "test-key"}
        mock_settings.save.assert_called_once()

    @patch('troop.commands.mcp.settings')
    @patch('troop.commands.mcp.typer.prompt')
    @patch('troop.commands.mcp.typer.confirm')
    def test_add_mcp_server_duplicate(self, mock_confirm, mock_prompt, mock_settings, runner):
        """Test adding a duplicate MCP server."""
        mock_settings.mcps = {"web-tools": {"command": ["old"], "env": {}}}
        mock_settings.save = MagicMock()
        
        mock_prompt.side_effect = ["web-tools"]
        mock_confirm.return_value = False
        
        result = runner.invoke(app, ["add"])
        
        assert result.exit_code == 0
        # Should not overwrite
        assert mock_settings.mcps["web-tools"]["command"] == ["old"]
        mock_settings.save.assert_not_called()

    @patch('troop.commands.mcp.settings')
    def test_remove_mcp_server_existing(self, mock_settings, runner):
        """Test removing an existing MCP server."""
        mock_settings.mcps = {
            "web-tools": {"command": ["test"], "env": {}},
            "other": {"command": ["other"], "env": {}}
        }
        mock_settings.save = MagicMock()
        
        result = runner.invoke(app, ["remove", "web-tools"])
        
        assert result.exit_code == 0
        assert "web-tools" not in mock_settings.mcps
        assert "other" in mock_settings.mcps
        mock_settings.save.assert_called_once()

    @patch('troop.commands.mcp.settings')
    def test_remove_mcp_server_non_existing(self, mock_settings, runner):
        """Test removing a non-existing MCP server."""
        mock_settings.mcps = {}
        mock_settings.save = MagicMock()
        
        result = runner.invoke(app, ["remove", "nonexistent"])
        
        assert result.exit_code == 0
        assert "No MCP server found" in result.stdout
        mock_settings.save.assert_not_called()

    @patch('troop.commands.mcp.settings')
    @patch('troop.commands.mcp.typer.prompt')
    def test_add_mcp_server_multiple_env_vars(self, mock_prompt, mock_settings, runner):
        """Test adding MCP server with multiple environment variables."""
        mock_settings.mcps = {}
        mock_settings.save = MagicMock()
        
        mock_prompt.side_effect = [
            "multi-env",  # name
            "uvx mcp-multi",  # command
            "API_KEY=key1",  # first env var
            "SECRET=secret1",  # second env var
            "DEBUG=true",  # third env var
            "",  # done
        ]
        
        result = runner.invoke(app, ["add"])
        
        assert result.exit_code == 0
        server = mock_settings.mcps["multi-env"]
        assert server["env"] == {
            "API_KEY": "key1",
            "SECRET": "secret1",
            "DEBUG": "true"
        }

    @patch('troop.commands.mcp.settings')
    @patch('troop.commands.mcp.typer.prompt')
    def test_add_mcp_server_invalid_env_format(self, mock_prompt, mock_settings, runner):
        """Test adding MCP server with invalid env var format."""
        mock_settings.mcps = {}
        mock_settings.save = MagicMock()
        
        mock_prompt.side_effect = [
            "test-server",  # name
            "uvx mcp-test",  # command
            "INVALID_FORMAT",  # invalid env var (no =)
            "VALID=value",  # valid env var
            "",  # done
        ]
        
        result = runner.invoke(app, ["add"])
        
        assert result.exit_code == 0
        server = mock_settings.mcps["test-server"]
        # Only valid env var should be added
        assert server["env"] == {"VALID": "value"}

    def test_shlex_command_parsing(self):
        """Test that commands are parsed correctly with shlex."""
        test_cases = [
            ("ls -la", ["ls", "-la"]),
            ("python script.py", ["python", "script.py"]),
            ('echo "hello world"', ["echo", "hello world"]),
            ("cmd --arg='value with spaces'", ["cmd", "--arg=value with spaces"]),
        ]
        
        for command, expected in test_cases:
            assert shlex.split(command) == expected