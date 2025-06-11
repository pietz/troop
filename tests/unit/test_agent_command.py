import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from troop.commands.agent import app
from troop.config import Settings, RESERVED_NAMES


class TestAgentCommand:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def mock_settings(self):
        """Create a mock settings object with agents and MCP servers."""
        settings = Settings(
            agents={
                "researcher": {
                    "instructions": "You are a helpful researcher",
                    "model": "openai:gpt-4",
                    "servers": ["web-tools"]
                },
                "coder": {
                    "instructions": "You are a coding assistant",
                    "model": "anthropic:claude-3",
                    "servers": ["filesystem", "git"]
                }
            },
            mcps={
                "web-tools": {
                    "command": ["uvx", "mcp-web-tools"],
                    "env": {}
                },
                "filesystem": {
                    "command": ["uvx", "mcp-filesystem"],
                    "env": {}
                },
                "git": {
                    "command": ["uvx", "mcp-git"],
                    "env": {}
                }
            },
            default_agent="researcher"
        )
        return settings

    @patch('troop.commands.agent.settings')
    def test_list_agents(self, mock_settings, runner):
        """Test listing agents."""
        mock_settings.agents = {
            "researcher": {
                "instructions": "You are a helpful researcher with web access",
                "model": "openai:gpt-4",
                "servers": ["web-tools"]
            },
            "coder": {
                "instructions": "You are a coding assistant",
                "model": "anthropic:claude-3",
                "servers": ["filesystem", "git"]
            }
        }
        
        result = runner.invoke(app, ["list"])
        
        assert result.exit_code == 0
        assert "researcher" in result.stdout
        assert "coder" in result.stdout
        assert "openai:gpt-4" in result.stdout
        assert "anthropic:claude-3" in result.stdout
        assert "web-tools" in result.stdout
        assert "filesystem, git" in result.stdout

    @patch('troop.commands.agent.settings')
    def test_list_agents_empty(self, mock_settings, runner):
        """Test listing agents when none exist."""
        mock_settings.agents = {}
        
        result = runner.invoke(app, ["list"])
        
        assert result.exit_code == 0
        # Empty table should still be shown

    @patch('troop.commands.agent.settings')
    @patch('troop.commands.agent.typer.prompt')
    def test_add_agent_interactive(self, mock_prompt, mock_settings, runner):
        """Test adding an agent interactively."""
        mock_settings.agents = {}
        mock_settings.mcps = {"web-tools": {"command": ["uvx", "mcp-web-tools"], "env": {}}}
        mock_settings.save = MagicMock()
        
        # Mock interactive prompts
        mock_prompt.side_effect = [
            "test-agent",  # name
            "openai:gpt-4",  # model
            "You are a test agent",  # instructions
            "web-tools",  # first MCP server
            "",  # no more MCP servers
        ]
        
        result = runner.invoke(app, ["add"])
        
        assert result.exit_code == 0
        assert "test-agent" in mock_settings.agents
        agent = mock_settings.agents["test-agent"]
        assert agent["instructions"] == "You are a test agent"
        assert agent["model"] == "openai:gpt-4"
        assert agent["servers"] == ["web-tools"]
        mock_settings.save.assert_called_once()

    @patch('troop.commands.agent.settings')
    @patch('troop.commands.agent.typer.prompt')
    def test_add_agent_reserved_name(self, mock_prompt, mock_settings, runner):
        """Test adding an agent with a reserved name."""
        mock_settings.agents = {}
        mock_settings.save = MagicMock()
        
        # Try to use reserved names
        for reserved_name in RESERVED_NAMES:
            mock_prompt.side_effect = [reserved_name]
            
            result = runner.invoke(app, ["add"])
            
            assert result.exit_code == 0
            assert f"'{reserved_name}' is a reserved command name" in result.stdout
            mock_settings.save.assert_not_called()

    @patch('troop.commands.agent.settings')
    @patch('troop.commands.agent.typer.prompt')
    @patch('troop.commands.agent.typer.confirm')
    def test_add_agent_duplicate(self, mock_confirm, mock_prompt, mock_settings, runner):
        """Test adding a duplicate agent."""
        mock_settings.agents = {"researcher": {"model": "old", "instructions": "old", "servers": []}}
        mock_settings.save = MagicMock()
        
        mock_prompt.side_effect = ["researcher"]
        mock_confirm.return_value = False
        
        result = runner.invoke(app, ["add"])
        
        assert result.exit_code == 0
        # Should not overwrite
        assert mock_settings.agents["researcher"]["model"] == "old"
        mock_settings.save.assert_not_called()

    @patch('troop.commands.agent.settings')
    @patch('troop.commands.agent.typer.prompt')
    def test_add_agent_non_existing_mcp_server(self, mock_prompt, mock_settings, runner):
        """Test adding an agent with non-existing MCP server."""
        mock_settings.agents = {}
        mock_settings.mcps = {}  # No MCP servers
        mock_settings.save = MagicMock()
        
        mock_prompt.side_effect = [
            "test-agent",  # name
            "gpt-4",  # model
            "Test instructions",  # instructions
            "nonexistent-server",  # non-existing MCP server
            "",  # finish
        ]
        
        result = runner.invoke(app, ["add"])
        
        assert result.exit_code == 0
        # Agent should be created with empty servers list
        agent = mock_settings.agents["test-agent"]
        assert agent["servers"] == []
        mock_settings.save.assert_called_once()

    @patch('troop.commands.agent.settings')
    def test_remove_agent_existing(self, mock_settings, runner):
        """Test removing an existing agent."""
        mock_settings.agents = {
            "researcher": {"model": "gpt-4", "instructions": "test", "servers": []},
            "coder": {"model": "claude", "instructions": "test", "servers": []}
        }
        mock_settings.save = MagicMock()
        
        result = runner.invoke(app, ["remove", "researcher"])
        
        assert result.exit_code == 0
        assert "researcher" not in mock_settings.agents
        assert "coder" in mock_settings.agents
        mock_settings.save.assert_called_once()

    @patch('troop.commands.agent.settings')
    def test_remove_agent_non_existing(self, mock_settings, runner):
        """Test removing a non-existing agent."""
        mock_settings.agents = {}
        mock_settings.save = MagicMock()
        
        result = runner.invoke(app, ["remove", "nonexistent"])
        
        assert result.exit_code == 0
        assert "No agent found" in result.stdout
        mock_settings.save.assert_not_called()

    @patch('troop.commands.agent.settings')
    def test_set_default_agent(self, mock_settings, runner):
        """Test setting a default agent."""
        mock_settings.agents = {"coder": {"model": "gpt-4", "instructions": "test", "servers": []}}
        mock_settings.default_agent = None
        mock_settings.save = MagicMock()
        
        result = runner.invoke(app, ["set", "coder"])
        
        assert result.exit_code == 0
        assert mock_settings.default_agent == "coder"
        mock_settings.save.assert_called_once()

    @patch('troop.commands.agent.settings')
    def test_set_default_agent_non_existing(self, mock_settings, runner):
        """Test setting a non-existing agent as default."""
        mock_settings.agents = {}
        mock_settings.save = MagicMock()
        
        result = runner.invoke(app, ["set", "nonexistent"])
        
        assert result.exit_code == 0
        assert "does not exist" in result.stdout
        mock_settings.save.assert_not_called()

    @patch('troop.commands.agent.settings')
    @patch('troop.commands.agent.typer.prompt')
    def test_add_agent_with_multiple_mcp_servers(self, mock_prompt, mock_settings, runner):
        """Test adding an agent with multiple MCP servers."""
        mock_settings.agents = {}
        mock_settings.mcps = {
            "server1": {"command": ["cmd1"], "env": {}},
            "server2": {"command": ["cmd2"], "env": {}},
            "server3": {"command": ["cmd3"], "env": {}}
        }
        mock_settings.save = MagicMock()
        
        mock_prompt.side_effect = [
            "multi-server-agent",  # name
            "gpt-4",  # model
            "Agent with multiple servers",  # instructions
            "server1",  # first server
            "server2",  # second server
            "server3",  # third server
            "",  # done with servers
        ]
        
        result = runner.invoke(app, ["add"])
        
        assert result.exit_code == 0
        agent = mock_settings.agents["multi-server-agent"]
        assert agent["servers"] == ["server1", "server2", "server3"]

    @patch('troop.commands.agent.settings')
    @patch('troop.commands.agent.typer.prompt')
    def test_add_agent_no_mcp_servers(self, mock_prompt, mock_settings, runner):
        """Test adding an agent without any MCP servers."""
        mock_settings.agents = {}
        mock_settings.mcps = {}
        mock_settings.save = MagicMock()
        
        mock_prompt.side_effect = [
            "simple-agent",  # name
            "gpt-4",  # model
            "Simple agent without tools",  # instructions
            "",  # no MCP servers
        ]
        
        result = runner.invoke(app, ["add"])
        
        assert result.exit_code == 0
        agent = mock_settings.agents["simple-agent"]
        assert agent["servers"] == []