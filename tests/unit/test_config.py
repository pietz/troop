import pytest
from pathlib import Path
import tempfile
import yaml
from unittest.mock import patch, mock_open
from troop.config import Settings, RESERVED_NAMES, config_path


class TestSettings:
    def test_default_settings(self):
        """Test Settings with default values."""
        settings = Settings()
        assert settings.providers == {}
        assert settings.mcps == {}
        assert settings.agents == {}
        assert settings.default_agent is None

    def test_settings_with_values(self):
        """Test Settings with custom values."""
        settings = Settings(
            providers={"openai": "sk-test"},
            mcps={
                "web-tools": {
                    "command": ["uvx", "mcp-web-tools"],
                    "env": {"BRAVE_API_KEY": "test"}
                }
            },
            agents={
                "researcher": {
                    "instructions": "Test instructions",
                    "model": "openai:gpt-4",
                    "servers": ["web-tools"]
                }
            },
            default_agent="researcher"
        )
        
        assert settings.providers["openai"] == "sk-test"
        assert settings.mcps["web-tools"]["command"] == ["uvx", "mcp-web-tools"]
        assert settings.agents["researcher"]["instructions"] == "Test instructions"
        assert settings.default_agent == "researcher"

    def test_reserved_names(self):
        """Test that reserved names are properly defined."""
        expected_reserved = {"provider", "mcp", "agent", "help", "version"}
        assert RESERVED_NAMES == expected_reserved

    @patch('troop.config.config_path')
    def test_load_no_config_file(self, mock_config_path):
        """Test load() when config file doesn't exist."""
        mock_config_path.exists.return_value = False
        settings = Settings.load()
        assert isinstance(settings, Settings)
        assert settings.providers == {}
        assert settings.mcps == {}
        assert settings.agents == {}

    @patch('troop.config.config_path')
    def test_load_with_valid_config(self, mock_config_path):
        """Test load() with valid config file."""
        config_data = {
            "providers": {"openai": "sk-test"},
            "mcps": {
                "web-tools": {
                    "command": ["uvx", "mcp-web-tools"],
                    "env": {"BRAVE_API_KEY": "test"}
                }
            },
            "agents": {
                "researcher": {
                    "instructions": "Test instructions",
                    "model": "openai:gpt-4",
                    "servers": ["web-tools"]
                }
            },
            "default_agent": "researcher"
        }
        
        mock_config_path.exists.return_value = True
        mock_file_content = yaml.dump(config_data)
        
        with patch('builtins.open', mock_open(read_data=mock_file_content)):
            settings = Settings.load()
        
        assert settings.providers["openai"] == "sk-test"
        assert settings.mcps["web-tools"]["command"] == ["uvx", "mcp-web-tools"]
        assert settings.agents["researcher"]["instructions"] == "Test instructions"
        assert settings.default_agent == "researcher"

    @patch('troop.config.config_path')
    def test_load_with_migration(self, mock_config_path):
        """Test load() with old config format requiring migration."""
        # Test migration from old field names
        old_config_data = {
            "keys": {"openai": "sk-test"},  # old name for providers
            "servers": {  # old name for mcps
                "web-tools": {
                    "command": ["uvx", "mcp-web-tools"],
                    "env": {}
                }
            },
            "agents": {
                "researcher": {
                    "instructions": "Test instructions",
                    "model": "openai:gpt-4",
                    "servers": ["web-tools"]
                }
            },
            "agent": "researcher",  # old name for default_agent
            "model": "gpt-4"  # deprecated field
        }
        
        mock_config_path.exists.return_value = True
        mock_file_content = yaml.dump(old_config_data)
        
        with patch('builtins.open', mock_open(read_data=mock_file_content)):
            settings = Settings.load()
        
        # Check migration worked
        assert settings.providers["openai"] == "sk-test"
        assert settings.mcps["web-tools"]["command"] == ["uvx", "mcp-web-tools"]
        assert settings.default_agent == "researcher"

    @patch('troop.config.config_path')
    def test_load_with_invalid_yaml(self, mock_config_path):
        """Test load() with invalid YAML file."""
        mock_config_path.exists.return_value = True
        
        with patch('builtins.open', mock_open(read_data="invalid: yaml: content: [")):
            # Should return default settings on YAML error
            with pytest.raises(yaml.YAMLError):
                settings = Settings.load()

    @patch('troop.config.config_path')
    def test_save(self, mock_config_path):
        """Test save() method."""
        mock_config_path.parent.mkdir = lambda **kwargs: None
        
        settings = Settings(
            providers={"openai": "sk-test"},
            mcps={
                "web-tools": {
                    "command": ["uvx", "mcp-web-tools"],
                    "env": {"BRAVE_API_KEY": "test"}
                }
            },
            agents={
                "researcher": {
                    "instructions": "Test instructions",
                    "model": "openai:gpt-4",
                    "servers": ["web-tools"]
                }
            }
        )
        
        mock_file = mock_open()
        with patch('builtins.open', mock_file):
            settings.save()
        
        # Verify file was opened for writing
        mock_file.assert_called_once_with(mock_config_path, 'w')
        
        # Verify content was written
        handle = mock_file()
        written_data = ''.join(call.args[0] for call in handle.write.call_args_list)
        saved_data = yaml.safe_load(written_data)
        
        assert saved_data["providers"]["openai"] == "sk-test"
        assert saved_data["mcps"]["web-tools"]["command"] == ["uvx", "mcp-web-tools"]
        assert saved_data["agents"]["researcher"]["instructions"] == "Test instructions"

    @patch('troop.config.config_path')
    def test_save_creates_parent_directory(self, mock_config_path):
        """Test that save() creates parent directory if it doesn't exist."""
        mock_mkdir = patch.object(mock_config_path.parent, 'mkdir')
        
        settings = Settings(providers={"test": "key"})
        
        with mock_mkdir as mkdir_mock:
            with patch('builtins.open', mock_open()):
                settings.save()
        
        mkdir_mock.assert_called_once_with(parents=True, exist_ok=True)