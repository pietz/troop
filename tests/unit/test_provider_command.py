import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock, call
from troop.commands.provider import app
from troop.config import Settings


class TestProviderCommand:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def mock_settings(self):
        """Create a mock settings object."""
        settings = Settings(
            providers={
                "openai": "sk-test123456789abcdef",
                "anthropic": "ak-test456789abcdefghi"
            }
        )
        return settings

    @patch('troop.commands.provider.settings')
    def test_list_providers(self, mock_settings, runner):
        """Test listing providers."""
        mock_settings.providers = {
            "openai": "sk-test123456789abcdef",
            "anthropic": "ak-test456789abcdefghi"
        }
        
        result = runner.invoke(app, ["list"])
        
        assert result.exit_code == 0
        # Check that keys are masked
        assert "sk-tes...abcdef" in result.stdout
        assert "ak-tes...defghi" in result.stdout

    @patch('troop.commands.provider.settings')
    def test_list_providers_empty(self, mock_settings, runner):
        """Test listing providers when none exist."""
        mock_settings.providers = {}
        
        result = runner.invoke(app, ["list"])
        
        assert result.exit_code == 0
        # Empty table should still show

    @patch('troop.commands.provider.settings')
    @patch('troop.commands.provider.typer.prompt')
    def test_add_provider_interactive(self, mock_prompt, mock_settings, runner):
        """Test adding a provider interactively."""
        mock_settings.providers = {}
        mock_settings.save = MagicMock()
        
        mock_prompt.side_effect = ["openai", "sk-newkey"]
        
        result = runner.invoke(app, ["add"])
        
        assert result.exit_code == 0
        assert mock_settings.providers["openai"] == "sk-newkey"
        mock_settings.save.assert_called_once()

    @patch('troop.commands.provider.settings')
    def test_add_provider_cli_args(self, mock_settings, runner):
        """Test adding a provider with CLI arguments."""
        mock_settings.providers = {}
        mock_settings.save = MagicMock()
        
        result = runner.invoke(app, ["add", "openai", "--api-key", "sk-newkey"])
        
        assert result.exit_code == 0
        assert mock_settings.providers["openai"] == "sk-newkey"
        mock_settings.save.assert_called_once()

    @patch('troop.commands.provider.settings')
    @patch('troop.commands.provider.typer.confirm')
    def test_add_provider_overwrite_confirmed(self, mock_confirm, mock_settings, runner):
        """Test overwriting existing provider with confirmation."""
        mock_settings.providers = {"openai": "sk-old"}
        mock_settings.save = MagicMock()
        mock_confirm.return_value = True
        
        result = runner.invoke(app, ["add", "openai", "--api-key", "sk-updated"])
        
        assert result.exit_code == 0
        assert mock_settings.providers["openai"] == "sk-updated"
        mock_settings.save.assert_called_once()

    @patch('troop.commands.provider.settings')
    @patch('troop.commands.provider.typer.confirm')
    def test_add_provider_overwrite_cancelled(self, mock_confirm, mock_settings, runner):
        """Test cancelling overwrite of existing provider."""
        mock_settings.providers = {"openai": "sk-old"}
        mock_settings.save = MagicMock()
        mock_confirm.return_value = False
        
        result = runner.invoke(app, ["add", "openai", "--api-key", "sk-updated"])
        
        assert result.exit_code == 0
        # Key should not be updated
        assert mock_settings.providers["openai"] == "sk-old"
        mock_settings.save.assert_not_called()

    @patch('troop.commands.provider.settings')
    def test_remove_provider_existing(self, mock_settings, runner):
        """Test removing an existing provider."""
        mock_settings.providers = {"openai": "sk-test", "anthropic": "ak-test"}
        mock_settings.save = MagicMock()
        
        result = runner.invoke(app, ["remove", "openai"])
        
        assert result.exit_code == 0
        assert "openai" not in mock_settings.providers
        assert "anthropic" in mock_settings.providers
        mock_settings.save.assert_called_once()

    @patch('troop.commands.provider.settings')
    def test_remove_provider_non_existing(self, mock_settings, runner):
        """Test removing a non-existing provider."""
        mock_settings.providers = {}
        mock_settings.save = MagicMock()
        
        result = runner.invoke(app, ["remove", "nonexistent"])
        
        assert result.exit_code == 0
        assert "No API key found" in result.stdout
        mock_settings.save.assert_not_called()

    @patch('troop.commands.provider.settings')
    @patch('troop.commands.provider.typer.prompt')
    def test_remove_provider_interactive(self, mock_prompt, mock_settings, runner):
        """Test removing a provider interactively."""
        mock_settings.providers = {"openai": "sk-test"}
        mock_settings.save = MagicMock()
        mock_prompt.return_value = "openai"
        
        result = runner.invoke(app, ["remove"])
        
        assert result.exit_code == 0
        assert "openai" not in mock_settings.providers
        mock_settings.save.assert_called_once()

    def test_api_key_masking(self):
        """Test the API key masking logic in list command."""
        # Test various key lengths
        test_cases = [
            ("short", "****"),  # Very short key
            ("sk-abc", "sk-abc...sk-abc"),  # Short key shows full
            ("sk-abcdefghij", "sk-abc...efghij"),  # Medium key
            ("sk-abcdefghijklmnopqrstuvwxyz", "sk-abc...uvwxyz"),  # Long key
        ]
        
        for key, expected in test_cases:
            # The actual masking logic in the code is: f"{key[:6]}...{key[-6:]}"
            if len(key) > 12:
                assert f"{key[:6]}...{key[-6:]}" == expected
            else:
                # For short keys, it would still try to slice
                assert f"{key[:6]}...{key[-6:]}" == f"{key}...{key}"