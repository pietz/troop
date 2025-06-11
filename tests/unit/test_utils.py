import pytest
import asyncio
import os
from unittest.mock import patch, AsyncMock, mock_open
from troop.utils import run_async, QuietMCPServer, get_servers
from troop.config import Settings


class TestRunAsync:
    def test_run_async_decorator(self):
        """Test run_async decorator with a simple async function."""
        @run_async
        async def async_function(value):
            await asyncio.sleep(0.01)
            return value * 2
        
        # Should be able to call directly without await
        result = async_function(5)
        assert result == 10

    def test_run_async_with_exception(self):
        """Test run_async decorator handles exceptions."""
        @run_async
        async def async_function_with_error():
            await asyncio.sleep(0.01)
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            async_function_with_error()

    def test_run_async_preserves_function_metadata(self):
        """Test run_async preserves function name and docstring."""
        @run_async
        async def my_function():
            """Test docstring"""
            return "result"
        
        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "Test docstring"


class TestQuietMCPServer:
    def test_quiet_mcp_server_inheritance(self):
        """Test QuietMCPServer inherits from MCPServerStdio."""
        from pydantic_ai.mcp import MCPServerStdio
        assert issubclass(QuietMCPServer, MCPServerStdio)

    def test_quiet_mcp_server_initialization(self):
        """Test QuietMCPServer can be initialized."""
        command = "uvx"
        args = ["mcp-test", "--arg"]
        env = {"TEST_KEY": "test_value"}
        
        server = QuietMCPServer(command=command, args=args, env=env)
        
        assert server.command == command
        assert server.args == args
        assert server.env == env

    @pytest.mark.asyncio
    async def test_quiet_mcp_server_client_streams(self):
        """Test QuietMCPServer.client_streams suppresses stderr."""
        server = QuietMCPServer(command="echo", args=["test"])
        
        # Mock the stdio_client from the mcp package
        with patch('mcp.client.stdio.stdio_client') as mock_stdio_client:
            mock_read = AsyncMock()
            mock_write = AsyncMock()
            
            # Make stdio_client an async context manager
            mock_stdio_client.return_value.__aenter__.return_value = (mock_read, mock_write)
            mock_stdio_client.return_value.__aexit__.return_value = None
            
            # Mock os.devnull
            with patch('builtins.open', mock_open()) as mock_file:
                async with server.client_streams() as (read_stream, write_stream):
                    # Verify devnull was opened
                    mock_file.assert_called_with(os.devnull, "w", encoding='utf-8')
                    
                    # Verify streams are returned
                    assert read_stream == mock_read
                    assert write_stream == mock_write


class TestGetServers:
    def test_get_servers_single_server(self):
        """Test get_servers with a single server."""
        settings = Settings(
            agents={
                "test-agent": {
                    "instructions": "Test",
                    "model": "gpt-4",
                    "servers": ["web-tools"]
                }
            },
            mcps={
                "web-tools": {
                    "command": ["uvx", "mcp-web-tools", "--port", "3000"],
                    "env": {"API_KEY": "test123"}
                }
            }
        )
        
        servers = get_servers(settings, "test-agent")
        
        assert len(servers) == 1
        assert isinstance(servers[0], QuietMCPServer)
        assert servers[0].command == "uvx"
        assert servers[0].args == ["mcp-web-tools", "--port", "3000"]

    def test_get_servers_multiple_servers(self):
        """Test get_servers with multiple servers."""
        settings = Settings(
            agents={
                "multi-agent": {
                    "instructions": "Test",
                    "model": "gpt-4",
                    "servers": ["server1", "server2", "server3"]
                }
            },
            mcps={
                "server1": {
                    "command": ["node", "server1.js"],
                    "env": {}
                },
                "server2": {
                    "command": ["python", "-m", "server2"],
                    "env": {}
                },
                "server3": {
                    "command": ["uvx", "server3"],
                    "env": {}
                }
            }
        )
        
        servers = get_servers(settings, "multi-agent")
        
        assert len(servers) == 3
        assert all(isinstance(s, QuietMCPServer) for s in servers)
        
        # Check commands
        assert servers[0].command == "node"
        assert servers[0].args == ["server1.js"]
        
        assert servers[1].command == "python"
        assert servers[1].args == ["-m", "server2"]
        
        assert servers[2].command == "uvx"
        assert servers[2].args == ["server3"]

    def test_get_servers_no_servers(self):
        """Test get_servers with agent having no servers."""
        settings = Settings(
            agents={
                "no-server-agent": {
                    "instructions": "Test",
                    "model": "gpt-4",
                    "servers": []
                }
            },
            mcps={}
        )
        
        servers = get_servers(settings, "no-server-agent")
        
        assert servers == []

    def test_get_servers_command_splitting(self):
        """Test get_servers correctly splits command into command and args."""
        settings = Settings(
            agents={
                "test": {
                    "instructions": "Test",
                    "model": "gpt-4",
                    "servers": ["complex"]
                }
            },
            mcps={
                "complex": {
                    "command": ["python", "-u", "-m", "my_module", "--flag", "value"],
                    "env": {}
                }
            }
        )
        
        servers = get_servers(settings, "test")
        
        assert len(servers) == 1
        assert servers[0].command == "python"
        assert servers[0].args == ["-u", "-m", "my_module", "--flag", "value"]