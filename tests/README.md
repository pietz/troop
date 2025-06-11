# Troop Test Suite

This directory contains the comprehensive test suite for the Troop CLI tool.

## Structure

- `unit/` - Unit tests for individual modules and components
- `integration/` - Integration tests for end-to-end functionality
- `fixtures/` - Test data and fixtures
- `conftest.py` - Shared pytest configuration and fixtures

## Running Tests

```bash
# Run all tests
uv run pytest

# Run unit tests only
uv run pytest tests/unit/

# Run with coverage report
uv run pytest --cov=troop --cov-report=term-missing

# Run specific test file
uv run pytest tests/unit/test_config.py

# Run with verbose output
uv run pytest -v
```

## Test Coverage

The test suite aims for high coverage of all critical functionality:

- **Config Management**: Loading, saving, and migrating settings
- **CLI Commands**: Provider, MCP, and Agent management commands
- **Utilities**: Async decorators, MCP server wrappers
- **Agent Execution**: Interactive REPL and single-prompt modes

Current coverage target: 40% (exceeded with 87%+)

## Writing Tests

When adding new features, please include corresponding tests:

1. Unit tests for new functions/methods
2. Integration tests for new CLI commands
3. Mock external dependencies (API calls, file I/O, subprocesses)
4. Use fixtures from `conftest.py` for common test data

## Test Dependencies

- pytest: Core testing framework
- pytest-asyncio: Async test support
- pytest-mock: Mocking utilities
- pytest-cov: Coverage reporting