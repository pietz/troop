# Repository Guidelines

## Project Structure & Module Organization
- `src/troop/`: CLI entry and modules (`app.py`, `commands/`, `config.py`, `display.py`, `utils.py`).
- `tests/`: Unit and integration tests with pytest; coverage config in `pytest.ini` and `.coveragerc`.
- Key entrypoints: `troop:main` (exposed via `pyproject.toml`).

## Build, Test, and Development Commands
- Run CLI locally: `uv run troop --help` (or `uv run troop researcher -p "..."`).
- Run all tests: `uv run pytest` (HTML coverage in `htmlcov/`).
- Focused tests: `uv run pytest tests/unit/` or a specific file.
- Sync env after editing deps: `uv sync`.
- Build package: `uv build` (hatchling backend).

## Coding Style & Naming Conventions
- Language: Python 3.12; prefer `list`/`dict` types and `int | None` over `Optional[int]`.
- Imports: standard lib → third‑party → local.
- CLI: Typer commands grouped under `provider`, `mcp`, and `agent`; keep new commands small and cohesive.
- Logging: keep HTTPX noise suppressed (see `src/troop/__init__.py`); add purposeful logs only.
- Filenames: snake_case; functions/methods use descriptive verbs; keep modules focused before introducing new packages.

## Testing Guidelines
- Frameworks: `pytest`, `pytest-asyncio`, `pytest-mock`, `pytest-cov`.
- Coverage: threshold enforced at 40% (`--cov-fail-under=40`). Aim higher for new code.
- Naming: files `test_*.py`, functions `test_*`, optional classes `Test*`.
- Integration tests live in `tests/integration/`; mock external processes, network, and file I/O.

## Commit & Pull Request Guidelines
- Commit style (observed): imperative, concise subjects (e.g., "add unit tests", "UI improvement including verbose mode"). Group related changes.
- PRs should include: clear description, scope of changes, linked issues, before/after notes or screenshots (for UX), and test coverage notes.
- Keep PRs small and reviewable; update README or help output if commands or flags change.

## Security & Configuration Tips
- Never commit API keys; provider and MCP secrets are read from environment variables and user config files.
- Validate CLI inputs; handle subprocess and network errors gracefully.
- Configuration paths are user-scoped; avoid writing within the repo at runtime.

