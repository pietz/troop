import logging


def main() -> None:
    # Suppress httpx INFO logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Import the Typer app lazily to avoid side effects at package import time
    from .app import app

    app()
