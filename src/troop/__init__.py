import logging
from .app import app


def main() -> None:
    # Suppress httpx INFO logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    app()
