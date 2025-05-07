import os
import asyncio
from functools import wraps

from .registry import get_key


def run_async(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


def setup_api_key_env(model: str) -> None:
    """Set up API key environment variables based on the model provider"""
    provider = model.split(":")[0] if ":" in model else model

    # Map providers to environment variable names
    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    env_var = env_var_map.get(provider.lower())
    if not env_var:
        return  # Unknown provider, nothing to do

    # Check if environment variable is already set
    if env_var in os.environ and os.environ[env_var]:
        return  # Already set, nothing to do

    # Try to get API key from registry
    key = get_key(provider.lower())
    if key:
        os.environ[env_var] = key
