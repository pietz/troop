#!/usr/bin/env python3
"""Test script to verify provider environment setup"""

import os
from src.troop.utils import setup_provider_env, PROVIDER_ENV_VARS

def test_provider_env_setup():
    """Test that provider environment variables are correctly set"""
    
    # Clean up any existing env vars
    for env_var in PROVIDER_ENV_VARS.values():
        os.environ.pop(env_var, None)
    
    # Test data
    providers = {
        "openai": "sk-test-openai-key",
        "anthropic": "sk-ant-test-key",
    }
    
    # Test OpenAI
    provider = setup_provider_env("openai:gpt-4o", providers)
    assert provider == "openai"
    assert os.environ.get("OPENAI_API_KEY") == "sk-test-openai-key"
    print("✓ OpenAI provider env setup works")
    
    # Test Anthropic
    provider = setup_provider_env("anthropic:claude-3-5-sonnet", providers)
    assert provider == "anthropic"
    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-test-key"
    print("✓ Anthropic provider env setup works")
    
    # Test unknown provider
    provider = setup_provider_env("unknown:model", providers)
    assert provider is None
    print("✓ Unknown provider returns None")
    
    # Test provider not in config
    provider = setup_provider_env("gemini:1.5-pro", providers)
    assert provider is None
    assert "GEMINI_API_KEY" not in os.environ
    print("✓ Provider not in config returns None")
    
    print("\nAll tests passed! ✅")

if __name__ == "__main__":
    test_provider_env_setup()