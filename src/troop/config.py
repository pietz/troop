import yaml
from pathlib import Path
from pydantic import BaseModel

config_path = Path.home() / ".troop" / "config.yaml"

# Reserved command names that cannot be used as agent names
RESERVED_NAMES = {
    "provider", "mcp", "agent",
    "help", "version"
}


class Settings(BaseModel):
    providers: dict[str, str] = {}  # API keys for LLM providers
    mcps: dict[str, dict] = {}  # MCP server configurations
    agents: dict[str, dict] = {}  # Agent configurations
    default_agent: str | None = None  # Default agent name

    def save(self):
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(self.model_dump(), f)

    @classmethod
    def load(cls):
        if config_path.exists():
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
                # Migrate old config format
                if "keys" in data:
                    data["providers"] = data.pop("keys")
                if "servers" in data:
                    data["mcps"] = data.pop("servers")
                if "agent" in data:
                    data["default_agent"] = data.pop("agent")
                if "model" in data:
                    data.pop("model")  # Remove deprecated field
                return cls(**data)
        return cls()


settings = Settings.load()
settings.save()
