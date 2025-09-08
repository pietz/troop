import yaml
from pathlib import Path
from pydantic import BaseModel

config_path = Path.home() / ".troop" / "config.yaml"

# Reserved command names that cannot be used as agent names
RESERVED_NAMES = {"provider", "mcp", "agent", "help", "version"}


class Settings(BaseModel):
    providers: dict[str, str] = {}
    mcps: dict[str, dict] = {}
    agents: dict[str, dict] = {}
    default_agent: str | None = None

    def save(self):
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(self.model_dump(), f)

    @classmethod
    def load(cls):
        """Load settings from the user config file with simple migrations."""
        if not config_path.exists():
            return cls()

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        if not data:
            return cls()

        # Migrations from old config format
        if "keys" in data:
            data["providers"] = data.pop("keys")
        if "servers" in data:
            data["mcps"] = data.pop("servers")
        if "agent" in data:
            data["default_agent"] = data.pop("agent")
        if "model" in data:
            # Deprecated field; ignore
            data.pop("model")

        return cls(**data)


# Global settings instance to be imported elsewhere
settings: Settings = Settings.load()
