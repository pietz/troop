import yaml
from pathlib import Path
from pydantic import BaseModel

config_path = Path.home() / ".troop" / "config.yaml"


class Settings(BaseModel):
    keys: dict[str, str] = {}
    servers: dict[str, dict] = {}
    agents: dict[str, dict] = {}
    model: str | None = None
    agent: str | None = None

    def save(self):
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(self.model_dump(), f)

    @classmethod
    def load(cls):
        if config_path.exists():
            with open(config_path, "r") as f:
                return cls(**yaml.safe_load(f))
        return cls()


settings = Settings.load()
settings.save()
