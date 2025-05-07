import json
import os
from pathlib import Path
from typing import Any, TypeVar, Generic

T = TypeVar('T')

class Registry(Generic[T]):
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._data: dict[str, T] = {}
        self.load()
    
    def load(self) -> None:
        """Load registry data from file"""
        if self.file_path.exists():
            with open(self.file_path, 'r') as f:
                self._data = json.load(f)
        else:
            self._data = {}
            # Ensure directory exists
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.save()
    
    def save(self) -> None:
        """Save registry data to file"""
        with open(self.file_path, 'w') as f:
            json.dump(self._data, indent=2, sort_keys=True, default=str)
    
    def get(self, key: str) -> T | None:
        """Get a value from the registry"""
        return self._data.get(key)
    
    def set(self, key: str, value: T) -> None:
        """Set a value in the registry"""
        self._data[key] = value
        self.save()
    
    def remove(self, key: str) -> bool:
        """Remove a value from the registry"""
        if key in self._data:
            del self._data[key]
            self.save()
            return True
        return False
    
    def list(self) -> dict[str, T]:
        """Get all values from the registry"""
        return dict(self._data)


class Config:
    def __init__(self):
        self.config_dir = Path.home() / '.troop'
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.agents_file = self.config_dir / 'agents.json'
        self.keys_file = self.config_dir / 'keys.json'
        
        self.agents = Registry[dict[str, Any]](self.agents_file)
        self.keys = Registry[str](self.keys_file)
        
        # Initialize with default agent if empty
        if not self.agents.list():
            self.agents.set("researcher", {
                "command": "uv",
                "args": ["run", "researcher.py"],
                "description": "Web search and page loading agent"
            })
        
    def get_agent(self, name: str) -> dict[str, Any] | None:
        """Get an agent configuration by name"""
        return self.agents.get(name)
    
    def get_key(self, provider: str) -> str | None:
        """Get an API key by provider name"""
        return self.keys.get(provider)


# Singleton instance
config = Config()
