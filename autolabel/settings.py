from __future__ import annotations

import copy
import json
from pathlib import Path

from .constants import APP_DIR, DEFAULT_SETTINGS

SETTINGS_FILE = APP_DIR / "settings.json"
MAX_RECENT_MODELS = 6


class Settings:
    def __init__(self):
        self.data = copy.deepcopy(DEFAULT_SETTINGS)
        self.load()

    def load(self) -> None:
        if not SETTINGS_FILE.is_file():
            return
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as handle:
                stored = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        for key, value in stored.items():
            if key == "export" and isinstance(value, dict):
                self.data["export"].update(value)
            else:
                self.data[key] = value

    def save(self) -> None:
        try:
            APP_DIR.mkdir(parents=True, exist_ok=True)
            with open(SETTINGS_FILE, "w", encoding="utf-8") as handle:
                json.dump(self.data, handle, ensure_ascii=False, indent=1)
        except OSError:
            pass

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value) -> None:
        self.data[key] = value
        self.save()

    # ---------------------------------------------------------- recent models

    def recent_models(self) -> list[str]:
        return list(self.data.get("recent_models", []))

    def remember_model(self, model_path: str) -> None:
        path = str(Path(model_path))
        recent = self.recent_models()
        recent = [item for item in recent if item.lower() != path.lower()]
        recent.insert(0, path)
        self.data["recent_models"] = recent[:MAX_RECENT_MODELS]
        self.data["last_model"] = path
        self.save()

    def existing_recent_models(self) -> list[str]:
        return [path for path in self.recent_models() if Path(path).is_file()]
