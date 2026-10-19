from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .constants import PROJECTS_DIR
from .models import ImageEntry

PROJECT_FILE_VERSION = 1


class ProjectStore:
    def __init__(self, image_dir: Path):
        self.image_dir = Path(image_dir).resolve()
        digest = hashlib.sha1(str(self.image_dir).lower().encode("utf-8")).hexdigest()[:16]
        self.root = PROJECTS_DIR / digest
        self.file = self.root / "project.json"

    def exists(self) -> bool:
        return self.file.is_file()

    def load(self) -> dict | None:
        if not self.exists():
            return None
        try:
            with open(self.file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if data.get("version") != PROJECT_FILE_VERSION:
                return None
            return data
        except (OSError, json.JSONDecodeError):
            return None

    def load_entries(self) -> tuple[list[ImageEntry], dict]:
        data = self.load()
        if data is None:
            return [], {}
        entries: list[ImageEntry] = []
        for rel, payload in data.get("images", {}).items():
            path = self.image_dir / rel
            entries.append(ImageEntry.from_dict(path, rel, payload))
        return entries, data

    def save(self, entries: list[ImageEntry], classes: list[str], settings: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": PROJECT_FILE_VERSION,
            "image_dir": str(self.image_dir),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "classes": classes,
            "settings": settings,
            "images": {entry.rel: entry.to_dict() for entry in entries},
        }
        temp = self.file.with_suffix(".tmp")
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, self.file)
