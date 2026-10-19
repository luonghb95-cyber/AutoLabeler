from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile

from .constants import IMAGE_EXTENSIONS

ImageFile.LOAD_TRUNCATED_IMAGES = True


def natural_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def list_images(folder: Path, recursive: bool = False) -> list[Path]:
    if recursive:
        paths = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    else:
        paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    paths.sort(key=lambda p: natural_key(str(p.relative_to(folder))))
    return paths


def probe_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        return 0, 0


def load_bgr(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            with Image.open(path) as img:
                image = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        return image
    except Exception:
        return None


def load_rgb(path: Path) -> np.ndarray | None:
    image = load_bgr(path)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def make_thumbnail(path: Path, width: int = 64, height: int = 48) -> tuple[np.ndarray | None, tuple[int, int]]:
    try:
        with Image.open(path) as img:
            size = img.size
            thumb = img.convert("RGB")
            thumb.thumbnail((width, height), Image.BILINEAR)
            return np.array(thumb), size
    except Exception:
        return None, (0, 0)
