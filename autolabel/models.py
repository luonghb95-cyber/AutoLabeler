from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Box:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float | None = None

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def score(self) -> float:
        return 1.0 if self.conf is None else self.conf

    def clip(self, img_w: int, img_h: int) -> None:
        self.x1 = min(max(self.x1, 0.0), img_w)
        self.y1 = min(max(self.y1, 0.0), img_h)
        self.x2 = min(max(self.x2, 0.0), img_w)
        self.y2 = min(max(self.y2, 0.0), img_h)

    def normalized(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        cx = (self.x1 + self.x2) / 2.0 / img_w
        cy = (self.y1 + self.y2) / 2.0 / img_h
        w = self.width / img_w
        h = self.height / img_h
        return cx, cy, w, h

    def to_dict(self) -> dict:
        data = {
            "label": self.label,
            "xyxy": [round(self.x1, 2), round(self.y1, 2), round(self.x2, 2), round(self.y2, 2)],
        }
        if self.conf is not None:
            data["conf"] = round(float(self.conf), 4)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Box":
        x1, y1, x2, y2 = data["xyxy"]
        return cls(
            label=data["label"],
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
            conf=data.get("conf"),
        )


@dataclass
class ImageEntry:
    path: Path
    rel: str
    width: int = 0
    height: int = 0
    boxes: list[Box] = field(default_factory=list)
    included: bool = True
    noise: bool = False

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def has_boxes(self) -> bool:
        return len(self.boxes) > 0

    def to_dict(self) -> dict:
        data = {
            "width": self.width,
            "height": self.height,
            "included": self.included,
            "boxes": [b.to_dict() for b in self.boxes],
        }
        if self.noise:
            data["noise"] = True
        return data

    @classmethod
    def from_dict(cls, path: Path, rel: str, data: dict) -> "ImageEntry":
        entry = cls(
            path=path,
            rel=rel,
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            included=bool(data.get("included", True)),
            noise=bool(data.get("noise", False)),
        )
        entry.boxes = [Box.from_dict(b) for b in data.get("boxes", [])]
        return entry
