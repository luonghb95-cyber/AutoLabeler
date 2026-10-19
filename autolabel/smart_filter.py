from __future__ import annotations

from dataclasses import dataclass, field

from .image_io import load_bgr
from .models import Box, ImageEntry

TTA_MATCH_IOU = 0.5


@dataclass
class SmartFilterOptions:
    conf_mode: str = "image"
    image_conf: float = 0.0
    per_class_conf: dict[str, float] = field(default_factory=dict)
    min_box_px: float = 0.0
    orphan_mode: str = "off"
    orphan_margin: float = 0.30
    orphan_small: str = "head"
    orphan_big: str = "person"
    tta_enabled: bool = False
    tta_min_votes: int = 2
    tta_conf: float = 0.25
    tta_iou: float = 0.45
    tta_imgsz: int = 640

    def active(self) -> bool:
        has_conf = (self.conf_mode == "image" and self.image_conf > 0) or (
            self.conf_mode == "box" and bool(self.per_class_conf)
        )
        has_orphan = (
            self.orphan_mode in ("safe", "aggressive")
            and self.orphan_small
            and self.orphan_big
            and self.orphan_small != self.orphan_big
        )
        return bool(has_conf or self.min_box_px > 0 or has_orphan or self.tta_enabled)


def iou(a: Box, b: Box) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a.width) * max(0.0, a.height)
    area_b = max(0.0, b.width) * max(0.0, b.height)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 1.0


def center_inside(box: Box, container: Box, margin: float) -> bool:
    cx = (box.x1 + box.x2) / 2.0
    cy = (box.y1 + box.y2) / 2.0
    mx = container.width * margin
    my = container.height * margin
    return (container.x1 - mx) <= cx <= (container.x2 + mx) and (container.y1 - my) <= cy <= (container.y2 + my)


def compute_removals(entries: list[ImageEntry], options: SmartFilterOptions) -> tuple[dict[int, set[int]], dict]:
    removals: dict[int, set[int]] = {}
    stats = {"removed": 0, "images": 0, "empties": 0}
    if not options.active():
        return removals, stats

    image_gate = options.conf_mode == "image" and options.image_conf > 0

    for index, entry in enumerate(entries):
        if not entry.boxes:
            continue
        has_big = False
        if options.orphan_mode in ("safe", "aggressive") and options.orphan_small != options.orphan_big:
            has_big = any(box.label == options.orphan_big for box in entry.boxes)

        drop: set[int] = set()
        if image_gate:
            detector_confs = [box.conf for box in entry.boxes if box.conf is not None]
            if detector_confs and max(detector_confs) < options.image_conf:
                drop.update(position for position, box in enumerate(entry.boxes) if box.conf is not None)

        for position, box in enumerate(entry.boxes):
            if box.conf is None:
                continue
            if position in drop:
                continue

            if not image_gate:
                threshold = options.per_class_conf.get(box.label)
                if threshold is not None and box.conf < threshold:
                    drop.add(position)
                    continue

            if options.min_box_px > 0 and min(box.width, box.height) < options.min_box_px:
                drop.add(position)
                continue

            if options.orphan_mode in ("safe", "aggressive") and box.label == options.orphan_small:
                if options.orphan_mode == "safe" and not has_big:
                    continue
                matched = any(
                    other.label == options.orphan_big and center_inside(box, other, options.orphan_margin)
                    for other in entry.boxes
                )
                if not matched:
                    drop.add(position)

        if drop:
            removals[index] = drop
            stats["removed"] += len(drop)
            stats["images"] += 1
            if len(drop) == len(entry.boxes):
                stats["empties"] += 1
    return removals, stats


def tta_removals(
    detector,
    entries: list[ImageEntry],
    options: SmartFilterOptions,
    progress=None,
    is_cancelled=None,
) -> dict[int, set[int]]:
    removals: dict[int, set[int]] = {}
    total = len(entries)
    names = getattr(detector, "names", [])

    for index, entry in enumerate(entries):
        if is_cancelled is not None and is_cancelled():
            break
        if progress is not None:
            progress(index, total)
        if not entry.boxes:
            continue

        image = load_bgr(entry.path)
        if image is None:
            continue
        height, width = image.shape[:2]
        try:
            detections = detector.predict(
                image[:, ::-1], options.tta_conf, options.tta_iou, options.tta_imgsz
            )
        except Exception:
            continue

        mirrored: list[Box] = []
        for det in detections:
            label = names[det.cls_id] if 0 <= det.cls_id < len(names) else f"class_{det.cls_id}"
            mirrored.append(
                Box(label=label, x1=width - det.x2, y1=det.y1, x2=width - det.x1, y2=det.y2, conf=det.conf)
            )

        drop: set[int] = set()
        for position, box in enumerate(entry.boxes):
            if box.conf is None:
                continue
            votes = 1
            for other in mirrored:
                if other.label == box.label and iou(box, other) >= TTA_MATCH_IOU:
                    votes += 1
                    break
            if votes < options.tta_min_votes:
                drop.add(position)
        if drop:
            removals[index] = drop

    if progress is not None:
        progress(total, total)
    return removals
