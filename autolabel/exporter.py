from __future__ import annotations

import os
import random
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .models import Box, ImageEntry


@dataclass
class ExportFilters:
    min_conf: float = 0.25
    max_conf: float = 1.0
    use_size: bool = False
    min_width: int = 0
    max_width: int = 100000
    min_height: int = 0
    max_height: int = 100000
    skip_empty: bool = True
    selected_classes: list[str] | None = None
    split_enabled: bool = False
    val_ratio: float = 0.2
    seed: int = 42
    conf_split: bool = False
    conf_low: float = 0.3
    conf_high: float = 0.7

    @classmethod
    def from_dict(cls, data: dict) -> "ExportFilters":
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in fields})

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


@dataclass
class ExportItem:
    entry: ImageEntry
    boxes: list[Box]
    split: str = "train"


@dataclass
class ExportStats:
    total_images: int = 0
    matched_images: int = 0
    skipped_size: int = 0
    skipped_empty: int = 0
    total_boxes: int = 0
    exported_boxes: int = 0
    classes_used: list[str] = field(default_factory=list)


def _box_score(box: Box) -> float:
    return box.score


CONF_BANDS = ("high", "medium", "low")


def item_confidence(item: ExportItem) -> float:
    return min((box.score for box in item.boxes), default=0.0)


def confidence_band(score: float, filters: ExportFilters) -> str:
    if score > filters.conf_high:
        return "high"
    if score >= filters.conf_low:
        return "medium"
    return "low"


def item_band(item: ExportItem, filters: ExportFilters) -> str:
    if not item.boxes:
        return "low"
    return confidence_band(item_confidence(item), filters)


def confidence_zip_paths(zip_path: Path) -> dict[str, Path]:
    zip_path = Path(zip_path)
    return {band: zip_path.with_name(f"{zip_path.stem}_{band}.zip") for band in CONF_BANDS}


def apply_filters(entries: list[ImageEntry], filters: ExportFilters) -> tuple[list[ExportItem], ExportStats]:
    stats = ExportStats(total_images=len(entries))
    items: list[ExportItem] = []
    used_classes: set[str] = set()

    for entry in entries:
        stats.total_boxes += len(entry.boxes)
        if not entry.included:
            continue

        if filters.use_size:
            if not (filters.min_width <= entry.width <= filters.max_width):
                stats.skipped_size += 1
                continue
            if not (filters.min_height <= entry.height <= filters.max_height):
                stats.skipped_size += 1
                continue

        kept: list[Box] = []
        for box in entry.boxes:
            if not filters.conf_split:
                score = _box_score(box)
                if score < filters.min_conf or score > filters.max_conf:
                    continue
            if filters.selected_classes is not None and box.label not in filters.selected_classes:
                continue
            kept.append(box)

        if not kept and filters.skip_empty and not filters.conf_split:
            stats.skipped_empty += 1
            continue

        stats.exported_boxes += len(kept)
        used_classes.update(box.label for box in kept)
        items.append(ExportItem(entry=entry, boxes=kept))

    stats.matched_images = len(items)
    stats.classes_used = sorted(used_classes)
    return items, stats


def _assign_splits(items: list[ExportItem], filters: ExportFilters) -> None:
    if not filters.split_enabled:
        for item in items:
            item.split = "train"
        return

    indices = list(range(len(items)))
    random.Random(filters.seed).shuffle(indices)
    val_count = max(1, int(round(len(items) * filters.val_ratio))) if len(items) > 1 else 0
    val_indices = set(indices[:val_count])
    for i, item in enumerate(items):
        item.split = "val" if i in val_indices else "train"


def _label_lines(boxes: list[Box], class_ids: dict[str, int], width: int, height: int) -> list[str]:
    lines: list[str] = []
    for box in boxes:
        cls_id = class_ids.get(box.label)
        if cls_id is None:
            continue
        cx, cy, w, h = box.normalized(width, height)
        if w <= 0 or h <= 0:
            continue
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def _unique_name(name: str, used: set[str]) -> str:
    if name not in used:
        used.add(name)
        return name
    stem = Path(name).stem
    suffix = Path(name).suffix
    counter = 2
    candidate = f"{stem}_{counter}{suffix}"
    while candidate in used:
        counter += 1
        candidate = f"{stem}_{counter}{suffix}"
    used.add(candidate)
    return candidate


def build_data_yaml(classes: list[str], filters: ExportFilters) -> str:
    names = {i: name for i, name in enumerate(classes)}
    data = {
        "path": ".",
        "nc": len(classes),
        "names": names,
    }
    if filters.split_enabled:
        data["train"] = "images/train"
        data["val"] = "images/val"
    else:
        data["train"] = "images"
        data["val"] = "images"
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def export_zip(
    zip_path: Path,
    items: list[ExportItem],
    classes: list[str],
    filters: ExportFilters,
    progress=None,
    is_cancelled=None,
) -> list[Path]:
    class_ids = {name: i for i, name in enumerate(classes)}
    _assign_splits(items, filters)

    used_names: set[str] = set()
    written: list[Path] = []
    total = len(items) + 1

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data.yaml", build_data_yaml(classes, filters))
        if progress:
            progress(1, total)

        for index, item in enumerate(items, start=1):
            if is_cancelled and is_cancelled():
                break
            split = item.split
            if filters.split_enabled:
                image_dir = f"images/{split}"
                label_dir = f"labels/{split}"
            else:
                image_dir = "images"
                label_dir = "labels"

            name = _unique_name(item.entry.name, used_names)
            image_arc = f"{image_dir}/{name}"
            label_arc = f"{label_dir}/{Path(name).stem}.txt"

            lines = _label_lines(item.boxes, class_ids, item.entry.width, item.entry.height)
            try:
                archive.write(item.entry.path, image_arc)
            except OSError:
                if progress:
                    progress(index + 1, total)
                continue
            archive.writestr(label_arc, "\n".join(lines) + ("\n" if lines else ""))
            written.append(Path(item.entry.path))

            if progress:
                progress(index + 1, total)

    return written


def export_confidence_zips(
    zip_path: Path,
    items: list[ExportItem],
    classes: list[str],
    filters: ExportFilters,
    progress=None,
    is_cancelled=None,
) -> dict[str, list[Path]]:
    class_ids = {name: i for i, name in enumerate(classes)}
    grouped: dict[str, list[ExportItem]] = {band: [] for band in CONF_BANDS}
    for item in items:
        grouped[item_band(item, filters)].append(item)

    paths = confidence_zip_paths(zip_path)
    written: dict[str, list[Path]] = {band: [] for band in CONF_BANDS}
    total = len(items)
    done = 0

    for band in CONF_BANDS:
        band_items = grouped[band]
        _assign_splits(band_items, filters)
        paths[band].parent.mkdir(parents=True, exist_ok=True)
        used_names: set[str] = set()
        with zipfile.ZipFile(paths[band], "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("data.yaml", build_data_yaml(classes, filters))
            for item in band_items:
                if is_cancelled and is_cancelled():
                    return written
                if filters.split_enabled:
                    image_dir = f"images/{item.split}"
                    label_dir = f"labels/{item.split}"
                else:
                    image_dir = "images"
                    label_dir = "labels"

                name = _unique_name(item.entry.name, used_names)
                image_arc = f"{image_dir}/{name}"
                label_arc = f"{label_dir}/{Path(name).stem}.txt"

                try:
                    archive.write(item.entry.path, image_arc)
                except OSError:
                    done += 1
                    if progress:
                        progress(done, max(1, total))
                    continue
                lines = _label_lines(item.boxes, class_ids, item.entry.width, item.entry.height)
                archive.writestr(label_arc, "\n".join(lines) + ("\n" if lines else ""))
                written[band].append(Path(item.entry.path))

                done += 1
                if progress:
                    progress(done, max(1, total))

    return written


def delete_files(paths: list[Path]) -> tuple[int, int]:
    """Delete files, preferring the Windows Recycle Bin. Returns (deleted, failed)."""
    existing = [Path(path) for path in paths if Path(path).is_file()]
    if not existing:
        return 0, 0

    if os.name == "nt":
        try:
            _recycle_windows(existing)
        except Exception:
            pass

    remaining = []
    for path in existing:
        if path.exists():
            remaining.append(path)

    failed = 0
    for path in remaining:
        try:
            path.unlink()
        except OSError:
            failed += 1

    return len(existing) - failed, failed


def _recycle_windows(paths: list[Path]) -> None:
    import ctypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [
            ("hwnd", ctypes.c_void_p),
            ("wFunc", ctypes.c_uint),
            ("pFrom", ctypes.c_wchar_p),
            ("pTo", ctypes.c_wchar_p),
            ("fFlags", ctypes.c_uint16),
            ("fAnyOperationsAborted", ctypes.c_int),
            ("hNameMappings", ctypes.c_void_p),
            ("lpszProgressTitle", ctypes.c_wchar_p),
        ]

    FO_DELETE = 3
    FOF_SILENT = 0x0004
    FOF_NOCONFIRMATION = 0x0010
    FOF_ALLOWUNDO = 0x0040
    FOF_NOERRORUI = 0x0400

    source = "\0".join(str(path) for path in paths) + "\0\0"
    operation = SHFILEOPSTRUCTW()
    operation.wFunc = FO_DELETE
    operation.pFrom = source
    operation.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI
    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation))
    if result != 0:
        raise OSError(f"SHFileOperation failed with code {result}")
