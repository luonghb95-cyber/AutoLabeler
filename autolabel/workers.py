from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from .anomaly import scan_anomalies
from .detector import load_detector
from .exporter import (
    CONF_BANDS,
    ExportFilters,
    ExportItem,
    confidence_zip_paths,
    delete_files,
    export_confidence_zips,
    export_zip,
)
from .image_io import list_images, load_bgr, make_thumbnail, probe_size
from .models import Box, ImageEntry
from .smart_filter import SmartFilterOptions, compute_removals, tta_removals


def numpy_rgb_to_qimage(array: np.ndarray) -> QImage:
    height, width, channels = array.shape
    image = QImage(array.data, width, height, channels * width, QImage.Format.Format_RGB888)
    return image.copy()


class FolderScanWorker(QThread):
    batch_ready = pyqtSignal(object)
    finished_scan = pyqtSignal(int)
    failed = pyqtSignal(str)

    def __init__(self, folder: Path, recursive: bool, parent=None):
        super().__init__(parent)
        self.folder = Path(folder)
        self.recursive = recursive
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            paths = list_images(self.folder, self.recursive)
            batch: list[ImageEntry] = []
            total = 0
            for path in paths:
                if self._cancelled:
                    return
                rel = str(path.relative_to(self.folder)).replace("\\", "/")
                width, height = probe_size(path)
                batch.append(ImageEntry(path=path, rel=rel, width=width, height=height))
                total += 1
                if len(batch) >= 200:
                    self.batch_ready.emit(batch)
                    batch = []
            if batch:
                self.batch_ready.emit(batch)
            self.finished_scan.emit(total)
        except Exception as exc:
            self.failed.emit(f"{exc}")


class ThumbnailWorker(QThread):
    thumb_ready = pyqtSignal(int, object)

    def __init__(self, entries: list[ImageEntry], parent=None):
        super().__init__(parent)
        self.entries = entries
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        for index, entry in enumerate(self.entries):
            if self._cancelled or self.isInterruptionRequested():
                return
            array, size = make_thumbnail(entry.path)
            if array is None:
                continue
            if entry.width == 0 and size != (0, 0):
                entry.width, entry.height = size
            self.thumb_ready.emit(index, numpy_rgb_to_qimage(array))


class ModelLoadWorker(QThread):
    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, model_path: Path, device: str, parent=None):
        super().__init__(parent)
        self.model_path = Path(model_path)
        self.device = device

    def run(self):
        try:
            detector = load_detector(self.model_path, self.device)
            self.loaded.emit(detector)
        except Exception as exc:
            self.failed.emit(f"{exc}")


class AutoLabelWorker(QThread):
    progress = pyqtSignal(int, int)
    image_labeled = pyqtSignal(int, object, float)
    finished_run = pyqtSignal(int, int)
    failed = pyqtSignal(str)

    def __init__(self, detector, entries: list[ImageEntry], indices: list[int], conf: float, iou: float, imgsz: int, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.entries = entries
        self.indices = indices
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        total_boxes = 0
        failed_images = 0
        for position, index in enumerate(self.indices):
            if self._cancelled:
                self.finished_run.emit(position, total_boxes)
                return
            entry = self.entries[index]
            started = time.perf_counter()
            try:
                image = load_bgr(entry.path)
                if image is None:
                    failed_images += 1
                    self.image_labeled.emit(index, [], 0.0)
                    continue
                detections = self.detector.predict(image, self.conf, self.iou, self.imgsz)
                boxes = self._to_boxes(detections, image.shape[1], image.shape[0])
            except Exception:
                failed_images += 1
                traceback.print_exc()
                boxes = []
            elapsed = (time.perf_counter() - started) * 1000.0
            total_boxes += len(boxes)
            self.image_labeled.emit(index, boxes, elapsed)
            self.progress.emit(position + 1, len(self.indices))
        self.finished_run.emit(len(self.indices), total_boxes)

    def _to_boxes(self, detections, img_w: int, img_h: int) -> list[Box]:
        names = getattr(self.detector, "names", [])
        boxes: list[Box] = []
        for det in detections:
            label = names[det.cls_id] if 0 <= det.cls_id < len(names) else f"class_{det.cls_id}"
            box = Box(
                label=label,
                x1=det.x1,
                y1=det.y1,
                x2=det.x2,
                y2=det.y2,
                conf=det.conf,
            )
            box.clip(img_w, img_h)
            if box.width >= 1.0 and box.height >= 1.0:
                boxes.append(box)
        return boxes


class SmartFilterWorker(QThread):
    progress = pyqtSignal(int, int)
    finished_run = pyqtSignal(int, int, int, object)
    failed = pyqtSignal(str)

    def __init__(self, entries: list[ImageEntry], options: SmartFilterOptions, detector=None, parent=None):
        super().__init__(parent)
        self.entries = entries
        self.options = options
        self.detector = detector
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            removals, _ = compute_removals(self.entries, self.options)
            if self.options.tta_enabled and self.detector is not None:
                tta_map = tta_removals(
                    self.detector,
                    self.entries,
                    self.options,
                    progress=lambda done, total: self.progress.emit(done, total),
                    is_cancelled=lambda: self._cancelled,
                )
                for index, drop in tta_map.items():
                    removals.setdefault(index, set()).update(drop)

            backup: dict[int, tuple[list[Box], bool]] = {}
            removed = affected = empties = 0
            for index in sorted(removals):
                entry = self.entries[index]
                drop = removals[index]
                if not drop:
                    continue
                backup[index] = (list(entry.boxes), entry.noise)
                entry.boxes = [box for position, box in enumerate(entry.boxes) if position not in drop]
                removed += len(drop)
                affected += 1
                if not entry.boxes:
                    entry.noise = True
                    empties += 1
                else:
                    entry.noise = False
            self.finished_run.emit(removed, affected, empties, backup)
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(f"{exc}")


class AnomalyWorker(QThread):
    progress = pyqtSignal(int, int)
    finished_scan = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, entries: list[ImageEntry], include_manual: bool = True, parent=None):
        super().__init__(parent)
        self.entries = entries
        self.include_manual = include_manual
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            hits, stats = scan_anomalies(
                self.entries,
                include_manual=self.include_manual,
                progress=lambda done, total: self.progress.emit(done, total),
                is_cancelled=lambda: self._cancelled,
            )
            if self._cancelled:
                return
            self.finished_scan.emit(hits, stats)
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(f"{exc}")


class ExportWorker(QThread):
    progress = pyqtSignal(int, int)
    done = pyqtSignal(object, int, int, int, int)
    failed = pyqtSignal(str)

    def __init__(
        self,
        zip_path: Path,
        items: list[ExportItem],
        classes: list[str],
        filters: ExportFilters,
        delete_sources: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.zip_path = Path(zip_path)
        self.items = items
        self.classes = classes
        self.filters = filters
        self.delete_sources = delete_sources
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self.filters.conf_split:
                written_by_band = export_confidence_zips(
                    self.zip_path,
                    self.items,
                    self.classes,
                    self.filters,
                    progress=lambda i, total: self.progress.emit(i, total),
                    is_cancelled=lambda: self._cancelled,
                )
                written = [path for band in CONF_BANDS for path in written_by_band[band]]
                paths = confidence_zip_paths(self.zip_path)
                summary = {
                    "conf_split": True,
                    "zip_paths": [str(paths[band]) for band in CONF_BANDS],
                    "bands": {band: len(written_by_band[band]) for band in CONF_BANDS},
                    "background": sum(1 for item in self.items if not item.boxes),
                }
            else:
                written = export_zip(
                    self.zip_path,
                    self.items,
                    self.classes,
                    self.filters,
                    progress=lambda i, total: self.progress.emit(i, total),
                    is_cancelled=lambda: self._cancelled,
                )
                summary = {
                    "conf_split": False,
                    "zip_paths": [str(self.zip_path)],
                    "bands": {},
                    "background": 0,
                }
            deleted = failed = 0
            if self.delete_sources and not self._cancelled:
                deleted, failed = delete_files(written)
            written_set = {str(path) for path in written}
            exported = [item for item in self.items if str(item.entry.path) in written_set]
            boxes = sum(len(item.boxes) for item in exported)
            self.done.emit(summary, len(written), boxes, deleted, failed)
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(f"{exc}")
