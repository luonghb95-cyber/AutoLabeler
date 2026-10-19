from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QColor

from .constants import CLASS_COLORS
from .models import ImageEntry
from .project import ProjectStore
from .settings import Settings


class AppState(QObject):
    entries_changed = pyqtSignal()
    image_changed = pyqtSignal(int)
    classes_changed = pyqtSignal()
    model_changed = pyqtSignal()
    annotations_changed = pyqtSignal()
    status_message = pyqtSignal(str, int)
    busy_changed = pyqtSignal(bool, str)
    folder_reload_requested = pyqtSignal()

    def __init__(self, settings: Settings, parent: QObject | None = None):
        super().__init__(parent)
        self.settings = settings
        self.image_dir: Path | None = None
        self.entries: list[ImageEntry] = []
        self.classes: list[str] = []
        self.current_index = -1
        self.detector = None
        self.detector_path: Path | None = None
        self.project: ProjectStore | None = None
        self._dirty = False
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(1200)
        self._save_timer.timeout.connect(self.save_project)

    # --------------------------------------------------------------- project

    def has_folder(self) -> bool:
        return self.image_dir is not None and len(self.entries) > 0

    def set_folder(self, image_dir: Path, entries: list[ImageEntry]) -> None:
        self.image_dir = Path(image_dir)
        self.entries = entries
        self.project = ProjectStore(self.image_dir)
        self.current_index = 0 if entries else -1
        saved_classes = []
        stored = self.project.load()
        if stored:
            saved_classes = [str(name) for name in stored.get("classes", [])]
        self.classes = self._merge_classes(saved_classes, self.model_class_names())
        self.entries_changed.emit()
        self.classes_changed.emit()
        self.image_changed.emit(self.current_index)
        self.mark_dirty()

    def set_current(self, index: int) -> None:
        if 0 <= index < len(self.entries):
            self.current_index = index
        elif not self.entries:
            self.current_index = -1
        else:
            self.current_index = max(0, min(index, len(self.entries) - 1))
        self.image_changed.emit(self.current_index)

    def current_entry(self) -> ImageEntry | None:
        if 0 <= self.current_index < len(self.entries):
            return self.entries[self.current_index]
        return None

    def mark_dirty(self) -> None:
        self._dirty = True
        self._save_timer.start()

    def save_project(self) -> None:
        if self.project is None or not self._dirty:
            return
        try:
            self.project.save(self.entries, self.classes, self.settings.data)
            self._dirty = False
        except OSError as exc:
            self.status_message.emit(f"Could not save project: {exc}", 5000)

    # --------------------------------------------------------------- classes

    def model_class_names(self) -> list[str]:
        if self.detector is None:
            return []
        return [str(name) for name in getattr(self.detector, "names", [])]

    @staticmethod
    def _merge_classes(existing: list[str], model_names: list[str]) -> list[str]:
        merged = list(model_names)
        for name in existing:
            if name not in merged:
                merged.append(name)
        return merged

    def set_detector(self, detector, path: Path) -> None:
        self.detector = detector
        self.detector_path = Path(path)
        self.classes = self._merge_classes(self.classes, self.model_class_names())
        self.model_changed.emit()
        self.classes_changed.emit()

    def add_class(self, name: str) -> bool:
        name = name.strip()
        if not name or name in self.classes:
            return False
        self.classes.append(name)
        self.classes_changed.emit()
        self.mark_dirty()
        return True

    def remove_class(self, name: str) -> bool:
        in_use = any(box.label == name for entry in self.entries for box in entry.boxes)
        if in_use or name not in self.classes or len(self.classes) <= 1:
            return False
        self.classes.remove(name)
        self.classes_changed.emit()
        self.mark_dirty()
        return True

    def class_index(self, label: str) -> int:
        try:
            return self.classes.index(label)
        except ValueError:
            return -1

    def color_for(self, label: str) -> QColor:
        index = self.class_index(label)
        if index < 0:
            index = abs(hash(label)) % len(CLASS_COLORS)
        return QColor(CLASS_COLORS[index % len(CLASS_COLORS)])

    def active_class(self) -> str | None:
        if not self.classes:
            return None
        index = int(self.settings.get("active_class", 0))
        index = max(0, min(index, len(self.classes) - 1))
        return self.classes[index]

    def set_active_class(self, name: str) -> None:
        index = self.class_index(name)
        if index >= 0:
            self.settings.set("active_class", index)

    def counts(self) -> tuple[int, int, int]:
        total_boxes = sum(len(entry.boxes) for entry in self.entries)
        labeled = sum(1 for entry in self.entries if entry.has_boxes)
        return len(self.entries), labeled, total_boxes
