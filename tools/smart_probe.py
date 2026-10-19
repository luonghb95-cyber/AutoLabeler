"""Offscreen probe for Smart Filter: dialog render, apply, undo, TTA."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QEventLoop, QTimer
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import QApplication, QMessageBox

FONT_DIR = Path("C:/Windows/Fonts")
FONT_FILES = ["segoeui.ttf", "segoeuib.ttf", "segoeuisb.ttf", "seguisb.ttf", "seguibl.ttf"]

from autolabel.app_state import AppState
from autolabel.settings import Settings
from autolabel.theme import apply_theme
from autolabel.ui.main_window import MainWindow
from autolabel.ui.smart_filter_dialog import SmartFilterDialog
from autolabel.smart_filter import SmartFilterOptions
from autolabel.workers import SmartFilterWorker

IMAGES = Path(r"C:\Users\Admin\Desktop\label\exp-22\images")
MODEL = Path(r"C:\Users\Admin\Desktop\label\exp-22.onnx")
OUT_DIR = ROOT / "tools" / "shots"
ORIGINAL_BOXES = 159


def wait_until(cond, timeout_s: float = 30.0, label: str = "") -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if cond():
            return True
        time.sleep(0.03)
    print(f"TIMEOUT waiting for {label or cond}")
    return False


def run_worker(entries, options, detector) -> tuple:
    result = {}

    def on_finished(removed, affected, empties, backup):
        result.update(removed=removed, affected=affected, empties=empties, backup=backup)

    worker = SmartFilterWorker(entries, options, detector)
    worker.finished_run.connect(on_finished)
    worker.failed.connect(lambda message: result.update(error=message))
    worker.start()
    wait_until(lambda: bool(result), 300.0, "smart filter worker")
    if "error" in result:
        print("WORKER ERROR:", result["error"])
    return result.get("removed", 0), result.get("affected", 0), result.get("empties", 0), result.get("backup")


def total_boxes(state) -> int:
    return sum(len(entry.boxes) for entry in state.entries)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.critical = staticmethod(lambda *a, **k: print("CRITICAL:", a))

    app = QApplication(sys.argv)
    for name in FONT_FILES:
        path = FONT_DIR / name
        if path.is_file():
            QFontDatabase.addApplicationFont(str(path))
    apply_theme(app)

    settings = Settings()
    state = AppState(settings)
    window = MainWindow(state)
    window.show()
    page = window.annotate_page

    settle_deadline = time.time() + 0.8
    while time.time() < settle_deadline:
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        time.sleep(0.03)

    page.open_folder(IMAGES, silent=True)
    if not wait_until(lambda: state.has_folder() and len(state.entries) > 0, 60, "folder scan"):
        return 1
    print(f"entries: {len(state.entries)}  boxes: {total_boxes(state)}")

    page.load_model(MODEL)
    if not wait_until(lambda: state.detector is not None, 120, "model load"):
        return 1
    print(f"model: {state.detector.describe()}  classes: {state.classes}")

    dialog = SmartFilterDialog(state, page)
    dialog.show()
    QApplication.processEvents()
    time.sleep(0.3)
    QApplication.processEvents()
    dialog_path = OUT_DIR / "12_smart_filter_dialog.png"
    dialog.grab().save(str(dialog_path))
    print("saved", dialog_path)
    dialog.reject()
    QApplication.processEvents()

    counts_before = [len(entry.boxes) for entry in state.entries]
    weak_indices = [
        index
        for index, entry in enumerate(state.entries)
        if entry.boxes and max(box.conf for box in entry.boxes if box.conf is not None) < 0.40
    ]
    weak_boxes = sum(counts_before[index] for index in weak_indices)
    print(f"weak images (max conf < 0.40): {len(weak_indices)}  boxes in them: {weak_boxes}")

    conf_only = SmartFilterOptions(conf_mode="image", image_conf=0.40)
    removed, affected, empties, backup = run_worker(state.entries, conf_only, None)
    print(f"[A] conf-only: removed={removed} affected={affected} empties={empties} (expected removed={weak_boxes})")
    strong_ok = all(
        len(state.entries[index].boxes) == counts_before[index]
        for index in range(len(state.entries))
        if index not in weak_indices
    )
    print(f"[A] strong images untouched: {'OK' if strong_ok else 'MISMATCH'}")
    page._filter_backup = backup
    page._undo_smart_filter()
    print(f"[A] boxes after undo: {total_boxes(state)} (expected {ORIGINAL_BOXES})")

    options = SmartFilterOptions(
        conf_mode="image",
        image_conf=0.40,
        min_box_px=6.0,
        orphan_mode="safe",
        orphan_margin=0.30,
        orphan_small="head",
        orphan_big="person",
    )
    removed, affected, empties, backup = run_worker(state.entries, options, None)
    print(f"heuristic filter: removed={removed} affected={affected} empties={empties} backup_entries={len(backup or {})}")
    print(f"boxes after filter: {total_boxes(state)} (expected {ORIGINAL_BOXES - removed})")
    state.annotations_changed.emit()
    QApplication.processEvents()
    time.sleep(0.4)
    QApplication.processEvents()
    page_path = OUT_DIR / "13_smart_filter_applied.png"
    window.grab().save(str(page_path))
    print("saved", page_path)

    page._filter_backup = backup
    page._undo_smart_filter()
    restored = total_boxes(state)
    print(f"boxes after undo: {restored} (expected {ORIGINAL_BOXES})  {'OK' if restored == ORIGINAL_BOXES else 'MISMATCH'}")

    tta_options = SmartFilterOptions(
        conf_mode="image",
        image_conf=0.40,
        min_box_px=6.0,
        orphan_mode="safe",
        orphan_margin=0.30,
        orphan_small="head",
        orphan_big="person",
        tta_enabled=True,
    )
    started = time.time()
    removed, affected, empties, backup = run_worker(state.entries, tta_options, state.detector)
    print(f"tta filter: removed={removed} affected={affected} empties={empties} elapsed={time.time() - started:.1f}s")
    print(f"boxes after tta: {total_boxes(state)}")
    page._filter_backup = backup
    page._undo_smart_filter()
    restored = total_boxes(state)
    print(f"boxes after undo: {restored} (expected {ORIGINAL_BOXES})  {'OK' if restored == ORIGINAL_BOXES else 'MISMATCH'}")

    state.dirty = False
    window.close()
    QApplication.processEvents()
    return 0


if __name__ == "__main__":
    sys.exit(main())
