"""Offscreen probe for Find Unusual: scan ranking, marks, dialog render."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QEventLoop
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import QApplication, QMessageBox

FONT_DIR = Path("C:/Windows/Fonts")
FONT_FILES = ["segoeui.ttf", "segoeuib.ttf", "segoeuisb.ttf", "seguisb.ttf", "seguibl.ttf"]

from autolabel.app_state import AppState
from autolabel.settings import Settings
from autolabel.theme import apply_theme
from autolabel.ui.main_window import MainWindow

IMAGES = Path(r"C:\Users\Admin\Desktop\label\exp-22\images")
OUT_DIR = ROOT / "tools" / "shots"


def wait_until(cond, timeout_s: float = 30.0, label: str = "") -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if cond():
            return True
        time.sleep(0.03)
    print(f"TIMEOUT waiting for {label or cond}")
    return False


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
    boxes = sum(len(entry.boxes) for entry in state.entries)
    print(f"entries: {len(state.entries)}  boxes: {boxes}")

    started = time.time()
    page.open_anomaly_dialog()
    page._run_anomaly_scan(True)
    if not wait_until(
        lambda: page._anomaly_worker is not None and not page._anomaly_worker.isRunning() and bool(page._anomaly_hits),
        300,
        "anomaly scan",
    ):
        return 1
    elapsed = time.time() - started
    hits = page._anomaly_hits
    print(f"scanned {page._anomaly_stats.get('boxes')} boxes in {elapsed:.1f}s")

    for threshold in (2.5, 3.0, 3.5, 4.0):
        flagged = [hit for hit in hits if hit.score >= threshold]
        images = len({hit.entry_index for hit in flagged})
        print(f"threshold {threshold}: {len(flagged)} boxes in {images} images")

    print("\ntop 20 unusual boxes:")
    for rank, hit in enumerate(hits[:20], 1):
        entry = state.entries[hit.entry_index]
        conf = "manual" if hit.conf is None else f"{hit.conf * 100:.0f}%"
        print(f"  {rank:2d}. score {hit.score:5.2f}  {entry.name}  {hit.label} {conf}  {' | '.join(hit.reasons)}")

    QApplication.processEvents()
    time.sleep(0.3)
    QApplication.processEvents()
    dialog_path = OUT_DIR / "14_anomaly_dialog.png"
    page._anomaly_dialog.grab().save(str(dialog_path))
    print("\nsaved", dialog_path)

    top = hits[0]
    page._goto_anomaly_box(top.entry_index, top.box_index)
    wait_until(lambda: state.current_index == top.entry_index, 5, "goto image")
    QApplication.processEvents()
    time.sleep(0.4)
    QApplication.processEvents()
    marked = page.canvas._marked
    entry = state.entries[top.entry_index]
    print(f"jumped to {entry.name}: current_index={state.current_index} marks={len(marked)} selected={page.canvas.selected_index()}")
    image_path = OUT_DIR / "15_anomaly_marks.png"
    window.grab().save(str(image_path))
    print("saved", image_path)

    rows = page._anomaly_dialog.table.rowCount()
    print(f"dialog table rows: {rows}")

    page._anomaly_dialog.close()
    state.dirty = False
    window.close()
    QApplication.processEvents()
    return 0


if __name__ == "__main__":
    sys.exit(main())
