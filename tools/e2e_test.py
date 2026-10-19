"""End-to-end smoke test (offscreen): YOLO model + real images -> auto-label -> export zip.

Usage:  python tools/e2e_test.py [path\\to\\model.pt|.onnx]
"""
from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import QApplication, QMessageBox

from autolabel.app_state import AppState
from autolabel.settings import Settings
from autolabel.theme import apply_theme
from autolabel.ui.main_window import MainWindow

FONT_DIR = Path("C:/Windows/Fonts")
FONT_FILES = ["segoeui.ttf", "segoeuib.ttf", "segoeuisb.ttf", "seguisb.ttf", "seguibl.ttf"]

IMAGES = Path(r"C:\Users\Admin\Desktop\label\exp-22\images")
MODEL = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Users\Admin\Desktop\label\exp-22.onnx")
ZIP = ROOT / "tools" / f"e2e_{MODEL.stem}_{MODEL.suffix.lstrip('.')}.zip"
SHOTS = ROOT / "tools" / "shots"

started = time.monotonic()


def note(message: str) -> None:
    print(f"[{time.monotonic() - started:7.1f}s] {message}", flush=True)


def main() -> int:
    app = QApplication(sys.argv)
    for name in FONT_FILES:
        path = FONT_DIR / name
        if path.is_file():
            QFontDatabase.addApplicationFont(str(path))
    apply_theme(app)

    QMessageBox.exec = lambda self: 0
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    QMessageBox.critical = staticmethod(lambda *a, **k: None)
    QMessageBox.warning = staticmethod(lambda *a, **k: None)

    settings = Settings()
    state = AppState(settings)
    window = MainWindow(state)
    window.resize(1380, 880)
    window.show()
    annotate = window.annotate_page
    export = window.export_page

    def fail(message: str) -> None:
        note(f"FAIL: {message}")
        app.exit(2)

    def finish(code: int) -> None:
        note(f"exit {code}")
        app.exit(code)

    def wait_model(elapsed: int = 0) -> None:
        loaded = state.detector is not None and state.detector_path == MODEL
        if loaded:
            note(f"model loaded: {state.detector.describe()} names={state.detector.names}")
            note(f"classes in state: {state.classes}")
            annotate.run_auto_label(mode="all")
            poll_label(0)
        elif elapsed > 90000:
            fail("model never loaded")
        else:
            QTimer.singleShot(400, lambda: wait_model(elapsed + 400))

    def poll_label(elapsed: int = 0) -> None:
        worker = annotate._label_worker
        running = worker is not None and worker.isRunning()
        if running:
            if elapsed > 300000:
                fail("auto-label timeout")
                return
            QTimer.singleShot(500, lambda: poll_label(elapsed + 500))
            return
        if worker is None:
            fail("label worker disappeared before starting")
            return

        total_boxes = sum(len(entry.boxes) for entry in state.entries)
        labeled = sum(1 for entry in state.entries if entry.boxes)
        note(f"auto-label done: {labeled}/{len(state.entries)} images with boxes, {total_boxes} boxes total")
        if total_boxes == 0:
            fail("no boxes produced")
            return

        first = next(i for i, entry in enumerate(state.entries) if entry.boxes)
        state.set_current(first)
        app.processEvents()
        QTimer.singleShot(400, lambda: after_label_shot(first))

    def after_label_shot(first: int) -> None:
        window._switch_page(0)
        app.processEvents()
        window.grab().save(str(SHOTS / "10_e2e_labeled.png"))
        note("saved 10_e2e_labeled.png")
        window._switch_page(1)
        app.processEvents()
        QTimer.singleShot(500, start_export)

    def start_export() -> None:
        if ZIP.exists():
            ZIP.unlink()
        export.zip_path_edit.setText(str(ZIP))
        app.processEvents()
        note(f"export preview: {len(export._items)} items, stats={export._stats}")
        export._start_export()
        poll_export(0)

    def poll_export(elapsed: int = 0) -> None:
        worker = export._worker
        running = worker is not None and worker.isRunning()
        if running:
            if elapsed > 120000:
                fail("export timeout")
                return
            QTimer.singleShot(400, lambda: poll_export(elapsed + 400))
            return
        if not ZIP.exists():
            fail("zip was not written")
            return
        window.grab().save(str(SHOTS / "11_e2e_export.png"))
        verify_zip()

    def verify_zip() -> None:
        with zipfile.ZipFile(ZIP) as zf:
            names = zf.namelist()
            images = [n for n in names if n.startswith("images/")]
            labels = [n for n in names if n.startswith("labels/")]
            data_yaml = zf.read("data.yaml").decode("utf-8")
        note(f"zip: {len(images)} images, {len(labels)} labels, data.yaml: {data_yaml.strip()!r}")
        if not images or not labels:
            fail("zip missing images/ or labels/")
            return
        with zipfile.ZipFile(ZIP) as zf:
            sample = zf.read(labels[0]).decode("utf-8").strip()
        note(f"first label file ({labels[0]}): {sample!r}")
        window._switch_page(1)
        QTimer.singleShot(300, lambda: finish(0))

    QTimer.singleShot(300, lambda: annotate.open_folder(IMAGES, silent=True))
    QTimer.singleShot(2500, lambda: annotate.load_model(MODEL))
    QTimer.singleShot(3100, lambda: wait_model(0))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
