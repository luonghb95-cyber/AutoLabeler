"""Offscreen probe: export with 'delete exported source images' enabled."""
from __future__ import annotations

import shutil
import sys
import time
import zipfile
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

SOURCE = Path(r"C:\Users\Admin\Desktop\label\exp-22\images")
WORK = Path(r"C:\Users\Admin\AppData\Local\Temp\opencode\export_del_test")
IMAGES = WORK / "images"
ZIP = WORK / "out.zip"
COUNT = 6


def wait_until(cond, timeout_s: float = 60.0, label: str = "") -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if cond():
            return True
        time.sleep(0.03)
    print(f"TIMEOUT waiting for {label or cond}")
    return False


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    IMAGES.mkdir(parents=True)
    originals = sorted(SOURCE.glob("*.jpg"))[:COUNT]
    for path in originals:
        shutil.copy2(path, IMAGES / path.name)
    print(f"copied {len(originals)} images to {IMAGES}")

    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.critical = staticmethod(lambda *a, **k: print("CRITICAL:", a))
    QMessageBox.exec = lambda self, *a, **k: 0

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
    export = window.export_page

    settle_deadline = time.time() + 0.8
    while time.time() < settle_deadline:
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        time.sleep(0.03)

    page.open_folder(IMAGES, silent=True)
    if not wait_until(lambda: state.has_folder() and len(state.entries) == COUNT, 60, "folder scan"):
        return 1
    print(f"entries: {len(state.entries)}")

    window._switch_page(1)
    export.skip_empty_check.setChecked(False)
    export.recompute()
    QApplication.processEvents()
    print(f"export items: {len(export._items)}  delete flag: {export.delete_after_check.isChecked()}")
    export.delete_after_check.setChecked(True)
    export.zip_path_edit.setText(str(ZIP))
    export._start_export()
    if not wait_until(lambda: export.export_hint.text() == "Done.", 90, "export done"):
        return 1

    remaining = sorted(path.name for path in IMAGES.glob("*.jpg"))
    print(f"source images remaining: {len(remaining)} (expected 0)  {'OK' if not remaining else 'MISMATCH'}")
    if remaining:
        print("  leftover:", remaining)

    with zipfile.ZipFile(ZIP) as archive:
        names = archive.namelist()
        images_in_zip = [n for n in names if n.startswith("images/")]
        labels_in_zip = [n for n in names if n.startswith("labels/")]
    print(f"zip contents: {len(images_in_zip)} images, {len(labels_in_zip)} labels (expected {COUNT}/{COUNT})")
    print("files:", names[:4], "…")

    entries_after = len(state.entries)
    print(f"entries after reload: {entries_after} (expected 0)")

    result = 0
    if remaining or len(images_in_zip) != COUNT or len(labels_in_zip) != COUNT:
        result = 1
    window.close()
    QApplication.processEvents()
    shutil.rmtree(WORK, ignore_errors=True)
    print("probe", "OK" if result == 0 else "FAILED")
    return result


if __name__ == "__main__":
    sys.exit(main())
