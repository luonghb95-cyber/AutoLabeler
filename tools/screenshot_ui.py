"""Dev helper: render the main window offscreen and save screenshots."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import QApplication

FONT_DIR = Path("C:/Windows/Fonts")
FONT_FILES = ["segoeui.ttf", "segoeuib.ttf", "segoeuisb.ttf", "seguisb.ttf", "seguibl.ttf"]

from autolabel.app_state import AppState
from autolabel.settings import Settings
from autolabel.theme import apply_theme
from autolabel.ui.main_window import MainWindow

OUT_DIR = ROOT / "tools" / "shots"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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

    steps = []

    def shot(name: str):
        def _take():
            pixmap = window.grab()
            path = OUT_DIR / f"{name}.png"
            pixmap.save(str(path))
            print("saved", path)

        return _take

    def open_test_folder():
        if len(sys.argv) > 1:
            window.annotate_page.open_folder(sys.argv[1], silent=True)

    steps = [
        (400, shot("01_annotate_empty")),
        (200, open_test_folder),
        (1500, shot("02_annotate_folder")),
        (100, lambda: window._switch_page(1)),
        (400, shot("03_export")),
        (200, lambda: window.close()),
        (100, lambda: app.quit()),
    ]

    delay = 0
    for delta, func in steps:
        delay += delta
        QTimer.singleShot(delay, func)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
