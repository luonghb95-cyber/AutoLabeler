from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from .app_state import AppState
from .constants import APP_NAME, ORG_NAME
from .settings import Settings
from .theme import COLORS, apply_theme
from .ui.icons import make_icon
from .ui.main_window import MainWindow


def run() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    apply_theme(app)
    app.setWindowIcon(make_icon("tag", COLORS["accent"], 64))

    settings = Settings()
    state = AppState(settings)
    window = MainWindow(state)
    window.show()
    return app.exec()
