from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..app_state import AppState
from ..constants import APP_NAME, APP_VERSION
from ..theme import COLORS
from .icons import make_icon
from .pages.annotate_page import AnnotatePage
from .pages.export_page import ExportPage
from .widgets import StatusDot, card, make_button


class MainWindow(QMainWindow):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state

        self.setWindowTitle(f"{APP_NAME} — YOLO auto-labeling toolkit")
        self.setWindowIcon(make_icon("tag", COLORS["accent"], 64))
        self.resize(1380, 880)
        self.setMinimumSize(1120, 720)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_sidebar())

        self.pages = QStackedWidget()
        self.annotate_page = AnnotatePage(self.state)
        self.export_page = ExportPage(self.state)
        self.pages.addWidget(self.annotate_page)
        self.pages.addWidget(self.export_page)
        layout.addWidget(self.pages, 1)

        self.setCentralWidget(central)
        self._build_status_bar()
        self._wire_state()
        self._wire_shortcuts()

        self.annotate_page.model_load_requested.connect(self._load_model)
        QTimer.singleShot(300, self._restore_session)

    # -------------------------------------------------------------- sidebar

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(224)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 18, 14, 16)
        layout.setSpacing(8)

        brand_row = QHBoxLayout()
        brand_row.setSpacing(10)
        logo = QLabel()
        logo.setPixmap(make_icon("tag", "#5b8cff", 26).pixmap(26, 26))
        brand_text = QVBoxLayout()
        brand_text.setSpacing(0)
        brand = QLabel(APP_NAME)
        brand.setProperty("class", "brand")
        brand_sub = QLabel("YOLO dataset toolkit")
        brand_sub.setProperty("class", "brand-sub")
        brand_text.addWidget(brand)
        brand_text.addWidget(brand_sub)
        brand_row.addWidget(logo)
        brand_row.addLayout(brand_text)
        brand_row.addStretch(1)
        layout.addLayout(brand_row)
        layout.addSpacing(16)

        nav_group = QButtonGroup(self)
        nav_group.setExclusive(True)
        self.annotate_nav = self._nav_button("Annotate & review", "tag")
        self.export_nav = self._nav_button("Export dataset", "download")
        nav_group.addButton(self.annotate_nav, 0)
        nav_group.addButton(self.export_nav, 1)
        nav_group.idClicked.connect(self._switch_page)
        layout.addWidget(self.annotate_nav)
        layout.addWidget(self.export_nav)
        self.annotate_nav.setChecked(True)

        layout.addStretch(1)

        model_card, model_layout = card()
        model_header = QHBoxLayout()
        model_header.setSpacing(8)
        self.model_dot = StatusDot(9)
        model_title = QLabel("MODEL")
        model_title.setProperty("class", "section")
        model_header.addWidget(self.model_dot)
        model_header.addWidget(model_title)
        model_header.addStretch(1)
        model_layout.addLayout(model_header)

        self.model_name_label = QLabel("No model loaded")
        self.model_name_label.setProperty("class", "model-name")
        self.model_name_label.setWordWrap(True)
        self.model_meta_label = QLabel("Load a .pt or .onnx model")
        self.model_meta_label.setProperty("class", "model-meta")
        self.model_meta_label.setWordWrap(True)
        model_layout.addWidget(self.model_name_label)
        model_layout.addWidget(self.model_meta_label)

        self.model_switch_button = QPushButton("Switch model")
        self.model_switch_button.setProperty("role", "sidebar-ghost")
        self.model_switch_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.model_switch_button.clicked.connect(self._show_model_menu)
        model_layout.addWidget(self.model_switch_button)
        layout.addWidget(model_card)

        version = QLabel(f"v{APP_VERSION}")
        version.setProperty("class", "brand-sub")
        layout.addWidget(version, alignment=Qt.AlignmentFlag.AlignHCenter)
        return sidebar

    def _nav_button(self, text: str, icon_name: str) -> QPushButton:
        button = QPushButton(text)
        button.setProperty("role", "nav")
        button.setIcon(make_icon(icon_name, "#8e9bb3", 18))
        button.setCheckable(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)
        self.busy_label = QLabel("")
        self.busy_label.setProperty("class", "muted")
        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setFixedWidth(110)
        self.busy_bar.setVisible(False)
        bar.addPermanentWidget(self.busy_label)
        bar.addPermanentWidget(self.busy_bar)
        bar.showMessage("Ready — open an image folder to begin.")

    # ---------------------------------------------------------------- wiring

    def _wire_state(self) -> None:
        self.state.status_message.connect(lambda text, timeout: self.statusBar().showMessage(text, timeout))
        self.state.busy_changed.connect(self._on_busy_changed)
        self.state.model_changed.connect(self._refresh_model_card)
        self.state.entries_changed.connect(self._refresh_window_title)

    def _wire_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+1"), self, activated=lambda: self._switch_page(0))
        QShortcut(QKeySequence("Ctrl+2"), self, activated=lambda: self._switch_page(1))
        QShortcut(QKeySequence("Ctrl+E"), self, activated=lambda: self._switch_page(1))

    def _switch_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        (self.annotate_nav if index == 0 else self.export_nav).setChecked(True)

    def _on_busy_changed(self, busy: bool, text: str) -> None:
        self.busy_bar.setVisible(busy)
        self.busy_label.setText(text if busy else "")

    def _refresh_model_card(self) -> None:
        detector = self.state.detector
        if detector is None or self.state.detector_path is None:
            self.model_dot.set_color(COLORS["text_faint"])
            self.model_name_label.setText("No model loaded")
            self.model_meta_label.setText("Load a .pt or .onnx model")
            return
        self.model_dot.set_color(COLORS["success"])
        self.model_name_label.setText(self.state.detector_path.name)
        classes = self.state.model_class_names()
        self.model_meta_label.setText(f"{detector.describe()} · {len(classes)} classes")

    def _refresh_window_title(self) -> None:
        folder = self.state.image_dir.name if self.state.image_dir else ""
        suffix = f" — {folder}" if folder else ""
        self.setWindowTitle(f"{APP_NAME} — YOLO auto-labeling toolkit{suffix}")

    # ----------------------------------------------------------------- model

    def _load_model(self, path: str) -> None:
        self.annotate_page.load_model(path or None)

    def _show_model_menu(self) -> None:
        menu = QMenu(self)
        load_action = menu.addAction(make_icon("cpu", COLORS["text"], 16), "Load model…")
        recent = self.state.settings.existing_recent_models()
        actions = {}
        if recent:
            menu.addSeparator()
            title = menu.addAction("Recent models")
            title.setEnabled(False)
            for path in recent:
                action = menu.addAction(make_icon("cpu", COLORS["text_muted"], 16), Path(path).name)
                action.setToolTip(path)
                actions[action] = path
        chosen = menu.exec(self.model_switch_button.mapToGlobal(self.model_switch_button.rect().topRight()))
        if chosen is None:
            return
        if chosen is load_action:
            self.annotate_page.load_model(None)
        elif chosen in actions:
            self.annotate_page.load_model(actions[chosen])

    # --------------------------------------------------------------- session

    def _restore_session(self) -> None:
        last_model = self.state.settings.get("last_model", "")
        if last_model and Path(last_model).is_file():
            self.annotate_page.load_model(last_model)

        last_folder = self.state.settings.get("last_folder", "")
        if last_folder and Path(last_folder).is_dir():
            self.annotate_page.open_folder(last_folder, silent=True)
