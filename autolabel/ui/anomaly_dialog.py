from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..app_state import AppState
from ..theme import COLORS
from .widgets import card, faint_label, make_button, section_label

MAX_ROWS = 1000


class AnomalyDialog(QDialog):
    scan_requested = pyqtSignal(bool)
    marks_changed = pyqtSignal(object)
    cleared = pyqtSignal()
    go_to = pyqtSignal(int, int)

    def __init__(self, state: AppState, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state
        self._hits: list = []
        self._stats: dict = {}

        self.setWindowTitle("Find Unusual")
        self.setMinimumSize(660, 540)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        layout.addWidget(self._build_controls_card())

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Score", "Image", "Class", "Conf", "Why"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setShowGrid(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 64)
        self.table.setColumnWidth(2, 80)
        self.table.setColumnWidth(3, 64)
        self.table.doubleClicked.connect(self._on_row_activated)
        layout.addWidget(self.table, 1)

        self.summary_label = faint_label("Scan the dataset to find unusual boxes.")
        layout.addWidget(self.summary_label)

        buttons = QHBoxLayout()
        self.scan_button = make_button("Scan dataset", "search", role="primary")
        self.scan_button.clicked.connect(self._on_scan_clicked)
        self.clear_button = make_button("Clear marks", "x", role="ghost")
        self.clear_button.clicked.connect(self._on_clear_clicked)
        self.clear_button.setEnabled(False)
        close_button = make_button("Close")
        close_button.clicked.connect(self.close)
        buttons.addWidget(self.scan_button)
        buttons.addWidget(self.clear_button)
        buttons.addStretch(1)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

        settings = state.settings.get("anomaly", {}) or {}
        if not isinstance(settings, dict):
            settings = {}
        try:
            self.threshold_spin.setValue(float(settings.get("threshold", 2.5)))
        except (TypeError, ValueError):
            pass
        self.manual_check.setChecked(bool(settings.get("include_manual", True)))
        self.threshold_spin.valueChanged.connect(self._on_settings_changed)
        self.manual_check.toggled.connect(self._on_manual_toggled)

    # ---------------------------------------------------------------- cards

    def _build_controls_card(self) -> QWidget:
        controls_card, controls_layout = card()
        controls_layout.addWidget(section_label("Sensitivity"))

        row = QHBoxLayout()
        row.setSpacing(8)
        label = QLabel("Flag boxes scoring above")
        label.setProperty("class", "muted")
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(1.5, 6.0)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSingleStep(0.25)
        self.threshold_spin.setValue(2.5)
        self.threshold_spin.setFixedWidth(88)
        self.threshold_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.threshold_spin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.threshold_spin.setToolTip("Lower = flag more boxes, higher = flag only the most extreme ones")
        self.manual_check = QCheckBox("Include manual boxes")
        self.manual_check.setToolTip("Manual boxes (drawn by hand) have no confidence but can still look unusual")
        row.addWidget(label)
        row.addStretch(1)
        row.addWidget(self.threshold_spin)
        row.addSpacing(12)
        row.addWidget(self.manual_check)
        controls_layout.addLayout(row)

        controls_layout.addWidget(
            faint_label(
                "Anomaly score combines shape, size, position and how much the box content "
                "differs from its immediate surroundings (colours, texture). Confidence is NOT used — "
                "false boxes often look very confident. Double-click a row to jump to that box."
            )
        )
        return controls_card

    # -------------------------------------------------------------- results

    def set_results(self, hits: list, stats: dict) -> None:
        self._hits = list(hits or [])
        self._stats = dict(stats or {})
        self._refresh_table()

    def set_scanning(self, running: bool) -> None:
        self.scan_button.setEnabled(not running)
        self.scan_button.setText("Scanning…" if running else "Scan dataset")

    def _on_scan_clicked(self) -> None:
        self._save_settings()
        self.scan_requested.emit(self.manual_check.isChecked())

    def _on_settings_changed(self, *args) -> None:
        self._save_settings()
        self._refresh_table()

    def _on_manual_toggled(self, checked: bool) -> None:
        self._save_settings()
        if self._hits:
            self.scan_requested.emit(checked)

    def _on_clear_clicked(self) -> None:
        self._hits = []
        self._stats = {}
        self._refresh_table()
        self.cleared.emit()

    def _save_settings(self) -> None:
        self.state.settings.set(
            "anomaly",
            {
                "threshold": self.threshold_spin.value(),
                "include_manual": self.manual_check.isChecked(),
            },
        )

    def _flagged(self) -> list:
        threshold = self.threshold_spin.value()
        return [hit for hit in self._hits if hit.score >= threshold]

    def _refresh_table(self) -> None:
        flagged = self._flagged()
        self.table.setRowCount(0)
        self.table.setRowCount(min(len(flagged), MAX_ROWS))

        marked: dict[int, dict[int, float]] = {}
        for row, hit in enumerate(flagged[:MAX_ROWS]):
            entry = self.state.entries[hit.entry_index] if 0 <= hit.entry_index < len(self.state.entries) else None
            conf_text = "manual" if hit.conf is None else f"{hit.conf * 100:.0f}%"
            name = entry.name if entry is not None else f"#{hit.entry_index}"
            values = [f"{hit.score:.2f}", name, hit.label, conf_text, ", ".join(hit.reasons) or "—"]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setToolTip(f"{name}\n{hit.label} · {conf_text}\nscore {hit.score:.2f}\n" + ", ".join(hit.reasons))
                if column == 0:
                    item.setForeground(QColor(COLORS["danger"] if hit.score >= self.threshold_spin.value() + 1.5 else COLORS["text"]))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.table.setItem(row, column, item)

        for hit in flagged:
            marked.setdefault(hit.entry_index, {})[hit.box_index] = hit.score

        images = len(marked)
        if self._hits:
            text = f"Flagged {len(flagged)} of {len(self._hits)} scored boxes in {images} images."
        else:
            text = "Scan the dataset to find unusual boxes."
        self.summary_label.setText(text)
        self.clear_button.setEnabled(bool(self._hits))

        self.marks_changed.emit(marked)

    def _on_row_activated(self, index) -> None:
        row = index.row()
        if row < 0:
            return
        flagged = self._flagged()
        if row >= len(flagged):
            return
        hit = flagged[row]
        self.go_to.emit(hit.entry_index, hit.box_index)
