from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..smart_filter import SmartFilterOptions, compute_removals
from ..app_state import AppState
from ..theme import COLORS
from .widgets import card, faint_label, make_button, section_label


class SmartFilterDialog(QDialog):
    def __init__(self, state: AppState, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("Smart Filter")
        self.setModal(True)
        self.setMinimumWidth(460)

        self._class_rows: dict[str, tuple[QCheckBox, QDoubleSpinBox]] = {}
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        self._preview_timer.timeout.connect(self._refresh_preview)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        layout.addWidget(self._build_conf_card())
        layout.addWidget(self._build_geometry_card())
        layout.addWidget(self._build_tta_card())

        self.preview_label = faint_label("")
        layout.addWidget(self.preview_label)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_button = make_button("Cancel")
        cancel_button.clicked.connect(self.reject)
        apply_button = make_button("Apply", "check", role="primary")
        apply_button.clicked.connect(self._on_apply)
        buttons.addWidget(cancel_button)
        buttons.addWidget(apply_button)
        layout.addLayout(buttons)

        self._load_settings()
        self._refresh_preview()

    # ---------------------------------------------------------------- cards

    def _build_conf_card(self) -> QWidget:
        conf_card, conf_layout = card()
        conf_layout.addWidget(section_label("Confidence filter"))

        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        mode_label = QLabel("Mode")
        mode_label.setProperty("class", "muted")
        mode_label.setMinimumWidth(150)
        self.conf_mode_combo = QComboBox()
        self.conf_mode_combo.addItem("Per image — clear weak images (max conf below X)", "image")
        self.conf_mode_combo.addItem("Per box — drop each box below its class threshold", "box")
        self.conf_mode_combo.currentIndexChanged.connect(self._on_conf_mode_changed)
        mode_row.addWidget(mode_label)
        mode_row.addStretch(1)
        mode_row.addWidget(self.conf_mode_combo)
        conf_layout.addLayout(mode_row)

        self.image_conf_widget = QWidget()
        image_conf_layout = QHBoxLayout(self.image_conf_widget)
        image_conf_layout.setContentsMargins(0, 0, 0, 0)
        image_conf_layout.setSpacing(8)
        image_conf_label = QLabel("Best box conf below")
        image_conf_label.setProperty("class", "muted")
        image_conf_label.setMinimumWidth(150)
        self.image_conf_spin = QDoubleSpinBox()
        self.image_conf_spin.setRange(0.05, 0.95)
        self.image_conf_spin.setDecimals(2)
        self.image_conf_spin.setSingleStep(0.05)
        self.image_conf_spin.setValue(0.40)
        self.image_conf_spin.setFixedWidth(88)
        self.image_conf_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.image_conf_spin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.image_conf_spin.valueChanged.connect(self._schedule_preview)
        image_conf_layout.addWidget(image_conf_label)
        image_conf_layout.addStretch(1)
        image_conf_layout.addWidget(self.image_conf_spin)
        conf_layout.addWidget(self.image_conf_widget)
        conf_layout.addWidget(
            faint_label(
                "An image is 'weak' when even its best detector box (head or person) is below this value. "
                "Weak images lose all their detector boxes and get flagged as noise; strong images keep every box."
            )
        )

        self.box_conf_widget = QWidget()
        box_conf_layout = QVBoxLayout(self.box_conf_widget)
        box_conf_layout.setContentsMargins(0, 0, 0, 0)
        box_conf_layout.setSpacing(8)
        if self.state.classes:
            default = float(self.state.settings.get("conf", 0.25))
            for name in self.state.classes:
                row = QHBoxLayout()
                row.setSpacing(8)
                check = QCheckBox(name)
                check.setMinimumWidth(150)
                spin = QDoubleSpinBox()
                spin.setRange(0.05, 0.95)
                spin.setDecimals(2)
                spin.setSingleStep(0.05)
                spin.setValue(default)
                spin.setFixedWidth(88)
                spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
                spin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                spin.setEnabled(False)
                check.toggled.connect(spin.setEnabled)
                check.toggled.connect(self._schedule_preview)
                spin.valueChanged.connect(self._schedule_preview)
                row.addWidget(check)
                row.addStretch(1)
                row.addWidget(QLabel("drop below"))
                row.addWidget(spin)
                box_conf_layout.addLayout(row)
                self._class_rows[name] = (check, spin)
        else:
            box_conf_layout.addWidget(faint_label("No classes yet — load a model or add classes first."))
        box_conf_layout.addWidget(
            faint_label("Careful: this deletes individual boxes inside good images too — prefer 'Per image'.")
        )
        self.box_conf_widget.setVisible(False)
        conf_layout.addWidget(self.box_conf_widget)

        conf_layout.addWidget(faint_label("Only detector boxes are affected (manual boxes are never removed)."))
        return conf_card

    def _on_conf_mode_changed(self, *args) -> None:
        image_mode = self.conf_mode_combo.currentData() == "image"
        self.image_conf_widget.setVisible(image_mode)
        self.box_conf_widget.setVisible(not image_mode)
        self._schedule_preview()

    def _build_geometry_card(self) -> QWidget:
        geo_card, geo_layout = card()
        geo_layout.addWidget(section_label("Geometry rules"))

        min_row = QHBoxLayout()
        min_row.setSpacing(8)
        min_label = QLabel("Minimum box size")
        min_label.setProperty("class", "muted")
        min_label.setMinimumWidth(150)
        self.min_box_spin = QSpinBox()
        self.min_box_spin.setRange(0, 200)
        self.min_box_spin.setSuffix(" px")
        self.min_box_spin.setSpecialValueText("off")
        self.min_box_spin.setValue(6)
        self.min_box_spin.setFixedWidth(88)
        self.min_box_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.min_box_spin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.min_box_spin.setToolTip("Drop boxes whose smaller side is below this size. Set to 'off' to disable.")
        self.min_box_spin.valueChanged.connect(self._schedule_preview)
        min_row.addWidget(min_label)
        min_row.addStretch(1)
        min_row.addWidget(self.min_box_spin)
        geo_layout.addLayout(min_row)

        orphan_row = QHBoxLayout()
        orphan_row.setSpacing(8)
        orphan_label = QLabel("Orphan rule")
        orphan_label.setProperty("class", "muted")
        orphan_label.setMinimumWidth(150)
        self.orphan_combo = QComboBox()
        self.orphan_combo.addItem("Off", "off")
        self.orphan_combo.addItem("Safe — only when the image has a big box", "safe")
        self.orphan_combo.addItem("Aggressive — always drop", "aggressive")
        self.orphan_combo.currentIndexChanged.connect(self._schedule_preview)
        orphan_row.addWidget(orphan_label)
        orphan_row.addStretch(1)
        orphan_row.addWidget(self.orphan_combo)
        geo_layout.addLayout(orphan_row)

        pair_row = QHBoxLayout()
        pair_row.setSpacing(8)
        pair_label = QLabel("Small class")
        pair_label.setProperty("class", "muted")
        pair_label.setMinimumWidth(150)
        self.small_combo = QComboBox()
        self.big_combo = QComboBox()
        self._fill_class_combo(self.small_combo, "head")
        self._fill_class_combo(self.big_combo, "person")
        self.small_combo.currentIndexChanged.connect(self._schedule_preview)
        self.big_combo.currentIndexChanged.connect(self._schedule_preview)
        pair_label2 = QLabel("inside")
        pair_label2.setProperty("class", "muted")
        pair_row.addWidget(pair_label)
        pair_row.addStretch(1)
        pair_row.addWidget(self.small_combo)
        pair_row.addWidget(pair_label2)
        pair_row.addWidget(self.big_combo)
        geo_layout.addLayout(pair_row)

        margin_row = QHBoxLayout()
        margin_row.setSpacing(8)
        margin_label = QLabel("Match margin")
        margin_label.setProperty("class", "muted")
        margin_label.setMinimumWidth(150)
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 100)
        self.margin_spin.setSuffix(" %")
        self.margin_spin.setValue(30)
        self.margin_spin.setFixedWidth(88)
        self.margin_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.margin_spin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.margin_spin.setToolTip("The small box center may fall this far outside the big box and still match.")
        self.margin_spin.valueChanged.connect(self._schedule_preview)
        margin_row.addWidget(margin_label)
        margin_row.addStretch(1)
        margin_row.addWidget(self.margin_spin)
        geo_layout.addLayout(margin_row)

        geo_layout.addWidget(
            faint_label(
                "Safe mode drops an orphan small box only when the same image contains a big box; "
                "occluded cases without a matching big box are kept."
            )
        )
        return geo_card

    def _build_tta_card(self) -> QWidget:
        tta_card, tta_layout = card()
        tta_layout.addWidget(section_label("Flip-consensus (TTA)"))
        self.tta_check = QCheckBox("Keep a box only if the mirrored image also finds it")
        has_model = self.state.detector is not None
        self.tta_check.setEnabled(has_model)
        if not has_model:
            self.tta_check.setToolTip("Load a model to enable flip-consensus")
        self.tta_check.toggled.connect(self._schedule_preview)
        tta_layout.addWidget(self.tta_check)
        tta_layout.addWidget(
            faint_label(
                "Runs detection once more on the horizontally flipped image and matches boxes back. "
                "Rare detections that do not survive the flip are removed. Adds one inference pass per image."
            )
        )
        return tta_card

    def _fill_class_combo(self, combo: QComboBox, fallback: str) -> None:
        combo.addItems(self.state.classes)
        if fallback in self.state.classes:
            combo.setCurrentText(fallback)

    # -------------------------------------------------------------- preview

    def _schedule_preview(self, *args) -> None:
        self._preview_timer.start()

    def _refresh_preview(self) -> None:
        options = self.options()
        if not options.active():
            self.preview_label.setText("Nothing selected — no boxes will be removed.")
            return
        _removals, stats = compute_removals(self.state.entries, options)
        text = (
            f"Will remove {stats['removed']} boxes from {stats['images']} images"
            f" · {stats['empties']} images become empty (noise)"
        )
        if options.tta_enabled:
            text += "\nFlip-consensus will remove additional boxes (not counted in this preview)."
        self.preview_label.setText(text)

    # -------------------------------------------------------------- options

    def options(self) -> SmartFilterOptions:
        per_class_conf = {
            name: spin.value()
            for name, (check, spin) in self._class_rows.items()
            if check.isChecked()
        }
        return SmartFilterOptions(
            conf_mode=self.conf_mode_combo.currentData(),
            image_conf=float(self.image_conf_spin.value()),
            per_class_conf=per_class_conf,
            min_box_px=float(self.min_box_spin.value()),
            orphan_mode=self.orphan_combo.currentData(),
            orphan_margin=self.margin_spin.value() / 100.0,
            orphan_small=self.small_combo.currentText(),
            orphan_big=self.big_combo.currentText(),
            tta_enabled=self.tta_check.isChecked(),
        )

    def _on_apply(self) -> None:
        data = {
            "conf_mode": self.conf_mode_combo.currentData(),
            "image_conf": self.image_conf_spin.value(),
            "per_class_conf": {name: spin.value() for name, (check, spin) in self._class_rows.items() if check.isChecked()},
            "min_box_px": self.min_box_spin.value(),
            "orphan_mode": self.orphan_combo.currentData(),
            "orphan_margin": self.margin_spin.value(),
            "orphan_small": self.small_combo.currentText(),
            "orphan_big": self.big_combo.currentText(),
            "tta_enabled": self.tta_check.isChecked(),
        }
        self.state.settings.set("smart_filter", data)
        self.accept()

    def _load_settings(self) -> None:
        data = self.state.settings.get("smart_filter", {}) or {}
        if not isinstance(data, dict):
            return
        mode_index = self.conf_mode_combo.findData(str(data.get("conf_mode", "image")))
        if mode_index >= 0:
            self.conf_mode_combo.setCurrentIndex(mode_index)
        try:
            self.image_conf_spin.setValue(float(data.get("image_conf", 0.40)))
        except (TypeError, ValueError):
            pass
        stored_conf = data.get("per_class_conf") or {}
        for name, (check, spin) in self._class_rows.items():
            if name in stored_conf:
                check.setChecked(True)
                spin.setValue(float(stored_conf[name]))
        try:
            self.min_box_spin.setValue(int(data.get("min_box_px", 6)))
        except (TypeError, ValueError):
            pass
        mode = str(data.get("orphan_mode", "safe"))
        index = self.orphan_combo.findData(mode)
        if index >= 0:
            self.orphan_combo.setCurrentIndex(index)
        try:
            self.margin_spin.setValue(int(data.get("orphan_margin", 30)))
        except (TypeError, ValueError):
            pass
        if data.get("orphan_small") in self.state.classes:
            self.small_combo.setCurrentText(data["orphan_small"])
        if data.get("orphan_big") in self.state.classes:
            self.big_combo.setCurrentText(data["orphan_big"])
        if bool(data.get("tta_enabled", False)) and self.tta_check.isEnabled():
            self.tta_check.setChecked(True)
