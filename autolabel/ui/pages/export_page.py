from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QTimer, Qt, QUrl
from PyQt6.QtGui import QColor, QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...app_state import AppState
from ...exporter import (
    CONF_BANDS,
    ExportFilters,
    apply_filters,
    confidence_zip_paths,
    item_band,
    item_confidence,
)
from ...theme import COLORS
from ...workers import ExportWorker
from ..widgets import SpinRow, StatCard, ValueSlider, card, faint_label, make_button, section_label

PREVIEW_ROW_LIMIT = 800


class ExportPage(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state
        self._worker: ExportWorker | None = None
        self._items = []
        self._stats = None
        self._loading = True

        self._build_ui()
        self._load_settings_into_ui()
        self._loading = False
        self._wire_state()

        self._recompute_timer = QTimer(self)
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.setInterval(200)
        self._recompute_timer.timeout.connect(self.recompute)
        self.recompute()

    # -------------------------------------------------------------- building

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("Export dataset")
        title.setProperty("class", "h1")
        self.header_sub = QLabel("")
        self.header_sub.setProperty("class", "muted")
        header.addWidget(title)
        header.addSpacing(10)
        header.addWidget(self.header_sub)
        header.addStretch(1)
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_settings_panel())
        splitter.addWidget(self._build_preview_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 720])
        root.addWidget(splitter, 1)

    def _build_settings_panel(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(10)

        conf_card, conf_layout = card()
        conf_layout.addWidget(section_label("Confidence filter"))
        self.min_conf_slider = ValueSlider("Min conf", 0.01, 1.0, 0.25)
        self.max_conf_slider = ValueSlider("Max conf", 0.0, 1.0, 1.0)
        conf_layout.addWidget(self.min_conf_slider)
        conf_layout.addWidget(self.max_conf_slider)
        self.conf_range_hint = faint_label("Boxes outside this confidence range are dropped from the export.")
        conf_layout.addWidget(self.conf_range_hint)
        layout.addWidget(conf_card)

        band_card, band_layout = card()
        band_layout.addWidget(section_label("Confidence split"))
        self.conf_split_check = QCheckBox("Export 3 ZIPs by confidence")
        band_layout.addWidget(self.conf_split_check)
        self.conf_low_row = SpinRow("Red below", 0, 100, 30, suffix="%")
        self.conf_high_row = SpinRow("Green above", 0, 100, 70, suffix="%")
        band_layout.addWidget(self.conf_low_row)
        band_layout.addWidget(self.conf_high_row)
        band_layout.addWidget(
            faint_label(
                "Each image goes to exactly one ZIP using its lowest box confidence: "
                "high (green), medium (yellow), low (red). Images without boxes count as 0% "
                "and go to the low ZIP with an empty label. The conf range above is ignored in this mode."
            )
        )
        layout.addWidget(band_card)

        size_card, size_layout = card()
        size_layout.addWidget(section_label("Image size filter"))
        self.use_size_check = QCheckBox("Only export images within this size range")
        size_layout.addWidget(self.use_size_check)
        self.width_min = SpinRow("Width min", 0, 100000, 0)
        self.width_max = SpinRow("Width max", 0, 100000, 100000)
        self.height_min = SpinRow("Height min", 0, 100000, 0)
        self.height_max = SpinRow("Height max", 0, 100000, 100000)
        for row in (self.width_min, self.width_max, self.height_min, self.height_max):
            size_layout.addWidget(row)
        size_layout.addWidget(faint_label("Images from X px to Y px — e.g. keep only images at least 1920×1080."))
        layout.addWidget(size_card)

        labels_card, labels_layout = card()
        labels_layout.addWidget(section_label("Label filter"))
        self.skip_empty_check = QCheckBox("Only export images containing at least one box")
        labels_layout.addWidget(self.skip_empty_check)
        labels_layout.addWidget(
            faint_label(
                "Keep this off to include empty and noise-flagged images — they export with an empty label "
                "file and teach the model background (negative samples)."
            )
        )

        classes_header = QHBoxLayout()
        classes_header.addWidget(QLabel("Classes to include"))
        classes_header.addStretch(1)
        all_button = make_button("All", None, role="ghost")
        all_button.setFixedWidth(46)
        all_button.clicked.connect(lambda: self._set_all_classes(True))
        none_button = make_button("None", None, role="ghost")
        none_button.setFixedWidth(54)
        none_button.clicked.connect(lambda: self._set_all_classes(False))
        classes_header.addWidget(all_button)
        classes_header.addWidget(none_button)
        labels_layout.addLayout(classes_header)

        self.class_list = QListWidget()
        self.class_list.setMaximumHeight(140)
        self.class_list.itemChanged.connect(self._schedule_recompute)
        labels_layout.addWidget(self.class_list)
        layout.addWidget(labels_card)

        split_card, split_layout = card()
        split_layout.addWidget(section_label("Train / Val split"))
        self.split_check = QCheckBox("Split into train and val sets")
        split_layout.addWidget(self.split_check)
        self.val_ratio_slider = ValueSlider("Val share", 5, 50, 20, decimals=0, step=1, suffix=" %")
        self.seed_row = SpinRow("Seed", 0, 99999, 42, suffix="")
        split_layout.addWidget(self.val_ratio_slider)
        split_layout.addWidget(self.seed_row)
        split_layout.addWidget(faint_label("Without a split everything goes into images/ and val reuses train."))
        layout.addWidget(split_card)

        clean_card, clean_layout = card()
        clean_layout.addWidget(section_label("Source cleanup"))
        self.delete_after_check = QCheckBox("Delete the exported source images after a successful export")
        self.delete_after_check.setToolTip(
            "Only images that were packaged into the ZIP are deleted (Recycle Bin). "
            "Skipped images stay in the source folder."
        )
        clean_layout.addWidget(self.delete_after_check)
        self.delete_hint = faint_label("")
        clean_layout.addWidget(self.delete_hint)
        layout.addWidget(clean_card)

        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        self.settings_scroll = scroll
        return scroll

    def _build_preview_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 0, 0, 0)
        layout.setSpacing(10)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(10)
        self.stat_images = StatCard("Images", "0")
        self.stat_boxes = StatCard("Boxes", "0")
        self.stat_classes = StatCard("Classes", "0")
        self.stat_skipped = StatCard("Excluded", "0")
        for stat in (self.stat_images, self.stat_boxes, self.stat_classes, self.stat_skipped):
            stats_row.addWidget(stat)
        layout.addLayout(stats_row)

        self.band_summary = QLabel("")
        self.band_summary.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.band_summary)

        self.preview_table = QTableWidget(0, 5)
        self.preview_table.setHorizontalHeaderLabels(["Image", "Size (px)", "Boxes", "Confidence", "Status"])
        header = self.preview_table.horizontalHeader()
        for column in range(4):
            header_item = self.preview_table.horizontalHeaderItem(column)
            header_item.setTextAlignment(
                (Qt.AlignmentFlag.AlignLeft if column == 0 else Qt.AlignmentFlag.AlignCenter)
                | Qt.AlignmentFlag.AlignVCenter
            )
        header.setStretchLastSection(True)
        self.preview_table.verticalHeader().setVisible(False)
        self.preview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.preview_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setShowGrid(False)
        self.preview_table.setColumnWidth(0, 300)
        self.preview_table.setColumnWidth(1, 110)
        self.preview_table.setColumnWidth(2, 70)
        self.preview_table.setColumnWidth(3, 140)
        layout.addWidget(self.preview_table, 1)

        self.split_summary = QLabel("")
        self.split_summary.setProperty("class", "muted")
        layout.addWidget(self.split_summary)

        dest_row = QHBoxLayout()
        dest_row.setSpacing(8)
        self.zip_path_edit = QLineEdit()
        self.zip_path_edit.setPlaceholderText("Choose where to save the .zip…")
        browse_button = make_button("Browse…", "folder")
        browse_button.clicked.connect(self._choose_zip_path)
        dest_row.addWidget(self.zip_path_edit, 1)
        dest_row.addWidget(browse_button)
        layout.addLayout(dest_row)

        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        self.export_button = make_button("Export ZIP (Ultralytics)", "download", role="primary")
        self.export_button.clicked.connect(self._start_export)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(220)
        self.progress_bar.setVisible(False)
        self.export_hint = QLabel("")
        self.export_hint.setProperty("class", "faint")
        action_row.addWidget(self.export_button)
        action_row.addWidget(self.progress_bar)
        action_row.addWidget(self.export_hint, 1)
        layout.addLayout(action_row)
        return panel

    # -------------------------------------------------------------- settings

    def _load_settings_into_ui(self) -> None:
        stored = dict(self.state.settings.data.get("export", {}))
        filters = ExportFilters.from_dict(stored)

        self.min_conf_slider.set_value(filters.min_conf)
        self.max_conf_slider.set_value(filters.max_conf)
        self.use_size_check.setChecked(filters.use_size)
        self.width_min.set_value(filters.min_width)
        self.width_max.set_value(filters.max_width)
        self.height_min.set_value(filters.min_height)
        self.height_max.set_value(filters.max_height)
        self.skip_empty_check.setChecked(filters.skip_empty)
        self.split_check.setChecked(filters.split_enabled)
        self.val_ratio_slider.set_value(filters.val_ratio * 100.0)
        self.seed_row.set_value(filters.seed)
        self.conf_split_check.setChecked(filters.conf_split)
        self.conf_low_row.set_value(int(round(filters.conf_low * 100)))
        self.conf_high_row.set_value(int(round(filters.conf_high * 100)))
        self.delete_after_check.setChecked(bool(stored.get("delete_after", True)))
        self._stored_selected_classes = filters.selected_classes

        self._update_size_rows_enabled()
        self._update_split_rows_enabled()
        self._update_conf_split_state()
        self._update_delete_hint()

    def _wire_state(self) -> None:
        self.state.entries_changed.connect(self._schedule_recompute)
        self.state.annotations_changed.connect(self._schedule_recompute)
        self.state.classes_changed.connect(self._on_classes_changed)

        self.min_conf_slider.value_changed.connect(self._schedule_recompute)
        self.max_conf_slider.value_changed.connect(self._schedule_recompute)
        self.use_size_check.toggled.connect(self._on_use_size_toggled)
        for row in (self.width_min, self.width_max, self.height_min, self.height_max):
            row.value_changed.connect(self._schedule_recompute)
        self.skip_empty_check.toggled.connect(self._schedule_recompute)
        self.split_check.toggled.connect(self._on_split_toggled)
        self.val_ratio_slider.value_changed.connect(self._schedule_recompute)
        self.seed_row.value_changed.connect(self._schedule_recompute)
        self.conf_split_check.toggled.connect(self._on_conf_split_toggled)
        self.conf_low_row.value_changed.connect(self._on_conf_low_changed)
        self.conf_high_row.value_changed.connect(self._on_conf_high_changed)
        self.delete_after_check.toggled.connect(self._on_delete_toggled)

    def _on_delete_toggled(self, checked: bool) -> None:
        self._update_delete_hint()
        if not self._loading:
            self._persist_delete_flag()

    def _persist_delete_flag(self) -> None:
        stored = dict(self.state.settings.data.get("export", {}))
        stored["delete_after"] = self.delete_after_check.isChecked()
        self.state.settings.data["export"] = stored
        self.state.settings.save()

    def _update_delete_hint(self) -> None:
        if self.delete_after_check.isChecked():
            count = len(self._items)
            self.delete_hint.setText(
                f"{count} source image{'s' if count != 1 else ''} will be deleted (Recycle Bin) after a successful export."
            )
        else:
            self.delete_hint.setText("Off — exported images stay in the source folder.")

    def _on_classes_changed(self) -> None:
        selected = set(self._stored_selected_classes) if self._stored_selected_classes else None
        self.class_list.blockSignals(True)
        self.class_list.clear()
        if self.state.classes:
            for name in self.state.classes:
                item = QListWidgetItem(name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                checked = selected is None or name in selected
                item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
                self.class_list.addItem(item)
        else:
            placeholder = QListWidgetItem("Classes appear after a model is loaded")
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            placeholder.setForeground(QColor(COLORS["text_faint"]))
            self.class_list.addItem(placeholder)
        self.class_list.blockSignals(False)
        self._schedule_recompute()

    def _set_all_classes(self, checked: bool) -> None:
        self.class_list.blockSignals(True)
        for row in range(self.class_list.count()):
            item = self.class_list.item(row)
            if not (item.flags() & Qt.ItemFlag.ItemIsUserCheckable):
                continue
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.class_list.blockSignals(False)
        self._schedule_recompute()

    def _on_use_size_toggled(self, checked: bool) -> None:
        self._update_size_rows_enabled()
        self._schedule_recompute()

    def _update_size_rows_enabled(self) -> None:
        enabled = self.use_size_check.isChecked()
        for row in (self.width_min, self.width_max, self.height_min, self.height_max):
            row.set_enabled(enabled)

    def _on_split_toggled(self, checked: bool) -> None:
        self._update_split_rows_enabled()
        self._schedule_recompute()

    def _update_split_rows_enabled(self) -> None:
        enabled = self.split_check.isChecked()
        self.val_ratio_slider.setEnabled(enabled)
        self.seed_row.setEnabled(enabled)

    def _on_conf_split_toggled(self, checked: bool) -> None:
        self._update_conf_split_state()
        self._schedule_recompute()

    def _on_conf_low_changed(self, value: int) -> None:
        if value > self.conf_high_row.value():
            self.conf_high_row.set_value(value)
        self._schedule_recompute()

    def _on_conf_high_changed(self, value: int) -> None:
        if value < self.conf_low_row.value():
            self.conf_low_row.set_value(value)
        self._schedule_recompute()

    def _update_conf_split_state(self) -> None:
        enabled = self.conf_split_check.isChecked()
        self.conf_low_row.set_enabled(enabled)
        self.conf_high_row.set_enabled(enabled)
        self.min_conf_slider.setEnabled(not enabled)
        self.max_conf_slider.setEnabled(not enabled)
        self.skip_empty_check.setEnabled(not enabled)
        self.conf_range_hint.setText(
            "Ignored in confidence-split mode — every box is kept and sorted into bands."
            if enabled
            else "Boxes outside this confidence range are dropped from the export."
        )
        self.export_button.setText("Export 3 ZIPs (Ultralytics)" if enabled else "Export ZIP (Ultralytics)")

    # ------------------------------------------------------------- filtering

    def _schedule_recompute(self, *args) -> None:
        if self._loading:
            return
        self._recompute_timer.start()

    def current_filters(self) -> ExportFilters:
        selected_classes = [
            self.class_list.item(row).text()
            for row in range(self.class_list.count())
            if (self.class_list.item(row).flags() & Qt.ItemFlag.ItemIsUserCheckable)
            and self.class_list.item(row).checkState() == Qt.CheckState.Checked
        ]
        if not self.state.classes or len(selected_classes) == len(self.state.classes):
            selected = None
        else:
            selected = selected_classes

        conf_low = min(self.conf_low_row.value(), self.conf_high_row.value()) / 100.0
        conf_high = max(self.conf_low_row.value(), self.conf_high_row.value()) / 100.0

        return ExportFilters(
            min_conf=self.min_conf_slider.value(),
            max_conf=self.max_conf_slider.value(),
            use_size=self.use_size_check.isChecked(),
            min_width=self.width_min.value(),
            max_width=self.width_max.value(),
            min_height=self.height_min.value(),
            max_height=self.height_max.value(),
            skip_empty=self.skip_empty_check.isChecked(),
            selected_classes=selected,
            split_enabled=self.split_check.isChecked(),
            val_ratio=self.val_ratio_slider.value() / 100.0,
            seed=self.seed_row.value(),
            conf_split=self.conf_split_check.isChecked(),
            conf_low=conf_low,
            conf_high=conf_high,
        )

    def recompute(self) -> None:
        filters = self.current_filters()
        stored = filters.to_dict()
        stored["delete_after"] = self.delete_after_check.isChecked()
        self.state.settings.data["export"] = stored
        self.state.settings.save()

        entries = self.state.entries
        if not entries:
            self._items, self._stats = [], None
            self.stat_images.set_value("0")
            self.stat_boxes.set_value("0")
            self.stat_classes.set_value("0")
            self.stat_skipped.set_value("0")
            self.preview_table.setRowCount(0)
            self.header_sub.setText("Open an image folder in the Annotate page first")
            self.export_button.setEnabled(False)
            self.split_summary.setText("")
            self.band_summary.setText("")
            self._update_delete_hint()
            return

        items, stats = apply_filters(entries, filters)
        self._items, self._stats = items, stats

        self.stat_images.set_value(f"{stats.matched_images}/{stats.total_images}")
        self.stat_boxes.set_value(str(stats.exported_boxes))
        self.stat_classes.set_value(str(len(stats.classes_used)))
        excluded = stats.skipped_size + stats.skipped_empty
        self.stat_skipped.set_value(str(excluded))
        self.header_sub.setText(f"{self.state.image_dir.name}  ·  {stats.total_images} images scanned")

        self._fill_preview(filters)
        self._update_split_summary(filters)
        self._update_band_summary(filters)
        self._update_delete_hint()
        self.export_button.setEnabled(bool(items))

        if not self.zip_path_edit.text() and self.state.image_dir:
            default_name = f"{self.state.image_dir.name}_yolo.zip"
            self.zip_path_edit.setText(str(self.state.image_dir.parent / default_name))

    def _fill_preview(self, filters: ExportFilters) -> None:
        included_rels = {item.entry.rel for item in self._items}
        kept_counts = {item.entry.rel: len(item.boxes) for item in self._items}

        rows = list(self._items)
        excluded_entries = [entry for entry in self.state.entries if entry.included and entry.rel not in included_rels]

        self.preview_table.setRowCount(0)
        self.preview_table.setUpdatesEnabled(False)
        total_rows = min(len(rows) + len(excluded_entries), PREVIEW_ROW_LIMIT)
        self.preview_table.setRowCount(total_rows)

        row = 0
        for item in rows:
            if row >= PREVIEW_ROW_LIMIT:
                break
            entry = item.entry
            score = item_confidence(item)
            if filters.conf_split:
                band = item_band(item, filters)
                label = {"high": "High", "medium": "Medium", "low": "Low"}[band]
                conf_color = {"high": COLORS["success"], "medium": COLORS["warning"], "low": COLORS["danger"]}[band]
                confidence = "0% · background" if not item.boxes else f"{score * 100:.0f}% · {label}"
            else:
                conf_color = None
                confidence = "—" if not item.boxes else f"{score * 100:.0f}%"
            self._set_row(
                row,
                entry.name,
                f"{entry.width} × {entry.height}",
                str(kept_counts.get(entry.rel, 0)),
                confidence,
                "Included",
                True,
                conf_color,
            )
            row += 1

        for entry in excluded_entries:
            if row >= PREVIEW_ROW_LIMIT:
                break
            reason = self._exclusion_reason(entry, filters)
            self._set_row(row, entry.name, f"{entry.width} × {entry.height}", "0", "—", reason, False)
            row += 1

        if row < total_rows:
            self.preview_table.setRowCount(row)
        self.preview_table.setUpdatesEnabled(True)

    def _exclusion_reason(self, entry, filters: ExportFilters) -> str:
        if filters.use_size:
            if not (filters.min_width <= entry.width <= filters.max_width):
                return "Size out of range"
            if not (filters.min_height <= entry.height <= filters.max_height):
                return "Size out of range"
        if filters.skip_empty:
            return "No boxes left"
        return "Excluded"

    def _set_row(
        self,
        row: int,
        name: str,
        size: str,
        boxes: str,
        confidence: str,
        status: str,
        included: bool,
        conf_color: str | None = None,
    ) -> None:
        name_item = QTableWidgetItem(name)
        name_item.setToolTip(name)
        size_item = QTableWidgetItem(size)
        boxes_item = QTableWidgetItem(boxes)
        boxes_item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
        conf_item = QTableWidgetItem(confidence)
        conf_item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
        if conf_color:
            conf_item.setForeground(QColor(conf_color))
        status_item = QTableWidgetItem(status)
        status_item.setForeground(QColor(COLORS["success"] if included else COLORS["text_faint"]))

        self.preview_table.setItem(row, 0, name_item)
        self.preview_table.setItem(row, 1, size_item)
        self.preview_table.setItem(row, 2, boxes_item)
        self.preview_table.setItem(row, 3, conf_item)
        self.preview_table.setItem(row, 4, status_item)

    def _update_split_summary(self, filters: ExportFilters) -> None:
        if not self._items or not filters.split_enabled:
            self.split_summary.setText("")
            return
        if filters.conf_split:
            groups: dict[str, list] = {band: [] for band in CONF_BANDS}
            for item in self._items:
                groups[item_band(item, filters)].append(item)
            train_count = val_count = 0
            for band_items in groups.values():
                count = len(band_items)
                band_val = max(1, int(round(count * filters.val_ratio))) if count > 1 else 0
                val_count += band_val
                train_count += count - band_val
        else:
            val_count = max(1, int(round(len(self._items) * filters.val_ratio))) if len(self._items) > 1 else 0
            train_count = len(self._items) - val_count
        suffix = " (per ZIP)" if filters.conf_split else ""
        self.split_summary.setText(f"Split preview — train: {train_count} images · val: {val_count} images{suffix}")

    def _update_band_summary(self, filters: ExportFilters) -> None:
        if not self._items or not filters.conf_split:
            self.band_summary.setText("")
            return
        counts = {band: 0 for band in CONF_BANDS}
        background = 0
        for item in self._items:
            counts[item_band(item, filters)] += 1
            if not item.boxes:
                background += 1
        self.band_summary.setText(
            f'<span style="color:{COLORS["success"]};font-weight:600">● High (green): {counts["high"]}</span>'
            f' &nbsp; <span style="color:{COLORS["warning"]};font-weight:600">'
            f'● Medium (yellow): {counts["medium"]}</span>'
            f' &nbsp; <span style="color:{COLORS["danger"]};font-weight:600">'
            f'● Low (red): {counts["low"]}</span>'
            f' &nbsp; <span style="color:{COLORS["text_faint"]}">({background} background)</span>'
        )

    # -------------------------------------------------------------- exporting

    def _choose_zip_path(self) -> None:
        start = self.zip_path_edit.text() or str(Path.home())
        chosen, _ = QFileDialog.getSaveFileName(self, "Save dataset zip", start, "Zip archive (*.zip)")
        if chosen:
            if not chosen.lower().endswith(".zip"):
                chosen += ".zip"
            self.zip_path_edit.setText(chosen)

    def _start_export(self) -> None:
        if not self._items:
            return
        zip_text = self.zip_path_edit.text().strip()
        if not zip_text:
            QMessageBox.information(self, "Choose destination", "Pick a destination .zip file first.")
            return
        zip_path = Path(zip_text)
        filters = self.current_filters()
        targets = list(confidence_zip_paths(zip_path).values()) if filters.conf_split else [zip_path]
        existing = [path for path in targets if path.exists()]
        if existing:
            names = ", ".join(path.name for path in existing)
            answer = QMessageBox.question(
                self,
                "Overwrite file?",
                f"{names} already exist. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        item_count = len(self._items)
        box_count = sum(len(item.boxes) for item in self._items)

        delete_sources = self.delete_after_check.isChecked()
        if delete_sources:
            folder = str(self.state.image_dir) if self.state.image_dir else "the source folder"
            packed = (
                "3 confidence ZIPs:\n" + "\n".join(str(path) for path in targets)
                if filters.conf_split
                else str(zip_path)
            )
            answer = QMessageBox.question(
                self,
                "Delete source images after export?",
                f"{item_count} images will be packed into {packed}.\n\n"
                f"After a successful export those {item_count} source files will be deleted "
                f"(Recycle Bin) from:\n{folder}\n\n"
                "Only successfully exported images are deleted — skipped images stay. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        self.export_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.export_hint.setText("Preparing…")
        self.state.busy_changed.emit(True, "Exporting dataset…")

        self._worker = ExportWorker(zip_path, list(self._items), list(self.state.classes), filters, delete_sources=delete_sources)
        self._worker.progress.connect(self._on_export_progress)
        self._worker.done.connect(
            lambda summary, images, boxes, deleted, failed: self._on_export_done(
                summary, images, boxes, deleted, failed, item_count, box_count
            )
        )
        self._worker.failed.connect(self._on_export_failed)
        self._worker.start()

    def _on_export_progress(self, done: int, total: int) -> None:
        percent = int(done / max(1, total) * 100)
        self.progress_bar.setValue(percent)
        self.export_hint.setText(f"Writing {done}/{total}…")

    def _on_export_done(
        self,
        summary: dict,
        images: int,
        boxes: int,
        deleted: int,
        failed: int,
        item_count: int,
        box_count: int,
    ) -> None:
        self.export_button.setEnabled(True)
        self.progress_bar.setValue(100)
        self.export_hint.setText("Done.")
        self.state.busy_changed.emit(False, "")
        zip_paths = summary.get("zip_paths", [])
        if summary.get("conf_split"):
            message = f"Export finished: {len(zip_paths)} ZIPs — {images} images"
        else:
            message = f"Export finished: {zip_paths[0] if zip_paths else ''}"
        if deleted or failed:
            message += f" — {deleted} source images deleted"
            if failed:
                message += f", {failed} could not be deleted"
        self.state.status_message.emit(message, 8000)

        details = f"{images} images and {boxes} boxes written to:\n" + "\n".join(zip_paths)
        if summary.get("conf_split"):
            bands = summary.get("bands", {})
            background = summary.get("background", 0)
            details += (
                f"\n\nHigh (green): {bands.get('high', 0)} images"
                f"\nMedium (yellow): {bands.get('medium', 0)} images"
                f"\nLow (red): {bands.get('low', 0)} images (incl. {background} background with empty labels)"
            )
        if deleted or failed:
            details += f"\n\n{deleted} source images deleted"
            if failed:
                details += f", {failed} could not be deleted"
            details += "."

        box = QMessageBox(self)
        box.setWindowTitle("Export finished")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText("Dataset exported successfully.")
        box.setInformativeText(details)
        open_button = box.addButton("Open containing folder", QMessageBox.ButtonRole.ActionRole)
        box.addButton("Close", QMessageBox.ButtonRole.AcceptRole)
        box.exec()
        if box.clickedButton() is open_button and zip_paths:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(zip_paths[0]).parent)))
        QTimer.singleShot(1200, lambda: self.progress_bar.setVisible(False))

        if deleted:
            self.state.folder_reload_requested.emit()

    def _on_export_failed(self, message: str) -> None:
        self.export_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.export_hint.setText("")
        self.state.busy_changed.emit(False, "")
        QMessageBox.critical(self, "Export failed", message)
