from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from ...app_state import AppState
from ...constants import MODEL_EXTENSIONS
from ...image_io import load_bgr
from ...models import Box
from ...project import ProjectStore
from ...theme import COLORS
from ...workers import AnomalyWorker, AutoLabelWorker, FolderScanWorker, ModelLoadWorker, SmartFilterWorker, ThumbnailWorker
from ..anomaly_dialog import AnomalyDialog
from ..canvas import ImageCanvas
from ..icons import make_icon
from ..smart_filter_dialog import SmartFilterDialog
from ..widgets import ElidedLabel, EmptyState, ValueSlider, card, color_chip, faint_label, icon_button, make_button, section_label


def bgr_to_qimage(image: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    height, width, channels = rgb.shape
    qimage = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888)
    return qimage.copy()


class ImageItemDelegate(QStyledItemDelegate):
    ROW_HEIGHT = 58

    def sizeHint(self, option, index) -> QSize:
        return QSize(0, self.ROW_HEIGHT)

    @staticmethod
    def _elide_name(painter, name: str, width: int) -> str:
        metrics = painter.fontMetrics()
        if metrics.horizontalAdvance(name) <= width:
            return name
        stem, dot, extension = name.rpartition(".")
        if not dot:
            return metrics.elidedText(name, Qt.TextElideMode.ElideMiddle, width)
        extension_width = metrics.horizontalAdvance("." + extension)
        stem_width = width - extension_width - metrics.horizontalAdvance("…")
        if stem_width < 24:
            return metrics.elidedText(name, Qt.TextElideMode.ElideMiddle, width)
        return metrics.elidedText(stem, Qt.TextElideMode.ElideMiddle, max(24, stem_width)) + "." + extension

    def paint(self, painter, option, index):
        painter.save()
        painter.setRenderHint(painter.RenderHint.Antialiasing, True)

        rect = option.rect.adjusted(3, 2, -3, -2)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)

        if selected:
            painter.setBrush(QColor(COLORS["accent_soft"]))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect, 8, 8)
        elif hovered:
            painter.setBrush(QColor(COLORS["item_hover"]))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect, 8, 8)

        thumb_size = QSize(62, 44)
        thumb_rect = rect.adjusted(6, 7, 0, -7)
        thumb_rect.setWidth(thumb_size.width())
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if icon is not None:
            pixmap = icon.pixmap(thumb_size)
            painter.drawPixmap(thumb_rect, pixmap)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QColor(0, 0, 0, 26))
            painter.drawRoundedRect(thumb_rect, 5, 5)
        else:
            painter.setBrush(QColor("#e7ebf2"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(thumb_rect, 5, 5)

        text_left = thumb_rect.right() + 10
        text_width = max(40, rect.right() - text_left - 42)

        name = index.data(Qt.ItemDataRole.DisplayRole) or ""
        name_font = QFont("Segoe UI", 9)
        name_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(name_font)
        painter.setPen(QColor(COLORS["text"]))
        painter.drawText(text_left, rect.top() + 16, text_width, 17, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), self._elide_name(painter, name, text_width))

        meta = index.data(Qt.ItemDataRole.UserRole + 2) or ""
        meta_font = QFont("Segoe UI", 8)
        painter.setFont(meta_font)
        painter.setPen(QColor(COLORS["text_faint"]))
        meta = painter.fontMetrics().elidedText(meta, Qt.TextElideMode.ElideRight, text_width)
        painter.drawText(text_left, rect.top() + 33, text_width, 14, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), meta)

        count = index.data(Qt.ItemDataRole.UserRole + 1) or 0
        noise = bool(index.data(Qt.ItemDataRole.UserRole + 3))
        badge_text = "noise" if (count == 0 and noise) else str(count)
        badge_font = QFont("Segoe UI", 8)
        badge_font.setWeight(QFont.Weight.Bold)
        painter.setFont(badge_font)
        metrics = painter.fontMetrics()
        badge_width = max(22, metrics.horizontalAdvance(badge_text) + 12)
        badge_rect = rect.adjusted(0, 0, -8, 0)
        badge_rect.setLeft(rect.right() - badge_width - 6)
        badge_rect.setTop(rect.top() + (rect.height() - 22) // 2)
        badge_rect.setHeight(22)
        if count > 0:
            painter.setBrush(QColor(COLORS["success_soft"]))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(badge_rect, 11, 11)
            painter.setPen(QColor(COLORS["success"]))
        elif noise:
            painter.setBrush(QColor(COLORS["warning_soft"]))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(badge_rect, 11, 11)
            painter.setPen(QColor(COLORS["warning"]))
        else:
            painter.setBrush(QColor("#eef1f6"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(badge_rect, 11, 11)
            painter.setPen(QColor(COLORS["text_faint"]))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)

        painter.restore()


class AnnotatePage(QWidget):
    model_load_requested = pyqtSignal(str)

    def __init__(self, state: AppState, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state

        self._scan_worker: FolderScanWorker | None = None
        self._thumb_worker: ThumbnailWorker | None = None
        self._model_worker: ModelLoadWorker | None = None
        self._label_worker: AutoLabelWorker | None = None
        self._smart_worker: SmartFilterWorker | None = None
        self._anomaly_worker: AnomalyWorker | None = None
        self._anomaly_dialog: AnomalyDialog | None = None
        self._anomaly_hits: list = []
        self._anomaly_stats: dict = {}
        self._anomaly_marks: dict[int, dict[int, float]] = {}
        self._filter_backup: dict[int, tuple[list, bool]] | None = None
        self._syncing_selection = False
        self._image_cache_index = -1

        self._build_ui()
        self._wire_state()
        self._wire_shortcuts()

    # -------------------------------------------------------------- building

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_toolbar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([286, 660, 320])
        root.addWidget(splitter, 1)

    def _build_toolbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("TopBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(8)

        self.open_button = make_button("Open Folder", "folder")
        self.open_button.setToolTip("Open a folder of images (Ctrl+O)")
        self.open_button.clicked.connect(lambda: self.open_folder())

        self.model_button = make_button("Load Model", "cpu")
        self.model_button.setToolTip("Load a YOLO model (.pt or .onnx) (Ctrl+M)")
        self.model_button.clicked.connect(lambda: self.model_load_requested.emit(""))

        self.auto_all_button = make_button("Auto-Label All", "zap", role="primary")
        self.auto_all_button.clicked.connect(lambda: self.run_auto_label(mode="all"))
        self.auto_all_button.setEnabled(False)

        self.auto_missing_button = make_button("Auto-Label Missing", "zap")
        self.auto_missing_button.clicked.connect(lambda: self.run_auto_label(mode="missing"))
        self.auto_missing_button.setEnabled(False)

        self.smart_filter_button = make_button("Smart Filter", "sliders")
        self.smart_filter_button.setToolTip("Preview and remove suspicious detector boxes (conf per class, geometry, flip-consensus)")
        self.smart_filter_button.clicked.connect(self.run_smart_filter)
        self.smart_filter_button.setEnabled(False)

        self.undo_filter_button = make_button("Undo Filter", "refresh", role="ghost")
        self.undo_filter_button.setToolTip("Restore the boxes removed by the last Smart Filter run")
        self.undo_filter_button.clicked.connect(self._undo_smart_filter)
        self.undo_filter_button.setVisible(False)

        self.anomaly_button = make_button("Find Unusual", "alert")
        self.anomaly_button.setToolTip("Score every box across the dataset and mark the most unusual ones")
        self.anomaly_button.clicked.connect(self.open_anomaly_dialog)
        self.anomaly_button.setEnabled(False)

        self.save_button = make_button("Save", "save")
        self.save_button.setToolTip("Save annotations (Ctrl+S)")
        self.save_button.clicked.connect(self._manual_save)

        layout.addWidget(self.open_button)
        layout.addWidget(self.model_button)
        layout.addSpacing(6)
        line = QFrame()
        line.setProperty("class", "hline")
        line.setFixedWidth(1)
        line.setFixedHeight(22)
        layout.addWidget(line)
        layout.addWidget(self.auto_all_button)
        layout.addWidget(self.auto_missing_button)
        layout.addWidget(self.smart_filter_button)
        layout.addWidget(self.undo_filter_button)
        layout.addWidget(self.anomaly_button)
        layout.addStretch(1)
        layout.addWidget(self.save_button)
        return bar

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(272)
        panel.setMaximumWidth(420)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 6, 12)
        layout.setSpacing(8)

        search_row = QHBoxLayout()
        search_row.setSpacing(6)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search images…")
        self.search_edit.addAction(make_icon("search", COLORS["text_faint"], 16), QLineEdit.ActionPosition.LeadingPosition)
        self.search_edit.textChanged.connect(self._apply_search_filter)
        self.recursive_check = QCheckBox("Subfolders")
        self.recursive_check.setToolTip("Scan subfolders when opening a folder")
        self.recursive_check.setChecked(bool(self.state.settings.get("recursive", False)))
        search_row.addWidget(self.search_edit, 1)
        search_row.addWidget(self.recursive_check)
        layout.addLayout(search_row)

        self.image_list = QListWidget()
        self.image_list.setItemDelegate(ImageItemDelegate(self.image_list))
        self.image_list.setUniformItemSizes(True)
        self.image_list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.image_list.currentRowChanged.connect(self._on_image_row_changed)
        layout.addWidget(self.image_list, 1)

        self.list_stats = QLabel("No folder opened")
        self.list_stats.setProperty("class", "faint")
        layout.addWidget(self.list_stats)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 12, 6, 12)
        layout.setSpacing(8)

        self.canvas = ImageCanvas()
        self.canvas.set_color_provider(self.state.color_for)
        self.canvas.box_created.connect(self._on_box_created)
        self.canvas.box_modified.connect(self._on_box_modified)
        self.canvas.box_delete_requested.connect(self._delete_box)
        self.canvas.selection_changed.connect(self._on_canvas_selection)
        self.canvas.zoom_changed.connect(self._on_zoom_changed)
        self.canvas.mouse_position.connect(self._on_mouse_position)
        self.canvas.mouse_left.connect(lambda: self.coord_label.setText(""))
        self.canvas.context_menu_requested.connect(self._on_canvas_context_menu)

        canvas_stack = QStackedWidget()
        canvas_stack.addWidget(self.canvas)

        self.canvas_empty = EmptyState(
            "image",
            "No images loaded",
            "Open a folder with your images, load a YOLO model (.pt / .onnx) and run auto-labeling.",
        )
        open_action = make_button("Open Folder", "folder", role="primary")
        open_action.clicked.connect(lambda: self.open_folder())
        self.canvas_empty.add_action(open_action)
        canvas_stack.addWidget(self.canvas_empty)
        self.canvas_stack = canvas_stack
        canvas_stack.setCurrentWidget(self.canvas_empty)
        layout.addWidget(canvas_stack, 1)

        layout.addWidget(self._build_canvas_toolbar())
        return panel

    def _build_canvas_toolbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("CanvasToolbar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(6)

        self.prev_button = icon_button("chevron-left", "Previous image (A)")
        self.prev_button.clicked.connect(lambda: self.step_image(-1))
        self.next_button = icon_button("chevron-right", "Next image (D)")
        self.next_button.clicked.connect(lambda: self.step_image(1))

        self.image_label = ElidedLabel("—")
        font = QFont("Segoe UI", 9)
        font.setWeight(QFont.Weight.DemiBold)
        self.image_label.setFont(font)
        self.image_label.setMinimumWidth(110)

        self.zoom_out_button = icon_button("zoom-out", "Zoom out")
        self.zoom_out_button.clicked.connect(lambda: self.canvas.zoom_by(1 / 1.25))
        self.zoom_in_button = icon_button("zoom-in", "Zoom in")
        self.zoom_in_button.clicked.connect(lambda: self.canvas.zoom_by(1.25))
        self.zoom_label = QLabel("100%")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setFixedWidth(52)
        self.fit_button = icon_button("fit", "Fit to window (F)")
        self.fit_button.clicked.connect(self.canvas.fit_to_window)

        self.boxes_label = QLabel("0 boxes")
        self.coord_label = QLabel("")
        self.coord_label.setMinimumWidth(96)
        self.coord_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self.prev_button)
        layout.addWidget(self.next_button)
        layout.addWidget(self.image_label, 1)
        layout.addSpacing(12)
        layout.addWidget(self.boxes_label)
        layout.addSpacing(10)
        layout.addWidget(self.coord_label)
        layout.addSpacing(10)
        layout.addWidget(self.zoom_out_button)
        layout.addWidget(self.zoom_label)
        layout.addWidget(self.zoom_in_button)
        layout.addWidget(self.fit_button)
        return bar

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(304)
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 12, 12, 12)
        layout.setSpacing(10)

        layout.addWidget(self._build_inference_card())

        classes_card, classes_layout = card()
        classes_layout.addWidget(section_label("Classes"))
        self.active_class_row = QHBoxLayout()
        self.active_class_row.setSpacing(8)
        self.active_dot = QLabel()
        self.active_class_combo = QComboBox()
        self.active_class_combo.currentTextChanged.connect(self._on_active_class_changed)
        self.active_class_row.addWidget(self.active_dot)
        self.active_class_row.addWidget(self.active_class_combo, 1)
        classes_layout.addLayout(self.active_class_row)

        self.class_list = QListWidget()
        self.class_list.setMaximumHeight(150)
        self.class_list.itemChanged.connect(self._on_class_item_changed)
        self.class_list.currentRowChanged.connect(self._on_class_row_changed)
        classes_layout.addWidget(self.class_list)

        class_buttons = QHBoxLayout()
        add_class_button = make_button("Add", "plus", role="ghost")
        add_class_button.clicked.connect(self._add_class_dialog)
        remove_class_button = make_button("Remove", "minus", role="ghost")
        remove_class_button.clicked.connect(self._remove_class)
        class_buttons.addWidget(add_class_button)
        class_buttons.addWidget(remove_class_button)
        class_buttons.addStretch(1)
        classes_layout.addLayout(class_buttons)
        classes_layout.addWidget(faint_label("Check to show on canvas. Click a row to set the active class for new boxes."))
        layout.addWidget(classes_card)

        boxes_card, boxes_layout = card()
        boxes_layout.addWidget(section_label("Boxes in image"))
        self.box_list = QListWidget()
        self.box_list.setMaximumHeight(170)
        self.box_list.currentRowChanged.connect(self._on_box_row_changed)
        boxes_layout.addWidget(self.box_list)

        delete_row = QHBoxLayout()
        self.delete_box_button = make_button("Delete selected", "trash", role="danger")
        self.delete_box_button.clicked.connect(lambda: self._delete_box(self.canvas.selected_index()))
        delete_row.addWidget(self.delete_box_button)
        delete_row.addStretch(1)
        boxes_layout.addLayout(delete_row)

        props = QHBoxLayout()
        props.setSpacing(6)
        self.label_combo = QComboBox()
        self.label_combo.currentTextChanged.connect(self._on_box_label_changed)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0, 100)
        self.conf_spin.setDecimals(1)
        self.conf_spin.setSuffix(" %")
        self.conf_spin.setToolTip("Detector score. Manual boxes score 100%.")
        self.conf_spin.valueChanged.connect(self._on_box_conf_changed)
        props.addWidget(QLabel("Class"))
        props.addWidget(self.label_combo, 2)
        props.addWidget(QLabel("Conf"))
        props.addWidget(self.conf_spin, 1)
        boxes_layout.addLayout(props)

        coords = QHBoxLayout()
        coords.setSpacing(6)
        self.coord_spins: dict[str, QSpinBox] = {}
        for key in ("x1", "y1", "x2", "y2"):
            spin = QSpinBox()
            spin.setRange(0, 200000)
            spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
            spin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            spin.valueChanged.connect(self._on_box_coords_changed)
            label = QLabel(key.upper())
            label.setProperty("class", "faint")
            coords.addWidget(label)
            coords.addWidget(spin, 1)
            self.coord_spins[key] = spin
        boxes_layout.addLayout(coords)
        layout.addWidget(boxes_card)

        layout.addStretch(1)
        return panel

    def _build_inference_card(self) -> QWidget:
        inference_card, inference_layout = card()
        inference_layout.addWidget(section_label("Inference"))

        device_row = QHBoxLayout()
        device_row.setSpacing(8)
        device_label = QLabel("Device")
        device_label.setMinimumWidth(66)
        device_label.setProperty("class", "muted")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto", "CPU", "GPU"])
        self.device_combo.setCurrentText(str(self.state.settings.get("device", "auto")).capitalize())
        self.device_combo.currentTextChanged.connect(self._on_device_changed)
        imgsz_label = QLabel("Size")
        imgsz_label.setProperty("class", "muted")
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(256, 1600)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(int(self.state.settings.get("imgsz", 640)))
        self.imgsz_spin.setFixedWidth(84)
        self.imgsz_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.imgsz_spin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.imgsz_spin.setToolTip("Inference resolution (used for .pt models)")
        self.imgsz_spin.valueChanged.connect(lambda value: self.state.settings.set("imgsz", int(value)))
        device_row.addWidget(device_label)
        device_row.addWidget(self.device_combo, 1)
        device_row.addWidget(imgsz_label)
        device_row.addWidget(self.imgsz_spin)
        inference_layout.addLayout(device_row)

        self.conf_slider = ValueSlider("Conf", 0.01, 1.0, float(self.state.settings.get("conf", 0.25)))
        self.conf_slider.value_changed.connect(lambda value: self.state.settings.set("conf", value))
        self.iou_slider = ValueSlider("IoU", 0.01, 1.0, float(self.state.settings.get("iou", 0.45)))
        self.iou_slider.value_changed.connect(lambda value: self.state.settings.set("iou", value))
        inference_layout.addWidget(self.conf_slider)
        inference_layout.addWidget(self.iou_slider)

        self.model_status_label = QLabel("No model loaded")
        self.model_status_label.setProperty("class", "faint")
        self.model_status_label.setWordWrap(True)
        inference_layout.addWidget(self.model_status_label)
        return inference_card

    # ------------------------------------------------------------- shortcuts

    def _wire_shortcuts(self) -> None:
        def shortcut(sequence: str, handler):
            QShortcut(QKeySequence(sequence), self, activated=handler)

        shortcut("A", lambda: self.step_image(-1))
        shortcut("D", lambda: self.step_image(1))
        shortcut("F", self.canvas.fit_to_window)
        shortcut("Ctrl+S", self._manual_save)
        shortcut("Ctrl+O", lambda: self.open_folder())
        shortcut("Ctrl+M", lambda: self.model_load_requested.emit(""))
        shortcut("+", lambda: self.canvas.zoom_by(1.25))
        shortcut("=", lambda: self.canvas.zoom_by(1.25))
        shortcut("-", lambda: self.canvas.zoom_by(1 / 1.25))

    # ----------------------------------------------------------------- state

    def _wire_state(self) -> None:
        self.state.entries_changed.connect(self._on_entries_changed)
        self.state.image_changed.connect(self._load_current_image)
        self.state.classes_changed.connect(self._refresh_classes_ui)
        self.state.model_changed.connect(self._refresh_model_status)
        self.state.annotations_changed.connect(self._refresh_image_badges)
        self.state.folder_reload_requested.connect(self._reload_folder)

    def _reload_folder(self) -> None:
        folder = self.state.image_dir
        if folder is not None and self.state.entries:
            self.open_folder(folder, silent=True)

    def _on_entries_changed(self) -> None:
        has = self.state.has_folder()
        self.auto_all_button.setEnabled(has and self.state.detector is not None)
        self.auto_missing_button.setEnabled(has and self.state.detector is not None)
        self.smart_filter_button.setEnabled(has and bool(self.state.entries))
        self.anomaly_button.setEnabled(has and bool(self.state.entries))
        self._update_list_stats()

    def _refresh_model_status(self) -> None:
        detector = self.state.detector
        if detector is None:
            self.model_status_label.setText("No model loaded")
            return
        classes = ", ".join(self.state.model_class_names()) or "no classes"
        self.model_status_label.setText(
            f"{self.state.detector_path.name} · {detector.describe()}\nClasses: {classes}"
        )
        self.auto_all_button.setEnabled(self.state.has_folder())
        self.auto_missing_button.setEnabled(self.state.has_folder())

    # ------------------------------------------------------------- folders

    def open_folder(self, folder: str | Path | None = None, silent: bool = False) -> None:
        folder = Path(folder) if folder else None
        if folder is None:
            start_dir = str(self.state.settings.get("last_folder", "") or Path.home())
            chosen = QFileDialog.getExistingDirectory(self, "Open images folder", start_dir)
            if not chosen:
                return
            folder = Path(chosen)

        if not folder.is_dir():
            if not silent:
                QMessageBox.warning(self, "Folder not found", f"The folder does not exist:\n{folder}")
            return

        self._cancel_workers()
        self.state.entries = []
        self.state.current_index = -1
        self.image_list.clear()
        self.box_list.clear()
        self.canvas_stack.setCurrentWidget(self.canvas_empty)
        self._anomaly_marks = {}
        if self._anomaly_dialog is not None:
            self._anomaly_dialog.set_results([], {})

        recursive = self.recursive_check.isChecked()
        self.state.settings.set("recursive", recursive)
        self.state.settings.set("last_folder", str(folder))

        self._scan_folder = folder
        self.state.busy_changed.emit(True, f"Scanning {folder.name}…")
        self._scan_worker = FolderScanWorker(folder, recursive)
        self._scan_worker.batch_ready.connect(self._on_scan_batch)
        self._scan_worker.finished_scan.connect(self._on_scan_finished)
        self._scan_worker.failed.connect(lambda message: self.state.status_message.emit(f"Scan failed: {message}", 6000))
        self._scan_worker.start()

    def _on_scan_batch(self, batch: list) -> None:
        for entry in batch:
            self.state.entries.append(entry)
            item = QListWidgetItem(entry.name)
            item.setData(Qt.ItemDataRole.UserRole, len(self.state.entries) - 1)
            item.setData(Qt.ItemDataRole.UserRole + 1, 0)
            item.setData(Qt.ItemDataRole.UserRole + 2, f"{entry.width}×{entry.height}")
            item.setData(Qt.ItemDataRole.UserRole + 3, bool(entry.noise))
            self.image_list.addItem(item)

    def _on_scan_finished(self, total: int) -> None:
        folder = getattr(self, "_scan_folder", None)
        if folder is None:
            return
        self.state.busy_changed.emit(False, "")

        stored_boxes = {}
        stored_sizes = {}
        store = ProjectStore(folder)
        if store.exists():
            saved, _ = store.load_entries()
            for entry in saved:
                stored_boxes[entry.rel] = entry.boxes
                if entry.width and entry.height:
                    stored_sizes[entry.rel] = (entry.width, entry.height)

        for entry in self.state.entries:
            if entry.rel in stored_boxes:
                entry.boxes = stored_boxes[entry.rel]
            if entry.rel in stored_sizes and (entry.width == 0 or entry.height == 0):
                entry.width, entry.height = stored_sizes[entry.rel]

        self.state.set_folder(folder, self.state.entries)
        self._refresh_image_badges()
        self._update_list_stats()
        self.state.status_message.emit(f"Loaded {total} images from {folder.name}", 4000)

        if self.state.entries:
            self.canvas_stack.setCurrentWidget(self.canvas)
            self.image_list.setCurrentRow(0)
            self._thumb_worker = ThumbnailWorker(self.state.entries)
            self._thumb_worker.thumb_ready.connect(self._on_thumb_ready)
            self._thumb_worker.start()

    def _on_thumb_ready(self, index: int, image: QImage) -> None:
        if index < self.image_list.count():
            item = self.image_list.item(index)
            item.setIcon(QIcon(QPixmap.fromImage(image)))

    def _update_list_stats(self) -> None:
        total, labeled, boxes = self.state.counts()
        if total == 0:
            self.list_stats.setText("No folder opened")
            return
        noise = sum(1 for entry in self.state.entries if entry.noise)
        text = f"{total} images · {labeled} labeled · {boxes} boxes"
        if noise:
            text += f" · {noise} noise"
        self.list_stats.setText(text)

    def _refresh_image_badges(self) -> None:
        for index, entry in enumerate(self.state.entries):
            if index >= self.image_list.count():
                break
            item = self.image_list.item(index)
            item.setData(Qt.ItemDataRole.UserRole + 1, len(entry.boxes))
            item.setData(Qt.ItemDataRole.UserRole + 2, f"{entry.width}×{entry.height}")
            item.setData(Qt.ItemDataRole.UserRole + 3, bool(entry.noise))
        self.image_list.viewport().update()
        self._update_list_stats()
        self._refresh_class_counts()

    def _apply_search_filter(self, text: str) -> None:
        text = text.strip().lower()
        for row in range(self.image_list.count()):
            item = self.image_list.item(row)
            item.setHidden(bool(text) and text not in item.text().lower())

    def _on_image_row_changed(self, row: int) -> None:
        if row < 0:
            return
        item = self.image_list.item(row)
        index = item.data(Qt.ItemDataRole.UserRole)
        if index is not None and index != self.state.current_index:
            self.state.set_current(index)

    # --------------------------------------------------------------- images

    def _load_current_image(self, index: int) -> None:
        if index < 0 or index >= len(self.state.entries):
            self.canvas_stack.setCurrentWidget(self.canvas_empty)
            self.image_label.setText("—")
            self.box_list.clear()
            return
        entry = self.state.entries[index]
        image = load_bgr(entry.path)
        if image is None:
            self.state.status_message.emit(f"Could not read {entry.name}", 5000)
            self.canvas_stack.setCurrentWidget(self.canvas_empty)
            return
        height, width = image.shape[:2]
        if entry.width == 0 or entry.height == 0:
            entry.width, entry.height = width, height
        qimage = bgr_to_qimage(image)
        self.canvas.set_image(qimage, entry.boxes)
        self.canvas_stack.setCurrentWidget(self.canvas)
        self._apply_current_marks()

        display_name = entry.name
        self.image_label.setText(display_name)
        self.image_label.setToolTip(str(entry.path))
        self._image_cache_index = index
        self._refresh_boxes_list()
        self._update_boxes_label()

    def step_image(self, delta: int) -> None:
        if not self.state.entries:
            return
        self.state.set_current(self.state.current_index + delta)
        self.image_list.setCurrentRow(self.state.current_index)

    # ---------------------------------------------------------------- boxes

    def _on_box_created(self, x1: float, y1: float, x2: float, y2: float) -> None:
        entry = self.state.current_entry()
        if entry is None:
            return
        label = self.state.active_class()
        if not label:
            QMessageBox.information(
                self,
                "No classes",
                "Load a model or add a class before drawing boxes.",
            )
            return
        box = Box(label=label, x1=x1, y1=y1, x2=x2, y2=y2, conf=None)
        box.clip(entry.width or int(box.x2) + 1, entry.height or int(box.y2) + 1)
        if box.width < 2 or box.height < 2:
            return
        entry.boxes.append(box)
        entry.noise = False
        self.state.mark_dirty()
        self.state.annotations_changed.emit()
        self.canvas.set_boxes(entry.boxes)
        self.canvas.select(len(entry.boxes) - 1)
        self._refresh_boxes_list()

    def _on_box_modified(self, index: int) -> None:
        entry = self.state.current_entry()
        if entry is None or index >= len(entry.boxes):
            return
        box = entry.boxes[index]
        box.clip(entry.width or 100000, entry.height or 100000)
        self.state.mark_dirty()
        self.state.annotations_changed.emit()
        self.canvas.refresh()
        self._refresh_boxes_list()
        self._refresh_properties()

    def _delete_box(self, index: int) -> None:
        entry = self.state.current_entry()
        if entry is None or not (0 <= index < len(entry.boxes)):
            return
        del entry.boxes[index]
        self._drop_anomaly_marks(self.state.current_index)
        self.state.mark_dirty()
        self.state.annotations_changed.emit()
        self.canvas.set_boxes(entry.boxes)
        self._apply_current_marks()
        self._refresh_boxes_list()

    def _refresh_boxes_list(self) -> None:
        entry = self.state.current_entry()
        self._syncing_selection = True
        self.box_list.clear()
        if entry is not None and entry.boxes:
            for box in entry.boxes:
                item = QListWidgetItem(f"{box.label}   {box.score * 100:.0f}%")
                item.setIcon(color_chip(self.state.color_for(box.label).name()))
                item.setToolTip(
                    f"x1={box.x1:.0f}  y1={box.y1:.0f}  x2={box.x2:.0f}  y2={box.y2:.0f}"
                    + ("" if box.conf is None else f"\nconfidence {box.conf:.3f}")
                )
                self.box_list.addItem(item)
        else:
            placeholder = QListWidgetItem(
                "No boxes here — run Auto-label or draw on the canvas"
                if entry is not None
                else "Open a folder to get started"
            )
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            placeholder.setForeground(QColor(COLORS["text_faint"]))
            self.box_list.addItem(placeholder)
        self._syncing_selection = False
        self._refresh_properties()
        self._update_boxes_label()

    def _update_boxes_label(self) -> None:
        entry = self.state.current_entry()
        count = len(entry.boxes) if entry else 0
        self.boxes_label.setText(f"{count} box{'es' if count != 1 else ''}")

    def _on_canvas_selection(self, index: int) -> None:
        if self._syncing_selection:
            return
        self._syncing_selection = True
        if 0 <= index < self.box_list.count():
            self.box_list.setCurrentRow(index)
        else:
            self.box_list.clearSelection()
            self.box_list.setCurrentRow(-1)
        self._syncing_selection = False
        self._refresh_properties()

    def _on_box_row_changed(self, row: int) -> None:
        if self._syncing_selection:
            return
        entry = self.state.current_entry()
        if entry is None or not (0 <= row < len(entry.boxes)):
            return
        self.canvas.select(row)
        self._refresh_properties()

    def _refresh_properties(self) -> None:
        entry = self.state.current_entry()
        index = self.canvas.selected_index()
        has_selection = entry is not None and 0 <= index < len(entry.boxes)

        self.label_combo.setEnabled(has_selection)
        self.conf_spin.setEnabled(has_selection)
        for spin in self.coord_spins.values():
            spin.setEnabled(has_selection)

        self.label_combo.blockSignals(True)
        self.label_combo.clear()
        self.label_combo.addItems(self.state.classes)
        self.conf_spin.blockSignals(True)
        for spin in self.coord_spins.values():
            spin.blockSignals(True)

        if has_selection:
            box = entry.boxes[index]
            self.label_combo.setCurrentText(box.label)
            self.conf_spin.setValue(box.score * 100.0)
            self.coord_spins["x1"].setValue(int(round(box.x1)))
            self.coord_spins["y1"].setValue(int(round(box.y1)))
            self.coord_spins["x2"].setValue(int(round(box.x2)))
            self.coord_spins["y2"].setValue(int(round(box.y2)))
        else:
            self.conf_spin.setValue(100.0)
            for spin in self.coord_spins.values():
                spin.setValue(0)

        self.label_combo.blockSignals(False)
        self.conf_spin.blockSignals(False)
        for spin in self.coord_spins.values():
            spin.blockSignals(False)

    def _apply_box_edit(self) -> None:
        entry = self.state.current_entry()
        index = self.canvas.selected_index()
        if entry is None or not (0 <= index < len(entry.boxes)):
            return
        box = entry.boxes[index]
        label = self.label_combo.currentText()
        conf = self.conf_spin.value() / 100.0
        changed = False
        if label and label != box.label:
            box.label = label
            changed = True
        if abs(conf - box.score) > 1e-6:
            box.conf = None if conf >= 0.999 else conf
            changed = True

        x1 = self.coord_spins["x1"].value()
        y1 = self.coord_spins["y1"].value()
        x2 = self.coord_spins["x2"].value()
        y2 = self.coord_spins["y2"].value()
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            if (round(box.x1), round(box.y1), round(box.x2), round(box.y2)) != (x1, y1, x2, y2):
                box.x1, box.y1, box.x2, box.y2 = float(x1), float(y1), float(x2), float(y2)
                changed = True

        if changed:
            self.state.mark_dirty()
            self.state.annotations_changed.emit()
            self.canvas.refresh()
            self._refresh_boxes_list()

    def _on_box_label_changed(self, text: str) -> None:
        self._apply_box_edit()

    def _on_box_conf_changed(self, value: float) -> None:
        self._apply_box_edit()

    def _on_box_coords_changed(self, value: int) -> None:
        self._apply_box_edit()

    def _on_canvas_context_menu(self, index: int, global_pos) -> None:
        if index < 0:
            return
        entry = self.state.current_entry()
        if entry is None or index >= len(entry.boxes):
            return
        box = entry.boxes[index]

        menu = QMenu(self)
        class_menu = menu.addMenu("Change class")
        for name in self.state.classes:
            action = class_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == box.label)
            action.triggered.connect(lambda checked=False, target=name: self._set_box_class(index, target))

        menu.addSeparator()
        delete_action = menu.addAction(make_icon("trash", COLORS["danger"], 16), "Delete")
        delete_action.triggered.connect(lambda: self._delete_box(index))
        menu.exec(global_pos)

    def _set_box_class(self, index: int, label: str) -> None:
        entry = self.state.current_entry()
        if entry is None or index >= len(entry.boxes):
            return
        entry.boxes[index].label = label
        self.state.mark_dirty()
        self.state.annotations_changed.emit()
        self.canvas.refresh()
        self._refresh_boxes_list()

    # --------------------------------------------------------------- classes

    def _refresh_classes_ui(self) -> None:
        self.class_list.blockSignals(True)
        self.class_list.clear()
        if self.state.classes:
            for name in self.state.classes:
                item = QListWidgetItem(name)
                item.setData(Qt.ItemDataRole.UserRole, name)
                item.setIcon(color_chip(self.state.color_for(name).name()))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                self.class_list.addItem(item)
        else:
            placeholder = QListWidgetItem("No classes yet — loaded from the model or added manually")
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            placeholder.setForeground(QColor(COLORS["text_faint"]))
            placeholder.setToolTip(placeholder.text())
            self.class_list.addItem(placeholder)
        active = self.state.active_class()
        if active is not None:
            row = self.state.class_index(active)
            if 0 <= row < self.class_list.count():
                self.class_list.setCurrentRow(row)
        self.class_list.blockSignals(False)
        self._refresh_class_counts()

        self.active_class_combo.blockSignals(True)
        self.active_class_combo.clear()
        self.active_class_combo.addItems(self.state.classes)
        if active is not None:
            self.active_class_combo.setCurrentText(active)
        self.active_class_combo.setEnabled(bool(self.state.classes))
        self.active_class_combo.blockSignals(False)
        self._update_active_dot()

    def _refresh_class_counts(self) -> None:
        counts: dict[str, int] = {}
        for entry in self.state.entries:
            for box in entry.boxes:
                counts[box.label] = counts.get(box.label, 0) + 1
        for row in range(self.class_list.count()):
            item = self.class_list.item(row)
            if not (item.flags() & Qt.ItemFlag.ItemIsUserCheckable):
                continue
            name = item.data(Qt.ItemDataRole.UserRole) or item.text()
            count = counts.get(name, 0)
            base = f"{name}   ·   {count}"
            if item.text() != base:
                item.setText(base)

    def _update_active_dot(self) -> None:
        active = self.state.active_class()
        if active:
            pixmap = color_chip(self.state.color_for(active).name(), 12).pixmap(12, 12)
            self.active_dot.setPixmap(pixmap)
        else:
            self.active_dot.clear()

    def _on_active_class_changed(self, name: str) -> None:
        if name:
            self.state.set_active_class(name)
            self._update_active_dot()

    def _on_class_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.state.classes):
            return
        name = self.state.classes[row]
        self.state.set_active_class(name)
        self.active_class_combo.blockSignals(True)
        self.active_class_combo.setCurrentText(name)
        self.active_class_combo.blockSignals(False)
        self._update_active_dot()

    def _on_class_item_changed(self, item: QListWidgetItem) -> None:
        hidden: set[str] = set()
        for row in range(self.class_list.count()):
            row_item = self.class_list.item(row)
            if not (row_item.flags() & Qt.ItemFlag.ItemIsUserCheckable):
                continue
            name = row_item.data(Qt.ItemDataRole.UserRole) or row_item.text()
            if row_item.checkState() != Qt.CheckState.Checked:
                hidden.add(name)
        self.canvas.set_hidden_labels(hidden)

    def _add_class_dialog(self) -> None:
        name, ok = QInputDialog.getText(self, "Add class", "Class name:")
        if ok and name.strip():
            if not self.state.add_class(name):
                QMessageBox.information(self, "Class exists", "That class name is empty or already exists.")

    def _remove_class(self) -> None:
        row = self.class_list.currentRow()
        if row < 0 or row >= len(self.state.classes):
            return
        name = self.state.classes[row]
        if not self.state.remove_class(name):
            QMessageBox.information(
                self,
                "Cannot remove class",
                f"'{name}' is still used by some boxes (or it is the last class).",
            )

    # ---------------------------------------------------------------- model

    def load_model(self, model_path: str | Path | None = None, device: str | None = None) -> None:
        if model_path in (None, ""):
            last_dir = ""
            if self.state.detector_path is not None:
                last_dir = str(self.state.detector_path.parent)
            patterns = " ".join(f"*{ext}" for ext in sorted(MODEL_EXTENSIONS))
            chosen, _ = QFileDialog.getOpenFileName(
                self, "Load YOLO model", last_dir, f"YOLO models ({patterns})"
            )
            if not chosen:
                return
            model_path = chosen

        model_path = Path(model_path)
        if not model_path.is_file():
            QMessageBox.warning(self, "Model not found", f"The model file does not exist:\n{model_path}")
            return

        device = device or self.state.settings.get("device", "auto")
        self.state.busy_changed.emit(True, f"Loading model {model_path.name}…")
        self.model_button.setEnabled(False)
        self._model_worker = ModelLoadWorker(model_path, device)
        self._model_worker.loaded.connect(lambda detector: self._on_model_loaded(detector, model_path))
        self._model_worker.failed.connect(self._on_model_failed)
        self._model_worker.start()

    def _on_model_loaded(self, detector, model_path: Path) -> None:
        self.model_button.setEnabled(True)
        self.state.busy_changed.emit(False, "")
        self.state.set_detector(detector, model_path)
        self.state.settings.remember_model(str(model_path))
        classes = ", ".join(getattr(detector, "names", [])[:6])
        self.state.status_message.emit(
            f"Model loaded: {model_path.name} ({detector.describe()}) — classes: {classes}", 6000
        )

    def _on_model_failed(self, message: str) -> None:
        self.model_button.setEnabled(True)
        self.state.busy_changed.emit(False, "")
        QMessageBox.critical(self, "Model error", message)

    # ------------------------------------------------------------ auto label

    def run_auto_label(self, mode: str = "all") -> None:
        if self._label_worker is not None and self._label_worker.isRunning():
            self._label_worker.cancel()
            self.state.status_message.emit("Stopping auto-labeling…", 3000)
            return
        if self.state.detector is None:
            QMessageBox.information(self, "No model", "Load a model first (Load Model button).")
            return
        if not self.state.entries:
            QMessageBox.information(self, "No images", "Open an image folder first.")
            return

        if mode == "all":
            if any(entry.has_boxes for entry in self.state.entries):
                answer = QMessageBox.question(
                    self,
                    "Replace existing boxes?",
                    "Auto-Label All replaces the current boxes on every image.\n"
                    "Use 'Auto-Label Missing' to keep already labeled images.\n\nContinue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
            indices = list(range(len(self.state.entries)))
        else:
            indices = [i for i, entry in enumerate(self.state.entries) if not entry.has_boxes]

        if not indices:
            self.state.status_message.emit("Nothing to label — every image already has boxes.", 4000)
            return

        conf = self.conf_slider.value()
        iou = self.iou_slider.value()
        imgsz = int(self.imgsz_spin.value())

        self._set_labeling_ui(running=True, total=len(indices))
        self._label_worker = AutoLabelWorker(self.state.detector, self.state.entries, indices, conf, iou, imgsz)
        self._label_worker.progress.connect(self._on_label_progress)
        self._label_worker.image_labeled.connect(self._on_image_labeled)
        self._label_worker.finished_run.connect(self._on_label_finished)
        self._label_worker.failed.connect(lambda message: self.state.status_message.emit(message, 6000))
        self._label_worker.start()

    def _set_labeling_ui(self, running: bool, total: int = 0) -> None:
        has = self.state.has_folder() and self.state.detector is not None
        self.auto_all_button.setEnabled(True if running else has)
        self.auto_missing_button.setEnabled(not running and has)
        self.open_button.setEnabled(not running)
        self.model_button.setEnabled(not running)
        self.smart_filter_button.setEnabled(not running and self.state.has_folder() and bool(self.state.entries))
        self.anomaly_button.setEnabled(not running and self.state.has_folder() and bool(self.state.entries))
        if running:
            self.auto_all_button.setText("Stop")
            self.auto_all_button.setIcon(make_icon("x", "#ffffff"))
            self.state.busy_changed.emit(True, f"Auto-labeling 0/{total}…")
        else:
            self.auto_all_button.setText("Auto-Label All")
            self.auto_all_button.setIcon(make_icon("zap", "#ffffff"))

    def _on_label_progress(self, done: int, total: int) -> None:
        self.state.busy_changed.emit(True, f"Auto-labeling {done}/{total}…")

    def _on_image_labeled(self, index: int, boxes: list, elapsed_ms: float) -> None:
        if index < 0 or index >= len(self.state.entries):
            return
        entry = self.state.entries[index]
        entry.boxes = boxes
        if boxes:
            entry.noise = False
        self._drop_anomaly_marks(index)
        self.state.mark_dirty()
        item = self.image_list.item(index)
        if item is not None:
            item.setData(Qt.ItemDataRole.UserRole + 1, len(boxes))
        if index == self.state.current_index:
            self.canvas.set_boxes(entry.boxes)
            self._refresh_boxes_list()
        self.image_list.viewport().update()

    def _on_label_finished(self, processed: int, total_boxes: int) -> None:
        self._set_labeling_ui(running=False)
        self.state.busy_changed.emit(False, "")
        self.state.annotations_changed.emit()
        self._refresh_class_counts()
        self.state.status_message.emit(
            f"Auto-labeling finished — {processed} images processed, {total_boxes} boxes in total.", 6000
        )

    # ----------------------------------------------------------- smart filter

    def run_smart_filter(self) -> None:
        if self._smart_worker is not None and self._smart_worker.isRunning():
            return
        if not self.state.entries:
            QMessageBox.information(self, "No images", "Open an image folder first.")
            return

        dialog = SmartFilterDialog(self.state, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        options = dialog.options()
        if not options.active():
            self.state.status_message.emit("Smart Filter: nothing selected.", 3000)
            return

        detector = self.state.detector if options.tta_enabled else None
        self._filter_backup = None
        self.undo_filter_button.setVisible(False)
        self.smart_filter_button.setEnabled(False)
        self.state.busy_changed.emit(True, "Smart Filter: scanning boxes…")

        self._smart_worker = SmartFilterWorker(self.state.entries, options, detector)
        self._smart_worker.progress.connect(self._on_smart_filter_progress)
        self._smart_worker.finished_run.connect(self._on_smart_filter_finished)
        self._smart_worker.failed.connect(self._on_smart_filter_failed)
        self._smart_worker.start()

    def _on_smart_filter_progress(self, done: int, total: int) -> None:
        self.state.busy_changed.emit(True, f"Flip-consensus {done}/{total}…")

    def _on_smart_filter_failed(self, message: str) -> None:
        self.smart_filter_button.setEnabled(True)
        self.state.busy_changed.emit(False, "")
        self.state.status_message.emit(f"Smart Filter failed: {message}", 6000)

    def _on_smart_filter_finished(self, removed: int, affected: int, empties: int, backup: object) -> None:
        self.smart_filter_button.setEnabled(True)
        self.state.busy_changed.emit(False, "")
        if not backup:
            self.state.status_message.emit("Smart Filter found nothing to remove.", 4000)
            return

        self._filter_backup = backup
        self.undo_filter_button.setVisible(True)
        for index in backup:
            self._drop_anomaly_marks(index)
        self.state.mark_dirty()
        self.state.annotations_changed.emit()
        self._reload_current_boxes()
        self.state.status_message.emit(
            f"Smart Filter removed {removed} boxes from {affected} images · {empties} flagged as noise. "
            "Use Undo Filter to restore.",
            8000,
        )

    def _undo_smart_filter(self) -> None:
        if not self._filter_backup:
            return
        for index, (boxes, noise) in self._filter_backup.items():
            if 0 <= index < len(self.state.entries):
                entry = self.state.entries[index]
                entry.boxes = list(boxes)
                entry.noise = bool(noise)
                self._drop_anomaly_marks(index)
        self._filter_backup = None
        self.undo_filter_button.setVisible(False)
        self.state.mark_dirty()
        self.state.annotations_changed.emit()
        self._reload_current_boxes()
        self.state.status_message.emit("Smart Filter undone — boxes restored.", 4000)

    def _reload_current_boxes(self) -> None:
        entry = self.state.current_entry()
        if entry is not None:
            self.canvas.set_boxes(entry.boxes)
            self._apply_current_marks()
            self._refresh_boxes_list()

    # --------------------------------------------------------- find unusual

    def open_anomaly_dialog(self) -> None:
        if not self.state.entries:
            QMessageBox.information(self, "No images", "Open an image folder first.")
            return
        if self._anomaly_dialog is None:
            dialog = AnomalyDialog(self.state, self)
            dialog.scan_requested.connect(self._run_anomaly_scan)
            dialog.marks_changed.connect(self._on_anomaly_marks)
            dialog.cleared.connect(self._on_anomaly_cleared)
            dialog.go_to.connect(self._goto_anomaly_box)
            self._anomaly_dialog = dialog
        self._anomaly_dialog.show()
        self._anomaly_dialog.raise_()
        self._anomaly_dialog.activateWindow()
        self._anomaly_dialog.set_results(self._anomaly_hits, self._anomaly_stats)

    def _run_anomaly_scan(self, include_manual: bool) -> None:
        if self._anomaly_worker is not None and self._anomaly_worker.isRunning():
            return
        if not any(entry.boxes for entry in self.state.entries):
            self.state.status_message.emit("Find Unusual: there are no boxes to score yet.", 4000)
            return
        if self._anomaly_dialog is not None:
            self._anomaly_dialog.set_scanning(True)
        self.state.busy_changed.emit(True, "Find Unusual: scoring boxes…")
        self._anomaly_worker = AnomalyWorker(self.state.entries, include_manual)
        self._anomaly_worker.progress.connect(lambda done, total: self.state.busy_changed.emit(True, f"Find Unusual {done}/{total}…"))
        self._anomaly_worker.finished_scan.connect(self._on_anomaly_finished)
        self._anomaly_worker.failed.connect(self._on_anomaly_failed)
        self._anomaly_worker.start()

    def _on_anomaly_finished(self, hits: object, stats: object) -> None:
        if self._anomaly_dialog is not None:
            self._anomaly_dialog.set_scanning(False)
        self.state.busy_changed.emit(False, "")
        self._anomaly_hits = list(hits or [])
        self._anomaly_stats = dict(stats or {})
        if self._anomaly_dialog is not None:
            self._anomaly_dialog.set_results(self._anomaly_hits, self._anomaly_stats)
        flagged = sum(1 for hit in self._anomaly_hits if self._anomaly_dialog is None or hit.score >= self._anomaly_dialog.threshold_spin.value())
        self.state.status_message.emit(
            f"Find Unusual: scored {self._anomaly_stats.get('boxes', 0)} boxes — {flagged} flagged. Double-click a row to jump to the box.",
            8000,
        )

    def _on_anomaly_failed(self, message: str) -> None:
        if self._anomaly_dialog is not None:
            self._anomaly_dialog.set_scanning(False)
        self.state.busy_changed.emit(False, "")
        self.state.status_message.emit(f"Find Unusual failed: {message}", 6000)

    def _on_anomaly_marks(self, marks: object) -> None:
        self._anomaly_marks = dict(marks or {})
        self._apply_current_marks()

    def _on_anomaly_cleared(self) -> None:
        self._anomaly_hits = []
        self._anomaly_stats = {}
        self._anomaly_marks = {}
        self._apply_current_marks()
        self.state.status_message.emit("Find Unusual: marks cleared.", 3000)

    def _goto_anomaly_box(self, entry_index: int, box_index: int) -> None:
        if not (0 <= entry_index < len(self.state.entries)):
            return
        if entry_index != self.state.current_index:
            self.state.set_current(entry_index)
        row_item = self.image_list.item(entry_index)
        if row_item is not None:
            self.image_list.setCurrentItem(row_item)
            self.image_list.scrollToItem(row_item)
        self.canvas.select(box_index)

    def _apply_current_marks(self) -> None:
        index = self.state.current_index
        self.canvas.set_marked(self._anomaly_marks.get(index, {}))

    def _drop_anomaly_marks(self, entry_index: int) -> None:
        if entry_index in self._anomaly_marks:
            del self._anomaly_marks[entry_index]
            if entry_index == self.state.current_index:
                self._apply_current_marks()

    # ---------------------------------------------------------------- misc

    def _on_zoom_changed(self, scale: float) -> None:
        self.zoom_label.setText(f"{scale * 100:.0f}%")

    def _on_mouse_position(self, x: float, y: float) -> None:
        if self.state.current_entry() is None:
            self.coord_label.setText("")
        else:
            self.coord_label.setText(f"x {x:.0f}   y {y:.0f}")

    def _on_device_changed(self, text: str) -> None:
        self.state.settings.set("device", text.lower())

    def _manual_save(self) -> None:
        self.state.save_project()
        self.state.status_message.emit("Annotations saved.", 2500)

    def _cancel_workers(self) -> None:
        for worker in (self._scan_worker, self._thumb_worker, self._label_worker, self._smart_worker, self._anomaly_worker):
            if worker is not None and worker.isRunning():
                worker.cancel()
                worker.wait(1500)
        self._scan_worker = None
        self._thumb_worker = None
        self._label_worker = None
        self._smart_worker = None
        self._anomaly_worker = None
        self._cancelling_model = False
