from __future__ import annotations

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from ..models import Box
from ..theme import COLORS

HANDLE_NONE = -1
HANDLES = [
    "top-left", "top", "top-right",
    "right", "bottom-right", "bottom",
    "bottom-left", "left",
]

HANDLE_CURSORS = {
    "top-left": Qt.CursorShape.SizeFDiagCursor,
    "bottom-right": Qt.CursorShape.SizeFDiagCursor,
    "top-right": Qt.CursorShape.SizeBDiagCursor,
    "bottom-left": Qt.CursorShape.SizeBDiagCursor,
    "top": Qt.CursorShape.SizeVerCursor,
    "bottom": Qt.CursorShape.SizeVerCursor,
    "left": Qt.CursorShape.SizeHorCursor,
    "right": Qt.CursorShape.SizeHorCursor,
}


class ImageCanvas(QWidget):
    box_created = pyqtSignal(float, float, float, float)
    box_modified = pyqtSignal(int)
    box_delete_requested = pyqtSignal(int)
    selection_changed = pyqtSignal(int)
    zoom_changed = pyqtSignal(float)
    mouse_position = pyqtSignal(float, float)
    mouse_left = pyqtSignal()
    context_menu_requested = pyqtSignal(int, QPoint)

    MIN_SCALE = 0.02
    MAX_SCALE = 24.0

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._image: QImage | None = None
        self._boxes: list[Box] = []
        self._color_for = lambda label: QColor(COLORS["accent"])
        self._hidden_labels: set[str] = set()
        self._marked: dict[int, float] = {}

        self._scale = 1.0
        self._origin = QPointF(0, 0)

        self._selected = -1
        self._hover = -1

        self._mode = "idle"
        self._pan_start = QPointF()
        self._pan_origin = QPointF()
        self._space_held = False
        self._press_pos = QPointF()
        self._move_start_boxes: tuple | None = None
        self._resize_handle = ""
        self._resize_start_box: tuple | None = None
        self._draw_start = QPointF()
        self._draw_current = QPointF()
        self._press_on_box = False

    # ------------------------------------------------------------------ data

    def set_image(self, image: QImage | None, boxes: list[Box] | None = None) -> None:
        self._image = image
        self._boxes = boxes if boxes is not None else []
        self._marked = {}
        self._selected = -1
        self._hover = -1
        self.fit_to_window()
        self.selection_changed.emit(-1)
        self.update()

    def set_boxes(self, boxes: list[Box], keep_selection: bool = False) -> None:
        selected = self._selected if keep_selection else -1
        self._boxes = boxes
        self._selected = selected if 0 <= selected < len(boxes) else -1
        self._hover = -1
        self.update()

    def refresh(self) -> None:
        self.update()

    def set_color_provider(self, provider) -> None:
        self._color_for = provider

    def set_hidden_labels(self, hidden: set[str]) -> None:
        self._hidden_labels = set(hidden)
        self.update()

    def set_marked(self, marks: dict[int, float] | None) -> None:
        self._marked = dict(marks or {})
        self.update()

    def selected_index(self) -> int:
        return self._selected

    def select(self, index: int, emit: bool = True) -> None:
        index = index if 0 <= index < len(self._boxes) else -1
        if self._selected != index:
            self._selected = index
            if emit:
                self.selection_changed.emit(index)
            self.update()

    def has_image(self) -> bool:
        return self._image is not None

    def image_size(self) -> tuple[int, int]:
        if self._image is None:
            return 0, 0
        return self._image.width(), self._image.height()

    # ------------------------------------------------------------- transform

    def _to_widget(self, x: float, y: float) -> QPointF:
        return QPointF(x * self._scale + self._origin.x(), y * self._scale + self._origin.y())

    def _to_image(self, pos: QPointF) -> QPointF:
        return QPointF((pos.x() - self._origin.x()) / self._scale, (pos.y() - self._origin.y()) / self._scale)

    def fit_to_window(self) -> None:
        if self._image is None or self._image.isNull():
            self._scale = 1.0
            self._origin = QPointF(0, 0)
            self.update()
            return
        margin = 16
        available_w = max(1, self.width() - margin * 2)
        available_h = max(1, self.height() - margin * 2)
        scale = min(available_w / self._image.width(), available_h / self._image.height())
        self._scale = min(max(scale, self.MIN_SCALE), self.MAX_SCALE)
        self._center()
        self.zoom_changed.emit(self._scale)
        self.update()

    def _center(self) -> None:
        if self._image is None:
            return
        w = self._image.width() * self._scale
        h = self._image.height() * self._scale
        self._origin = QPointF((self.width() - w) / 2, (self.height() - h) / 2)

    def zoom_by(self, factor: float, anchor: QPointF | None = None) -> None:
        if self._image is None:
            return
        old_scale = self._scale
        new_scale = min(max(old_scale * factor, self.MIN_SCALE), self.MAX_SCALE)
        if abs(new_scale - old_scale) < 1e-9:
            return
        if anchor is None:
            anchor = QPointF(self.width() / 2, self.height() / 2)
        image_point = self._to_image(anchor)
        self._scale = new_scale
        self._origin = QPointF(
            anchor.x() - image_point.x() * new_scale,
            anchor.y() - image_point.y() * new_scale,
        )
        self.zoom_changed.emit(self._scale)
        self.update()

    # ------------------------------------------------------------- painters

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(COLORS["canvas_bg"]))

        if self._image is None or self._image.isNull():
            painter.setPen(QColor("#5b6577"))
            font = QFont("Segoe UI", 11)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded\nOpen a folder to start annotating")
            painter.end()
            return

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        target = QRectF(
            self._origin.x(),
            self._origin.y(),
            self._image.width() * self._scale,
            self._image.height() * self._scale,
        )
        painter.drawImage(target, self._image)

        for index, box in enumerate(self._boxes):
            if box.label in self._hidden_labels:
                continue
            self._paint_box(painter, index, box)

        if self._mode == "drawing":
            self._paint_drawing(painter)

        painter.end()

    def _paint_box(self, painter: QPainter, index: int, box: Box) -> None:
        color = self._color_for(box.label)
        selected = index == self._selected
        hovered = index == self._hover
        marked = index in self._marked

        top_left = self._to_widget(box.x1, box.y1)
        bottom_right = self._to_widget(box.x2, box.y2)
        rect = QRectF(top_left, bottom_right)

        pen = QPen(color)
        pen.setWidthF(2.5 if selected else (2.2 if hovered else 1.8))
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

        if marked:
            mark_pen = QPen(QColor(COLORS["danger"]))
            mark_pen.setWidthF(2.4)
            mark_pen.setStyle(Qt.PenStyle.DashLine)
            mark_pen.setCosmetic(True)
            painter.setPen(mark_pen)
            painter.drawRect(rect.adjusted(-3.0, -3.0, 3.0, 3.0))
            painter.setPen(pen)

        if selected:
            painter.fillRect(rect, QColor(color.red(), color.green(), color.blue(), 26))

        tag_text = box.label
        if box.conf is not None:
            tag_text = f"{box.label}  {box.conf * 100:.0f}%"
        if marked:
            tag_text = "! " + tag_text

        font = QFont("Segoe UI", 8)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(tag_text) + 12
        text_height = metrics.height() + 4

        tag_rect = QRectF(rect.left(), rect.top() - text_height - 2, text_width, text_height)
        if tag_rect.top() < 0:
            tag_rect.moveTop(rect.top() + 2)

        tag_color = QColor(color)
        tag_color.setAlpha(235 if selected or hovered else 210)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(tag_color)
        radius = min(4.0, text_height / 2)
        painter.drawRoundedRect(tag_rect, radius, radius)

        painter.setPen(QColor("#ffffff"))
        painter.drawText(tag_rect, Qt.AlignmentFlag.AlignCenter, tag_text)

        if selected:
            self._paint_handles(painter, rect, color)

    def _paint_handles(self, painter: QPainter, rect: QRectF, color: QColor) -> None:
        size = 7.0
        painter.setPen(QPen(QColor(COLORS["accent"]), 1.4))
        painter.setBrush(QColor("#ffffff"))
        for point in self._handle_points(rect).values():
            painter.drawRect(QRectF(point.x() - size / 2, point.y() - size / 2, size, size))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _paint_drawing(self, painter: QPainter) -> None:
        rect = QRectF(self._to_widget(*self._draw_start), self._to_widget(*self._draw_current)).normalized()
        pen = QPen(QColor(COLORS["accent"]))
        pen.setWidthF(1.8)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        fill = QColor(COLORS["accent"])
        fill.setAlpha(30)
        painter.setBrush(fill)
        painter.drawRect(rect)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    # ------------------------------------------------------------ hit tests

    def _handle_points(self, rect: QRectF) -> dict[str, QPointF]:
        cx = rect.center().x()
        cy = rect.center().y()
        return {
            "top-left": QPointF(rect.left(), rect.top()),
            "top": QPointF(cx, rect.top()),
            "top-right": QPointF(rect.right(), rect.top()),
            "right": QPointF(rect.right(), cy),
            "bottom-right": QPointF(rect.right(), rect.bottom()),
            "bottom": QPointF(cx, rect.bottom()),
            "bottom-left": QPointF(rect.left(), rect.bottom()),
            "left": QPointF(rect.left(), cy),
        }

    def _hit_handle(self, pos: QPointF) -> str | None:
        if self._selected < 0 or self._selected >= len(self._boxes):
            return None
        box = self._boxes[self._selected]
        rect = QRectF(self._to_widget(box.x1, box.y1), self._to_widget(box.x2, box.y2))
        for name, point in self._handle_points(rect).items():
            if abs(pos.x() - point.x()) <= 6 and abs(pos.y() - point.y()) <= 6:
                return name
        return None

    def _hit_box(self, pos: QPointF) -> int:
        image_pos = self._to_image(pos)
        for index in range(len(self._boxes) - 1, -1, -1):
            box = self._boxes[index]
            if box.label in self._hidden_labels:
                continue
            if box.x1 <= image_pos.x() <= box.x2 and box.y1 <= image_pos.y() <= box.y2:
                return index
        return -1

    # ---------------------------------------------------------------- events

    def mousePressEvent(self, event):
        if self._image is None:
            return
        pos = event.position()
        self._press_pos = pos

        if event.button() == Qt.MouseButton.MiddleButton or (self._space_held and event.button() == Qt.MouseButton.LeftButton):
            self._mode = "panning"
            self._pan_start = pos
            self._pan_origin = QPointF(self._origin)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        handle = self._hit_handle(pos)
        if handle:
            self._mode = "resizing"
            self._resize_handle = handle
            box = self._boxes[self._selected]
            self._resize_start_box = (box.x1, box.y1, box.x2, box.y2)
            return

        index = self._hit_box(pos)
        if index >= 0:
            self._press_on_box = True
            self.select(index)
            self._mode = "moving"
            box = self._boxes[index]
            self._move_start_boxes = (box.x1, box.y1, box.x2, box.y2)
            self.setCursor(Qt.CursorShape.SizeAllCursor)
            return

        self._press_on_box = False
        self.select(-1)
        self._mode = "drawing"
        image_pos = self._to_image(pos)
        self._draw_start = (image_pos.x(), image_pos.y())
        self._draw_current = self._draw_start
        self.update()

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self._image is not None:
            image_pos = self._to_image(pos)
            self.mouse_position.emit(image_pos.x(), image_pos.y())

        if self._mode == "panning":
            delta = pos - self._pan_start
            self._origin = self._pan_origin + delta
            self.update()
            return

        if self._mode == "moving" and self._move_start_boxes is not None:
            delta = self._to_image(pos) - self._to_image(self._press_pos)
            x1, y1, x2, y2 = self._move_start_boxes
            box = self._boxes[self._selected]
            box.x1, box.y1, box.x2, box.y2 = x1 + delta.x(), y1 + delta.y(), x2 + delta.x(), y2 + delta.y()
            self._clamp_box(box)
            self.update()
            return

        if self._mode == "resizing" and self._resize_start_box is not None:
            self._apply_resize(pos)
            self.update()
            return

        if self._mode == "drawing":
            image_pos = self._to_image(pos)
            self._draw_current = (image_pos.x(), image_pos.y())
            self.update()
            return

        handle = self._hit_handle(pos)
        if handle:
            self.setCursor(HANDLE_CURSORS.get(handle, Qt.CursorShape.CrossCursor))
        else:
            index = self._hit_box(pos)
            if index != self._hover:
                self._hover = index
                self.update()
            if index >= 0:
                self.setCursor(Qt.CursorShape.SizeAllCursor if index == self._selected else Qt.CursorShape.PointingHandCursor)
            else:
                self.setCursor(Qt.CursorShape.OpenHandCursor if self._space_held else Qt.CursorShape.CrossCursor)

    def mouseReleaseEvent(self, event):
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            return
        pos = event.position()
        mode = self._mode
        self._mode = "idle"

        if mode == "panning":
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._space_held else Qt.CursorShape.CrossCursor)
            return

        if mode == "moving":
            if self._selected >= 0:
                box = self._boxes[self._selected]
                x1, y1, x2, y2 = self._move_start_boxes
                moved = (abs(box.x1 - x1) > 0.5 or abs(box.y1 - y1) > 0.5 or abs(box.x2 - x2) > 0.5 or abs(box.y2 - y2) > 0.5)
                if moved:
                    self.box_modified.emit(self._selected)
            self._move_start_boxes = None
            self.setCursor(Qt.CursorShape.SizeAllCursor)
            return

        if mode == "resizing":
            self.box_modified.emit(self._selected)
            self._resize_start_box = None
            self.setCursor(Qt.CursorShape.CrossCursor)
            return

        if mode == "drawing":
            start = self._to_image(self._press_pos)
            end = self._to_image(pos)
            rect = QRectF(start, end).normalized()
            if rect.width() >= 3 and rect.height() >= 3:
                self.box_created.emit(rect.left(), rect.top(), rect.right(), rect.bottom())
            self.update()
            return

    def _apply_resize(self, pos: QPointF) -> None:
        x1, y1, x2, y2 = self._resize_start_box
        image_pos = self._to_image(pos)
        px, py = image_pos.x(), image_pos.y()
        min_size = 2.0

        if "left" in self._resize_handle:
            x1 = min(px, x2 - min_size)
        if "right" in self._resize_handle:
            x2 = max(px, x1 + min_size)
        if "top" in self._resize_handle:
            y1 = min(py, y2 - min_size)
        if "bottom" in self._resize_handle:
            y2 = max(py, y1 + min_size)

        box = self._boxes[self._selected]
        box.x1, box.y1, box.x2, box.y2 = x1, y1, x2, y2
        self._clamp_box(box)

    def _clamp_box(self, box: Box) -> None:
        width, height = self.image_size()
        if width == 0:
            return
        box_width = box.x2 - box.x1
        box_height = box.y2 - box.y1
        box.x1 = min(max(box.x1, 0.0), width - box_width)
        box.y1 = min(max(box.y1, 0.0), height - box_height)
        box.x2 = box.x1 + box_width
        box.y2 = box.y1 + box_height

    def contextMenuEvent(self, event):
        if self._image is None:
            return
        index = self._hit_box(QPointF(event.pos()))
        if index >= 0:
            self.select(index)
        self.context_menu_requested.emit(index, event.globalPos())

    def wheelEvent(self, event):
        if self._image is None:
            return
        steps = event.angleDelta().y() / 120.0
        factor = 1.25 ** steps
        self.zoom_by(factor, event.position())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self._space_held = True
            if self._mode == "idle":
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        if event.key() == Qt.Key.Key_Escape:
            if self._mode == "drawing":
                self._mode = "idle"
                self.update()
            else:
                self.select(-1)
            return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self._selected >= 0:
                self.box_delete_requested.emit(self._selected)
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self._space_held = False
            if self._mode == "idle":
                self.setCursor(Qt.CursorShape.CrossCursor)
        super().keyReleaseEvent(event)

    def leaveEvent(self, event):
        self.mouse_left.emit()
        if self._hover != -1:
            self._hover = -1
            self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._image is not None and self._mode == "idle":
            pass
