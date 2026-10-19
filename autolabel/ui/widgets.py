from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .icons import make_icon
from ..theme import COLORS


def card(parent: QWidget | None = None) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame(parent)
    frame.setProperty("class", "card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 12, 14, 14)
    layout.setSpacing(10)
    return frame, layout


def section_label(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setProperty("class", "section")
    return label


def muted_label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setProperty("class", "muted")
    return label


def faint_label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setProperty("class", "faint")
    label.setWordWrap(True)
    return label


class ElidedLabel(QLabel):
    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(80)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        elided = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideMiddle, self.width())
        painter.setPen(self.palette().windowText().color())
        painter.drawText(self.rect(), int(self.alignment()), elided)


def hline() -> QFrame:
    line = QFrame()
    line.setProperty("class", "hline")
    line.setFixedHeight(1)
    return line


def make_button(text: str, icon_name: str | None = None, role: str | None = None, icon_color: str | None = None) -> QPushButton:
    button = QPushButton(text)
    if role:
        button.setProperty("role", role)
    if icon_name:
        if icon_color is None:
            icon_color = "#ffffff" if role == "primary" else COLORS["text_muted"]
        button.setIcon(make_icon(icon_name, icon_color))
    return button


def icon_button(icon_name: str, tooltip: str = "", color: str | None = None, size: int = 18, checkable: bool = False) -> QToolButton:
    button = QToolButton()
    button.setIcon(make_icon(icon_name, color or COLORS["text_muted"], size))
    button.setIconSize(button.iconSize())
    if tooltip:
        button.setToolTip(tooltip)
    button.setCheckable(checkable)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


class StatusDot(QWidget):
    def __init__(self, diameter: int = 9, parent: QWidget | None = None):
        super().__init__(parent)
        self._diameter = diameter
        self._color = QColor(COLORS["text_faint"])
        self.setFixedSize(diameter + 4, diameter + 4)

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._color)
        offset = 2
        painter.drawEllipse(offset, offset, self._diameter, self._diameter)


class StatCard(QFrame):
    def __init__(self, caption: str, value: str = "0", parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("class", "card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(2)
        self.value_label = QLabel(value)
        self.value_label.setProperty("class", "value")
        self.caption_label = QLabel(caption.upper())
        self.caption_label.setProperty("class", "section")
        layout.addWidget(self.value_label)
        layout.addWidget(self.caption_label)

    def set_value(self, value) -> None:
        self.value_label.setText(str(value))

    def set_caption(self, caption: str) -> None:
        self.caption_label.setText(caption.upper())


class ValueSlider(QWidget):
    value_changed = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        value: float,
        decimals: int = 2,
        step: float = 0.01,
        suffix: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._decimals = decimals
        self._scale = 10 ** decimals

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.label = QLabel(label)
        self.label.setMinimumWidth(66)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(minimum * self._scale), int(maximum * self._scale))
        self.spin = QDoubleSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setSuffix(suffix)
        self.spin.setFixedWidth(92)
        self.spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)
        self.set_value(value)

    def _on_slider(self, raw: int) -> None:
        value = raw / self._scale
        if abs(self.spin.value() - value) > 1e-9:
            self.spin.blockSignals(True)
            self.spin.setValue(value)
            self.spin.blockSignals(False)
        self.value_changed.emit(value)

    def _on_spin(self, value: float) -> None:
        raw = int(round(value * self._scale))
        if self.slider.value() != raw:
            self.slider.blockSignals(True)
            self.slider.setValue(raw)
            self.slider.blockSignals(False)
        self.value_changed.emit(value)

    def value(self) -> float:
        return self.spin.value()

    def set_value(self, value: float) -> None:
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(value * self._scale)))
        self.slider.blockSignals(False)


class SpinRow(QWidget):
    value_changed = pyqtSignal(int)

    def __init__(self, label: str, minimum: int, maximum: int, value: int, suffix: str = " px", parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.label = QLabel(label)
        self.label.setMinimumWidth(66)
        self.label.setProperty("class", "muted")
        self.spin = QSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setSuffix(suffix)
        self.spin.setValue(value)
        self.spin.setFixedWidth(110)
        self.spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self.label)
        layout.addStretch(1)
        layout.addWidget(self.spin)

        self.spin.valueChanged.connect(self.value_changed.emit)

    def value(self) -> int:
        return self.spin.value()

    def set_value(self, value: int) -> None:
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)

    def set_enabled(self, enabled: bool) -> None:
        self.spin.setEnabled(enabled)
        self.label.setEnabled(enabled)


def color_chip(color: str, size: int = 12) -> QIcon:
    pixmap = QPixmap(size * 2, size * 2)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(color))
    painter.drawRoundedRect(2, 2, size * 2 - 4, size * 2 - 4, 4, 4)
    painter.end()
    pixmap.setDevicePixelRatio(2)
    return QIcon(pixmap)


class EmptyState(QWidget):
    def __init__(self, icon_name: str, title: str, subtitle: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self._action: QPushButton | None = None

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(8)

        icon_label = QLabel()
        icon_label.setPixmap(make_icon(icon_name, "#a6b0c3", 56).pixmap(56, 56))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_label = QLabel(title)
        title_label.setProperty("class", "empty-title")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle_label = QLabel(subtitle)
        subtitle_label.setProperty("class", "muted")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setWordWrap(True)
        subtitle_label.setMaximumWidth(420)
        subtitle_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        self.button_row = QHBoxLayout()
        self.button_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addStretch(2)
        layout.addWidget(icon_label)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addSpacing(6)
        layout.addLayout(self.button_row)
        layout.addStretch(3)

    def add_action(self, button: QPushButton) -> None:
        self.button_row.addWidget(button)
