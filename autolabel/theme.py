from __future__ import annotations

import re

from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import QApplication

COLORS = {
    "bg": "#edf0f5",
    "panel": "#ffffff",
    "panel_border": "#e2e6ed",
    "sidebar": "#151d2e",
    "sidebar_hover": "#1f2a41",
    "sidebar_active": "#2f6bff",
    "sidebar_text": "#8e9bb3",
    "sidebar_title": "#ffffff",
    "text": "#1b2437",
    "text_muted": "#68738a",
    "text_faint": "#98a2b5",
    "accent": "#2f6bff",
    "accent_hover": "#2456d8",
    "accent_soft": "#e9efff",
    "danger": "#df3a3a",
    "danger_soft": "#fdeeee",
    "success": "#1d9e57",
    "success_soft": "#e7f7ee",
    "warning": "#d98014",
    "warning_soft": "#fdf3e3",
    "canvas_bg": "#272c37",
    "input_border": "#d4dae4",
    "item_hover": "#f3f6fa",
}

STYLESHEET = """
* { outline: none; }
QWidget {
    font-family: "Segoe UI";
    font-size: 13px;
    color: @text;
}
QMainWindow, QDialog { background: @bg; }

/* ---------- Sidebar ---------- */
#Sidebar { background: @sidebar; }
#Sidebar QLabel { color: @sidebar_text; }
#Sidebar QLabel[class="brand"] {
    color: @sidebar_title;
    font-size: 17px;
    font-weight: 700;
}
#Sidebar QLabel[class="brand-sub"] {
    color: #5f6c85;
    font-size: 11px;
    font-weight: 600;
}
#Sidebar QFrame[class="card"] {
    background: @sidebar_hover;
    border: 1px solid #26324c;
    border-radius: 10px;
}
#Sidebar QLabel[class="model-name"] { color: #e4e9f2; font-weight: 600; font-size: 12px; }
#Sidebar QLabel[class="model-meta"] { color: #7a879e; font-size: 11px; }
#Sidebar QPushButton[role="nav"] {
    color: @sidebar_text;
    background: transparent;
    border: none;
    border-radius: 8px;
    padding: 9px 14px;
    text-align: left;
    font-size: 13.5px;
    font-weight: 600;
}
#Sidebar QPushButton[role="nav"]:hover { background: @sidebar_hover; color: #dbe2ee; }
#Sidebar QPushButton[role="nav"]:checked { background: @sidebar_active; color: #ffffff; }
QPushButton[role="sidebar-ghost"] {
    color: @sidebar_text;
    background: transparent;
    border: 1px dashed #33405e;
    border-radius: 8px;
    padding: 7px 12px;
    font-size: 12px;
}
QPushButton[role="sidebar-ghost"]:hover { color: #ffffff; border-color: #47547a; }

/* ---------- Buttons ---------- */
QPushButton {
    background: @panel;
    border: 1px solid @input_border;
    border-radius: 7px;
    padding: 6px 14px;
    font-weight: 600;
    color: @text;
}
QPushButton:hover { background: #f6f8fb; border-color: #c3cbd8; }
QPushButton:pressed { background: #eef1f6; }
QPushButton:disabled { color: @text_faint; background: #f2f4f7; border-color: @panel_border; }
QPushButton[role="primary"] {
    background: @accent;
    border-color: @accent;
    color: white;
    padding: 7px 18px;
}
QPushButton[role="primary"]:hover { background: @accent_hover; border-color: @accent_hover; }
QPushButton[role="primary"]:disabled { background: #a9bdf3; border-color: #a9bdf3; color: #f2f5ff; }
QPushButton[role="danger"] { color: @danger; border-color: #f0c4c4; background: @panel; }
QPushButton[role="danger"]:hover { background: @danger_soft; }
QPushButton[role="ghost"] { background: transparent; border: none; color: @text_muted; }
QPushButton[role="ghost"]:hover { background: @item_hover; color: @text; }
QToolButton {
    background: transparent;
    border: none;
    border-radius: 6px;
    padding: 5px;
}
QToolButton:hover { background: @item_hover; }
QToolButton:checked { background: @accent_soft; }
QToolButton:disabled { opacity: 0.4; }

/* ---------- Cards / labels ---------- */
QFrame[class="card"] {
    background: @panel;
    border: 1px solid @panel_border;
    border-radius: 10px;
}
QFrame[class="hline"] { background: @panel_border; border: none; }
QLabel[class="section"] {
    color: @text_muted;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}
QLabel[class="h1"] { font-size: 17px; font-weight: 700; }
QLabel[class="muted"] { color: @text_muted; font-size: 12px; }
QLabel[class="faint"] { color: @text_faint; font-size: 11.5px; }
QLabel[class="value"] { font-size: 21px; font-weight: 700; }
QLabel[class="empty-title"] { font-size: 15px; font-weight: 700; color: #3c4657; }
QLabel[class="chip"] {
    background: #eef1f6;
    color: @text_muted;
    border-radius: 9px;
    padding: 2px 10px;
    font-size: 11.5px;
    font-weight: 600;
}
QLabel[class="chip-ok"] { background: @success_soft; color: @success; }
QLabel[class="chip-warn"] { background: @warning_soft; color: @warning; }
QLabel[class="chip-err"] { background: @danger_soft; color: @danger; }

/* ---------- Inputs ---------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: @panel;
    border: 1px solid @input_border;
    border-radius: 7px;
    padding: 5px 9px;
    selection-background-color: @accent;
    selection-color: white;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: @accent; }
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    background: #f2f4f7;
    color: @text_faint;
}
QComboBox::drop-down { border: none; width: 22px; }
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid @text_muted;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background: @panel;
    border: 1px solid @panel_border;
    border-radius: 8px;
    padding: 4px;
    selection-background-color: @accent_soft;
    selection-color: @accent;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    border: none;
    border-left: 1px solid @panel_border;
    border-top-right-radius: 7px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    border: none;
    border-left: 1px solid @panel_border;
    border-bottom-right-radius: 7px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: @item_hover; }

QCheckBox { spacing: 7px; color: @text; }
QCheckBox::indicator {
    width: 15px; height: 15px;
    border: 1.5px solid #c3cbd8;
    border-radius: 4px;
    background: white;
}
QCheckBox::indicator:hover { border-color: @accent; }
QCheckBox::indicator:checked { background: @accent; border-color: @accent; }

QSlider::groove:horizontal { height: 4px; background: #e3e8f0; border-radius: 2px; }
QSlider::sub-page:horizontal { background: @accent; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 14px; height: 14px;
    margin: -6px 0;
    border-radius: 7px;
    background: white;
    border: 2px solid @accent;
}
QSlider::handle:horizontal:hover { background: @accent_soft; }
QSlider::handle:horizontal:disabled { border-color: #c3cbd8; }

/* ---------- Lists / tables ---------- */
QListWidget, QTableWidget, QTreeWidget {
    background: transparent;
    border: none;
    padding: 2px;
}
QListWidget::item, QTreeWidget::item {
    border-radius: 7px;
    padding: 5px 7px;
    margin: 1px 2px;
}
QListWidget::item:hover, QTreeWidget::item:hover { background: @item_hover; }
QListWidget::item:selected, QTreeWidget::item:selected {
    background: @accent_soft;
    color: @text;
}
QTableWidget { gridline-color: transparent; alternate-background-color: #fafbfd; }
QTableWidget::item { padding: 4px 6px; border: none; border-radius: 4px; color: @text; }
QTableWidget::item:selected { background: @accent_soft; color: @text; }
QHeaderView::section {
    background: transparent;
    color: @text_muted;
    font-size: 11px;
    font-weight: 700;
    border: none;
    border-bottom: 1px solid @panel_border;
    padding: 6px 6px;
}
QTableCornerButton::section { background: transparent; border: none; border-bottom: 1px solid @panel_border; }

/* ---------- Scrollbars ---------- */
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #c8cfda; border-radius: 4px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #aeb7c6; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: #c8cfda; border-radius: 4px; min-width: 30px; }
QScrollBar::handle:horizontal:hover { background: #aeb7c6; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
#Sidebar QScrollBar::handle:vertical { background: #33405e; }

/* ---------- Splitter / status bar ---------- */
QSplitter::handle { background: transparent; width: 5px; height: 5px; }
QSplitter::handle:hover { background: #d8dde6; }
QStatusBar {
    background: @panel;
    border-top: 1px solid @panel_border;
    color: @text_muted;
    font-size: 12px;
}
QStatusBar::item { border: none; }

/* ---------- Menus / tooltips / progress ---------- */
QMenu {
    background: @panel;
    border: 1px solid @panel_border;
    border-radius: 8px;
    padding: 5px;
}
QMenu::item { padding: 6px 12px; border-radius: 6px; }
QMenu::item:selected { background: @accent_soft; color: @accent; }
QMenu::separator { height: 1px; background: @panel_border; margin: 4px 6px; }
QToolTip {
    background: #1c2434;
    color: #e8ecf4;
    border: none;
    border-radius: 5px;
    padding: 6px 9px;
    font-size: 12px;
}
QProgressBar {
    background: #e8ecf3;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}
QProgressBar::chunk { background: @accent; border-radius: 4px; }

/* ---------- Toolbars ---------- */
#CanvasToolbar { background: @panel; border: 1px solid @panel_border; border-radius: 10px; }
#CanvasToolbar QLabel { color: @text_muted; font-size: 12px; font-weight: 600; }
#TopBar { background: @panel; border-bottom: 1px solid @panel_border; }
"""

TOKEN_PATTERN = re.compile(r"@(\w+)")


def _expand(match: re.Match) -> str:
    name = match.group(1)
    if name not in COLORS:
        raise KeyError(f"Unknown theme token: @{name}")
    return COLORS[name]


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    font = QFont("Segoe UI", 9)
    font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    app.setFont(font)
    app.setStyleSheet(TOKEN_PATTERN.sub(_expand, STYLESHEET))


def color(name: str) -> QColor:
    return QColor(COLORS[name])
