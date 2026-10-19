from __future__ import annotations

from PyQt6.QtCore import QByteArray, Qt
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

# name: (path data, filled)
ICONS: dict[str, tuple[str, bool]] = {
    "folder": ("M3.5 7.5a2 2 0 0 1 2-2h3.4l1.8 2h6.8a2 2 0 0 1 2 2v7.5a2 2 0 0 1-2 2h-12a2 2 0 0 1-2-2z", False),
    "cpu": ("M7 7h10v10H7z M10 10h4v4h-4z M12 3.5V7 M12 17v3.5 M3.5 12H7 M17 12h3.5", False),
    "play": ("M8.5 5.8 18 12l-9.5 6.2z", True),
    "zap": ("M13 2.5 5 13.5h5.5L10 21.5l8-11h-5.5z", True),
    "save": ("M6 3.5h9l3.5 3.5v12a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 4.5 19V5A1.5 1.5 0 0 1 6 3.5z M8 3.5V9h7V3.5 M8 20.5v-5.5h8v5.5", False),
    "download": ("M12 3.5v11 M7.5 10.2 12 14.7l4.5-4.5 M4.5 19.5h15", False),
    "trash": ("M4.5 7h15 M9.5 7V4.5h5V7 M6.5 7l1 12.5h9l1-12.5 M10.2 11v5.5 M13.8 11v5.5", False),
    "chevron-left": ("M14.5 6 9 12l5.5 6", False),
    "chevron-right": ("M9.5 6 15 12l-5.5 6", False),
    "zoom-in": ("M10.5 4a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13z M15.6 15.6 20 20 M10.5 8v5 M8 10.5h5", False),
    "zoom-out": ("M10.5 4a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13z M15.6 15.6 20 20 M8 10.5h5", False),
    "fit": ("M4 9V5.5A1.5 1.5 0 0 1 5.5 4H9 M15 4h3.5A1.5 1.5 0 0 1 20 5.5V9 M20 15v3.5a1.5 1.5 0 0 1-1.5 1.5H15 M9 20H5.5A1.5 1.5 0 0 1 4 18.5V15", False),
    "search": ("M10.5 4a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13z M15.6 15.6 20 20", False),
    "check": ("M5 12.5 10 17.5 19 7", False),
    "x": ("M6.5 6.5l11 11 M17.5 6.5l-11 11", False),
    "eye": ("M2.5 12S6 5.8 12 5.8 21.5 12 21.5 12 18 18.2 12 18.2 2.5 12 2.5 12z M12 9.4a2.6 2.6 0 1 0 0 5.2 2.6 2.6 0 0 0 0-5.2z", False),
    "eye-off": ("M2.5 12S6 5.8 12 5.8c1.5 0 2.9.4 4.1 1 M21.5 12S18 18.2 12 18.2c-1.5 0-2.9-.4-4.1-1 M4 4l16 16", False),
    "plus": ("M12 5.5v13 M5.5 12h13", False),
    "minus": ("M5.5 12h13", False),
    "refresh": ("M19.5 12a7.5 7.5 0 1 1-2.2-5.3 M19.5 4.5V9H15", False),
    "info": ("M12 3.5a8.5 8.5 0 1 0 0 17 8.5 8.5 0 0 0 0-17z M12 11v5.5 M12 7.8v.05", False),
    "alert": ("M12 3.5a8.5 8.5 0 1 0 0 17 8.5 8.5 0 0 0 0-17z M12 7.5V13 M12 16.2v.05", False),
    "image": ("M5.5 5h13a1.5 1.5 0 0 1 1.5 1.5v11a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 17.5v-11A1.5 1.5 0 0 1 5.5 5z M8.6 10.2a1.4 1.4 0 1 0 0-2.8 1.4 1.4 0 0 0 0 2.8z M4.3 16.2 9 12l3.2 3 4-3.6 4.4 4", False),
    "sliders": ("M4 8h5.6 M14.4 8h5.6 M4 16h1.6 M10.4 16h9.6 M12 8m-2.2 0a2.2 2.2 0 1 0 4.4 0a2.2 2.2 0 1 0-4.4 0 M8 16m-2.2 0a2.2 2.2 0 1 0 4.4 0a2.2 2.2 0 1 0-4.4 0", False),
    "tag": ("M4 4h6.8l9.2 9.2-6.8 6.8L4 10.8z M8 8m-1.3 0a1.3 1.3 0 1 0 2.6 0a1.3 1.3 0 1 0-2.6 0", False),
    "layers": ("M12 3.5 3.5 8l8.5 4.5L20.5 8z M3.5 12.5 12 17l8.5-4.5 M3.5 16.5 12 21l8.5-4.5", False),
    "edit": ("M4 20l1-4L16.5 4.5a2.1 2.1 0 0 1 3 3L8 19z M14 6.5l3.5 3.5", False),
    "crosshair": ("M12 3.5v4 M12 16.5v4 M3.5 12h4 M16.5 12h4 M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7z", False),
    "box": ("M5 5h14v14H5z", False),
}

_cache: dict[tuple[str, str, int], QIcon] = {}


def make_icon(name: str, color: str = "#68738a", size: int = 18, stroke_width: float = 1.7) -> QIcon:
    key = (name, color, size)
    if key in _cache:
        return _cache[key]

    path_data, filled = ICONS.get(name, ICONS["box"])
    if filled:
        body = f'fill="{color}" stroke="none"'
    else:
        body = f'fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"'
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
        f'<path {body} d="{path_data}"/></svg>'
    )

    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    ratio = 2
    pixmap = QPixmap(size * ratio, size * ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(ratio)

    icon = QIcon(pixmap)
    _cache[key] = icon
    return icon
