from pathlib import Path

APP_NAME = "AutoLabeler"
APP_VERSION = "1.0.0"
ORG_NAME = "AutoLabeler"

APP_DIR = Path.home() / ".yolo-autolabeler"
PROJECTS_DIR = APP_DIR / "projects"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".tif", ".tiff", ".jfif", ".pjpeg", ".avif",
}

MODEL_EXTENSIONS = {".pt", ".onnx"}

CLASS_COLORS = [
    "#f87171", "#fb923c", "#fbbf24", "#a3e635",
    "#34d399", "#2dd4bf", "#38bdf8", "#60a5fa",
    "#818cf8", "#a78bfa", "#e879f9", "#f472b6",
    "#facc15", "#4ade80", "#22d3ee", "#93c5fd",
    "#c4b5fd", "#fda4af", "#fdba74", "#bef264",
]

DEFAULT_SETTINGS = {
    "conf": 0.25,
    "iou": 0.45,
    "imgsz": 640,
    "device": "auto",
    "active_class": 0,
    "recursive": False,
    "last_model": "",
    "last_folder": "",
    "recent_models": [],
    "export": {
        "min_conf": 0.25,
        "max_conf": 1.0,
        "use_size": False,
        "min_width": 0,
        "max_width": 100000,
        "min_height": 0,
        "max_height": 100000,
        "skip_empty": True,
        "selected_classes": None,
        "split_enabled": False,
        "val_ratio": 0.2,
        "seed": 42,
        "conf_split": False,
        "conf_low": 0.3,
        "conf_high": 0.7,
        "delete_after": True,
    },
}
