from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    cls_id: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


class DetectorError(Exception):
    pass


def _letterbox(image: np.ndarray, new_shape: tuple[int, int], color=(114, 114, 114)):
    height, width = image.shape[:2]
    new_h, new_w = new_shape
    ratio = min(new_w / width, new_h / height)
    resized_w, resized_h = int(round(width * ratio)), int(round(height * ratio))
    if (resized_w, resized_h) != (width, height):
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (new_w - resized_w) / 2
    pad_h = (new_h - resized_h) / 2
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, ratio, left, top


def _nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thres: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = scores.argsort()[::-1]
    max_coord = float(boxes.max()) + 4096.0
    offsets = classes.astype(np.float64) * max_coord
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        shift = np.abs(offsets[i] - offsets[rest]) < max_coord * 0.5
        order = rest[(iou <= iou_thres) | ~shift]
    return keep


class BaseEngine:
    backend = "base"

    def predict(self, image_bgr: np.ndarray, conf: float, iou: float, imgsz: int, max_det: int = 300) -> list[Detection]:
        raise NotImplementedError

    def describe(self) -> str:
        return self.backend


class PtEngine(BaseEngine):
    backend = "PyTorch"

    def __init__(self, model_path: Path, device: str):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError(
                "The 'ultralytics' package is required for .pt models.\n"
                "Install it with:  pip install ultralytics"
            ) from exc

        self.model = YOLO(str(model_path))
        self.device = self._resolve_device(device)
        names = self.model.names
        self.names = list(names.values()) if isinstance(names, dict) else list(names)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cpu":
            return "cpu"
        try:
            import torch

            if device in ("auto", "cuda", "gpu", "0"):
                return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            pass
        return "cpu"

    def describe(self) -> str:
        label = "GPU" if self.device != "cpu" else "CPU"
        return f"PyTorch · {label}"

    def predict(self, image_bgr: np.ndarray, conf: float, iou: float, imgsz: int, max_det: int = 300) -> list[Detection]:
        results = self.model.predict(
            source=image_bgr,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            max_det=max_det,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), score, cls_id in zip(xyxy, scores, classes):
                detections.append(Detection(int(cls_id), float(score), float(x1), float(y1), float(x2), float(y2)))
        return detections


class OnnxEngine(BaseEngine):
    backend = "ONNX Runtime"

    def __init__(self, model_path: Path, device: str):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise DetectorError(
                "The 'onnxruntime' package is required for .onnx models.\n"
                "Install it with:  pip install onnxruntime"
            ) from exc

        available = ort.get_available_providers()
        wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device in ("auto", "cuda", "gpu") else ["CPUExecutionProvider"]
        providers = [p for p in wanted if p in available] or ["CPUExecutionProvider"]

        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
        self.active_provider = self.session.get_providers()[0]

        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_height, self.input_width = self._parse_input_shape(input_meta.shape)
        self.names = self._read_names(model_path)

    @staticmethod
    def _parse_input_shape(shape: list) -> tuple[int, int]:
        height = width = 640
        try:
            if len(shape) == 4:
                if isinstance(shape[2], int) and shape[2] > 0:
                    height = shape[2]
                if isinstance(shape[3], int) and shape[3] > 0:
                    width = shape[3]
        except Exception:
            pass
        return max(32, int(height)), max(32, int(width))

    def _read_names(self, model_path: Path) -> list[str]:
        try:
            meta = self.session.get_modelmeta().custom_metadata_map
        except Exception:
            meta = {}
        raw = meta.get("names")
        if raw:
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, dict):
                    return [str(v) for _, v in sorted(parsed.items(), key=lambda kv: int(kv[0]))]
                if isinstance(parsed, (list, tuple)):
                    return [str(item) for item in parsed]
            except (ValueError, SyntaxError):
                pass

        sidecars = [
            model_path.with_suffix(".names.txt"),
            model_path.parent / "classes.txt",
            model_path.parent / "names.txt",
        ]
        for sidecar in sidecars:
            if sidecar.is_file():
                lines = [line.strip() for line in sidecar.read_text(encoding="utf-8").splitlines()]
                names = [line for line in lines if line]
                if names:
                    return names

        return [f"class_{i}" for i in range(self._guess_class_count())]

    def _guess_class_count(self) -> int:
        try:
            output = self.session.get_outputs()[0]
            shape = output.shape
            if len(shape) == 3:
                dim = shape[1] if isinstance(shape[1], int) else shape[2]
                if isinstance(dim, int) and dim > 4:
                    channels = dim
                    if channels % 2 == 1:
                        return max(1, channels - 5)
                    return max(1, channels - 4)
        except Exception:
            pass
        return 1

    def describe(self) -> str:
        label = "GPU" if "CUDA" in self.active_provider else "CPU"
        return f"ONNX · {label}"

    def predict(self, image_bgr: np.ndarray, conf: float, iou: float, imgsz: int, max_det: int = 300) -> list[Detection]:
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        padded, ratio, pad_x, pad_y = _letterbox(image, (self.input_height, self.input_width))
        blob = padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob)

        outputs = self.session.run(None, {self.input_name: blob})
        raw = outputs[0]
        if raw.ndim == 3:
            raw = raw[0]

        boxes, scores, class_ids = self._decode(raw, len(self.names))
        keep = _nms(boxes, scores, class_ids, iou)
        keep = [i for i in keep if scores[i] >= conf][:max_det]

        detections: list[Detection] = []
        for i in keep:
            x1 = (boxes[i, 0] - pad_x) / ratio
            y1 = (boxes[i, 1] - pad_y) / ratio
            x2 = (boxes[i, 2] - pad_x) / ratio
            y2 = (boxes[i, 3] - pad_y) / ratio
            detections.append(Detection(int(class_ids[i]), float(scores[i]), float(x1), float(y1), float(x2), float(y2)))
        return detections

    @staticmethod
    def _decode(raw: np.ndarray, num_names: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T

        cols = raw.shape[1]
        if OnnxEngine._is_end2end(raw, cols):
            boxes = raw[:, :4]
            scores = raw[:, 4]
            class_ids = np.round(raw[:, 5]).astype(int)
            return boxes, scores, class_ids

        if cols == 4 + num_names:
            cls_scores = raw[:, 4:]
        elif cols == 5 + num_names:
            cls_scores = raw[:, 5:] * raw[:, 4:5]
        else:
            v5_like = abs(cols - 5 - num_names)
            v8_like = abs(cols - 4 - num_names)
            if num_names > 0 and v5_like < v8_like:
                cls_scores = raw[:, 5:] * raw[:, 4:5]
            else:
                cls_scores = raw[:, 4:]

        class_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(len(cls_scores)), class_ids]
        cx, cy, w, h = raw[:, :4].T
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        return xyxy, scores, class_ids

    @staticmethod
    def _is_end2end(raw: np.ndarray, cols: int) -> bool:
        if cols != 6 or raw.shape[0] > 1000:
            return False
        if raw.shape[0] == 0:
            return False
        class_col = raw[:, 5]
        return bool(np.allclose(class_col, np.round(class_col), atol=1e-3))


def load_detector(model_path: Path, device: str = "auto") -> BaseEngine:
    if not model_path.is_file():
        raise DetectorError(f"Model file not found: {model_path}")
    suffix = model_path.suffix.lower()
    if suffix == ".pt":
        return PtEngine(model_path, device)
    if suffix == ".onnx":
        return OnnxEngine(model_path, device)
    raise DetectorError(f"Unsupported model format: {suffix} (expected .pt or .onnx)")
