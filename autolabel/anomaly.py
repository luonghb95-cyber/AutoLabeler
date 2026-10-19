"""Detect the most unusual boxes in a dataset (geometry + local context).

For every box we extract cheap, interpretable features:
  aspect, size, x, y   — geometry, normalised per class over the whole dataset
  color                — colour distance between the box crop and its surrounding ring
  texture              — edge-density ratio between the box crop and its surrounding ring

Robust z-scores (median/MAD) per class are combined into an anomaly score.
False detections usually blend into their background or have odd geometry no matter
how confident the model was, so the score deliberately ignores `conf`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .image_io import load_bgr
from .models import ImageEntry

FEATURE_NAMES = ["shape", "size", "pos_x", "pos_y", "color", "texture"]
FEATURE_WEIGHTS = np.array([0.9, 0.9, 0.5, 0.5, 1.5, 1.3], dtype=np.float64)

REASON_TEXTS = {
    "shape": ({"low": "odd shape", "high": "odd shape"}),
    "size": ({"low": "odd size (too small)", "high": "odd size (too big)"}),
    "pos_x": ({"low": "unusual position", "high": "unusual position"}),
    "pos_y": ({"low": "unusual position", "high": "unusual position"}),
    "color": ({"low": "blends into the background", "high": "colours unlike neighbours"}),
    "texture": ({"low": "flat texture vs neighbours", "high": "texture unlike neighbours"}),
}

MIN_CLASS_SAMPLES = 8
REASON_Z = 1.6


@dataclass
class AnomalyHit:
    entry_index: int
    box_index: int
    label: str
    conf: float | None
    score: float
    z: dict
    reasons: list = field(default_factory=list)


def _clip_int(value: float, low: int, high: int) -> int:
    return int(min(max(value, low), high))


def extract_features(image: np.ndarray, lab: np.ndarray, gradient: np.ndarray, box, width: int, height: int) -> list[float] | None:
    bw = max(1.0, box.width)
    bh = max(1.0, box.height)
    aspect = math.log(bw / bh)
    size = math.sqrt((bw * bh) / max(1.0, float(width) * float(height)))
    cx = ((box.x1 + box.x2) / 2.0) / max(1, width)
    cy = ((box.y1 + box.y2) / 2.0) / max(1, height)

    ix1 = _clip_int(math.floor(box.x1), 0, width - 1)
    iy1 = _clip_int(math.floor(box.y1), 0, height - 1)
    ix2 = _clip_int(math.ceil(box.x2), ix1 + 1, width)
    iy2 = _clip_int(math.ceil(box.y2), iy1 + 1, height)

    pad_x = max(2, int(0.5 * (ix2 - ix1)))
    pad_y = max(2, int(0.5 * (iy2 - iy1)))
    ex1 = max(0, ix1 - pad_x)
    ey1 = max(0, iy1 - pad_y)
    ex2 = min(width, ix2 + pad_x)
    ey2 = min(height, iy2 + pad_y)

    inner_lab = lab[iy1:iy2, ix1:ix2].reshape(-1, 3)
    outer_lab = lab[ey1:ey2, ex1:ex2].reshape(-1, 3)
    inner_pixels = inner_lab.shape[0]
    outer_pixels = outer_lab.shape[0]
    raw_ring = outer_pixels - inner_pixels
    if raw_ring <= 0:
        color = 0.0
        texture = 0.0
    else:
        ring_pixels = float(raw_ring)
        inner_mean = inner_lab.mean(axis=0)
        ring_mean = (outer_lab.sum(axis=0) - inner_lab.sum(axis=0)) / ring_pixels
        color = float(np.linalg.norm(inner_mean - ring_mean)) / 100.0

        inner_grad = gradient[iy1:iy2, ix1:ix2]
        outer_grad = gradient[ey1:ey2, ex1:ex2]
        inner_edge = float(inner_grad.mean())
        ring_edge = (float(outer_grad.sum()) - float(inner_grad.sum())) / ring_pixels
        texture = math.log((inner_edge + 4.0) / (max(0.0, ring_edge) + 4.0))

    return [aspect, size, cx, cy, color, texture]


def _robust_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0) * 1.4826
    std = values.std(axis=0)
    scale = np.where(mad > 1e-6, mad, np.where(std > 1e-6, std, 1.0))
    return median, scale


def scan_anomalies(entries: list[ImageEntry], include_manual: bool = True, progress=None, is_cancelled=None) -> tuple[list[AnomalyHit], dict]:
    rows: list[tuple[int, int, object, list[float]]] = []
    total = len(entries)
    for index, entry in enumerate(entries):
        if is_cancelled is not None and is_cancelled():
            break
        if progress is not None:
            progress(index, total)
        boxes = [(position, box) for position, box in enumerate(entry.boxes) if include_manual or box.conf is not None]
        if not boxes:
            continue
        image = load_bgr(entry.path)
        if image is None:
            continue
        height, width = image.shape[:2]
        width = entry.width or width
        height = entry.height or height
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(gx, gy)
        for position, box in boxes:
            features = extract_features(image, lab, gradient, box, width, height)
            if features is not None:
                rows.append((index, position, box, features))
    if progress is not None:
        progress(total, total)

    if not rows:
        return [], {"boxes": 0, "images": 0}

    matrix = np.array([row[3] for row in rows], dtype=np.float64)
    labels = [row[2].label for row in rows]

    global_median, global_scale = _robust_stats(matrix)
    per_class: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label in set(labels):
        mask = np.array([name == label for name in labels])
        if int(mask.sum()) >= MIN_CLASS_SAMPLES:
            per_class[label] = _robust_stats(matrix[mask])

    weight_sq = FEATURE_WEIGHTS ** 2
    weight_norm = float(weight_sq.sum())

    hits: list[AnomalyHit] = []
    for row, features in zip(rows, matrix):
        entry_index, box_index, box, _ = row
        median, scale = per_class.get(box.label, (global_median, global_scale))
        z = (features - median) / scale
        z = np.clip(z, -8.0, 8.0)
        score = math.sqrt(float(np.sum(weight_sq * z * z)) / weight_norm)
        reasons = []
        for position, name in enumerate(FEATURE_NAMES):
            if abs(z[position]) >= REASON_Z:
                side = "low" if z[position] < 0 else "high"
                text = REASON_TEXTS[name][side]
                if text not in reasons:
                    reasons.append(text)
        hits.append(
            AnomalyHit(
                entry_index=entry_index,
                box_index=box_index,
                label=box.label,
                conf=box.conf,
                score=score,
                z={name: float(z[position]) for position, name in enumerate(FEATURE_NAMES)},
                reasons=reasons[:3],
            )
        )

    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits, {"boxes": len(hits), "images": total}
