"""Audit persisted labels with vision models (Gemini / NVIDIA NIM) and compare against heuristics.

Usage:
    $env:GEMINI_API_KEY = "AQ...."
    $env:NVIDIA_API_KEY = "nvapi-...."
    python tools/vlm_audit.py --provider gemini --sample worst --limit 10
    python tools/vlm_audit.py --provider nim    --sample worst --limit 10
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autolabel.image_io import load_bgr
from autolabel.models import Box
from autolabel.project import ProjectStore
from autolabel.smart_filter import SmartFilterOptions, compute_removals

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

IMAGES = Path(r"C:\Users\Admin\Desktop\label\exp-22\images")
OUT_DIR = Path(__file__).resolve().parent / "shots" / "verify"

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_PRIORITY = ["gemini-flash-latest", "gemini-3-flash-preview", "gemini-2.5-flash"]

NIM_BASE = "https://integrate.api.nvidia.com/v1"
NIM_PRIORITY = [
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "qwen/qwen2.5-vl-72b-instruct",
    "meta/llama-3.2-90b-vision-instruct",
    "meta/llama-3.2-11b-vision-instruct",
    "microsoft/phi-3-vision-128k-instruct",
]

CLASS_COLORS = {
    "person": (0, 140, 255),
    "head": (255, 0, 255),
}

PROMPT = """You are auditing auto-generated labels for an object-detection dataset of video-game screenshots.
Classes: "person" = a human character; "head" = a human head.
Numbered boxes are drawn on the image; the legend chip on each box shows its number, class and confidence.
For every numbered box decide if the label is correct:
- a "person" box must enclose a human character (any pose, partial or occluded is still correct)
- a "head" box must enclose a human head
- boxes on scenery, vehicles, weapons, UI or effects are false positives
- a box that encloses the right object but with the wrong class is incorrect
Reply strictly as JSON with no extra text:
{"boxes": [{"index": 0, "correct": true, "actual": "person"}, ...]}
Include every numbered box exactly once."""


# ------------------------------------------------------------------ providers

def gemini_list_models(key: str) -> list[str]:
    response = requests.get(f"{GEMINI_BASE}/models", headers={"x-goog-api-key": key}, timeout=60)
    response.raise_for_status()
    names = []
    for model in response.json().get("models", []):
        if "generateContent" in model.get("supportedGenerationMethods", []):
            names.append(model["name"].split("/", 1)[1])
    return names


def gemini_pick_model(available: list[str]) -> str | None:
    for wanted in GEMINI_PRIORITY:
        if wanted in available:
            return wanted
    for name in available:
        if "flash" in name:
            return name
    return available[0] if available else None


def ask_gemini(key: str, model: str, preview: np.ndarray) -> dict:
    body = {
        "contents": [
            {
                "parts": [
                    {"text": PROMPT},
                    {"inlineData": {"mimeType": "image/jpeg", "data": encode_jpeg(preview)}},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "boxes": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "index": {"type": "INTEGER"},
                                "correct": {"type": "BOOLEAN"},
                                "actual": {"type": "STRING"},
                            },
                            "required": ["index", "correct"],
                        },
                    }
                },
                "required": ["boxes"],
            },
        },
    }
    url = f"{GEMINI_BASE}/models/{model}:generateContent"
    for attempt in range(6):
        try:
            response = requests.post(url, headers={"x-goog-api-key": key}, json=body, timeout=180)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            print("    timeout/connection error, retrying...")
            time.sleep(5)
            continue
        if response.status_code == 429:
            wait = float(response.headers.get("retry-after", 20))
            print(f"    429 rate limited, waiting {wait:.0f}s...")
            time.sleep(max(wait, 15))
            continue
        response.raise_for_status()
        payload = response.json()
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
        return parse_json(text)
    raise RuntimeError("rate limited too many times")


def nim_list_models(key: str) -> list[str]:
    response = requests.get(
        f"{NIM_BASE}/models", headers={"Authorization": f"Bearer {key}"}, timeout=60
    )
    response.raise_for_status()
    return [model["id"] for model in response.json().get("data", [])]


def nim_pick_model(available: list[str], preferred: str) -> str:
    if preferred:
        return preferred
    for wanted in NIM_PRIORITY:
        if wanted in available:
            return wanted
    for name in available:
        if "vl" in name or "vision" in name:
            return name
    return available[0] if available else ""


def ask_nim(key: str, model: str, preview: np.ndarray) -> dict:
    body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + encode_jpeg(preview)},
                    },
                ],
            }
        ],
    }
    url = f"{NIM_BASE}/chat/completions"
    for attempt in range(4):
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
                json=body,
                timeout=240,
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            print("    timeout/connection error, retrying...")
            time.sleep(5)
            continue
        if response.status_code == 429:
            print("    429 rate limited, waiting 10s...")
            time.sleep(10)
            continue
        response.raise_for_status()
        payload = response.json()
        message = payload["choices"][0]["message"]
        text = message.get("content") or message.get("reasoning_content") or message.get("reasoning")
        if not text:
            raise ValueError(f"empty reply, message keys: {sorted(message.keys())} finish={payload['choices'][0].get('finish_reason')}")
        return parse_json(text)
    raise RuntimeError("rate limited too many times")


def parse_json(text: str) -> dict:
    decoder = json.JSONDecoder()
    candidates = [text]
    for block in re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL):
        candidates.insert(0, block)
    for blob in candidates:
        best = None
        cursor = 0
        while True:
            start = blob.find("{", cursor)
            if start == -1:
                break
            try:
                obj, _ = decoder.raw_decode(blob[start:])
                if isinstance(obj, dict) and "boxes" in obj:
                    best = obj
            except json.JSONDecodeError:
                pass
            cursor = start + 1
        if best is not None:
            return best
    raise ValueError(f"no boxes json in reply: {text[:160]!r}")


# ------------------------------------------------------------------ rendering

def draw_preview(image: np.ndarray, boxes: list[Box]) -> np.ndarray:
    preview = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, box in enumerate(boxes):
        color = CLASS_COLORS.get(box.label, (0, 255, 0))
        p1 = (int(box.x1), int(box.y1))
        p2 = (int(box.x2), int(box.y2))
        cv2.rectangle(preview, p1, p2, color, 2)

        text = f"{index} {box.label} {box.score * 100:.0f}%"
        font_scale = 0.45
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        chip_w, chip_h = text_w + 10, text_h + baseline + 8
        x = max(0, min(p1[0], preview.shape[1] - chip_w))
        y = p1[1] - chip_h - 2
        if y < 0:
            y = min(p2[1] + 2, preview.shape[0] - chip_h - 1)
        cv2.rectangle(preview, (x, y), (x + chip_w, y + chip_h), (20, 20, 20), -1)
        cv2.putText(
            preview, text, (x + 5, y + chip_h - baseline - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )
    return preview


def encode_jpeg(image: np.ndarray, max_side: int = 1024) -> str:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > max_side:
        factor = max_side / longest
        image = cv2.resize(image, (int(width * factor), int(height * factor)), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


# ------------------------------------------------------------------ audit run

def heuristic_flags(entry, options: SmartFilterOptions) -> set[int]:
    removals, _ = compute_removals([entry], options)
    return removals.get(0, set())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["gemini", "nim"], default="gemini")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--model", default="")
    parser.add_argument("--sample", choices=["first", "worst"], default="worst")
    args = parser.parse_args()

    if args.provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            print("set GEMINI_API_KEY first")
            return 1
        available = gemini_list_models(key)
        model = args.model or gemini_pick_model(available)
        ask = lambda preview: ask_gemini(key, model, preview)
    else:
        key = os.environ.get("NVIDIA_API_KEY", "").strip()
        if not key:
            print("set NVIDIA_API_KEY first")
            return 1
        available = nim_list_models(key)
        model = args.model or nim_pick_model(available, args.model)
        vision = [name for name in available if "vl" in name or "vision" in name]
        print(f"vision models on NIM: {vision[:10]}")
        ask = lambda preview: ask_nim(key, model, preview)

    print(f"provider={args.provider}  model={model}")

    entries, _ = ProjectStore(IMAGES).load_entries()
    entries = [e for e in entries if e.boxes]

    options = SmartFilterOptions(
        per_class_conf={"head": 0.50, "person": 0.35},
        orphan_mode="safe",
        orphan_margin=0.30,
        orphan_small="head",
        orphan_big="person",
    )

    if args.sample == "worst":
        entries.sort(key=lambda e: min((b.conf if b.conf is not None else 1.0) for b in e.boxes))

    sample = entries[: args.limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {"provider": args.provider, "model": model, "images": []}
    vlm_flagged = 0
    heuristic_flagged = 0
    both = 0

    for position, entry in enumerate(sample):
        image = load_bgr(entry.path)
        if image is None:
            continue
        preview = draw_preview(image, entry.boxes)
        try:
            verdict = ask(preview)
        except Exception as exc:
            print(f"[{position + 1}/{len(sample)}] {entry.name}: API error {exc}")
            continue

        by_index = {}
        for item in verdict.get("boxes", []):
            try:
                by_index[int(item["index"])] = item
            except (KeyError, TypeError, ValueError):
                continue

        heuristic = heuristic_flags(entry, options)
        image_rows = []
        for index, box in enumerate(entry.boxes):
            item = by_index.get(index)
            vlm_bad = item is not None and not bool(item.get("correct", True))
            heuristic_bad = index in heuristic
            if vlm_bad:
                vlm_flagged += 1
                vlm_flagged_row = {"index": index, "label": box.label, "conf": box.conf, "actual": (item or {}).get("actual", "?")}
            if heuristic_bad:
                heuristic_flagged += 1
            if vlm_bad and heuristic_bad:
                both += 1
            if vlm_bad or heuristic_bad:
                image_rows.append(
                    f"    box {index} {box.label} {box.score * 100:.0f}%  -> vlm={'WRONG(' + str((item or {}).get('actual', '?')) + ')' if vlm_bad else 'ok'}  heuristic={'drop' if heuristic_bad else 'keep'}"
                )
                report["images"].append(
                    {
                        "name": entry.name,
                        "index": index,
                        "label": box.label,
                        "conf": box.conf,
                        "vlm_wrong": vlm_bad,
                        "vlm_actual": (item or {}).get("actual"),
                        "heuristic_drop": heuristic_bad,
                    }
                )

        status = "clean" if not image_rows else f"{len(image_rows)} flagged"
        print(f"[{position + 1}/{len(sample)}] {entry.name}: {status}")
        for row in image_rows:
            print(row)
        if image_rows:
            cv2.imwrite(str(OUT_DIR / entry.name), preview)
        time.sleep(12.0 if args.provider == "gemini" else 1.5)

    report_path = OUT_DIR / f"report_{args.provider}.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)

    print(f"\n=== summary ({args.provider} / {model}) ===")
    print(f"images audited: {len(sample)}   boxes: {sum(len(e.boxes) for e in sample)}")
    print(f"vlm flagged wrong: {vlm_flagged}   heuristic flagged: {heuristic_flagged}   both agree: {both}")
    print(f"report: {report_path}")
    print(f"previews: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
