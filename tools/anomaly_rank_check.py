"""Check where the VLM-audit suspicious boxes rank in the anomaly score."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autolabel.anomaly import scan_anomalies
from autolabel.image_io import list_images
from autolabel.models import ImageEntry
from autolabel.project import ProjectStore

FOLDER = Path(r"C:\Users\Admin\Desktop\label\exp-22\images")
AUDITED = [
    "20261016_121329_616_s000068.jpg",
    "20261016_121330_277_s000069.jpg",
    "20261016_121616_967_s000075.jpg",
    "20261016_121622_510_s000078.jpg",
    "20261016_121627_254_s000083.jpg",
    "20261016_121629_332_s000086.jpg",
    "20261016_121759_675_s000093.jpg",
    "20261016_121803_622_s000094.jpg",
    "20261016_122602_895_s000110.jpg",
    "20261016_122628_549_s000115.jpg",
]


def main() -> int:
    entries: list[ImageEntry] = []
    for path in list_images(FOLDER, False):
        entries.append(ImageEntry(path=path, rel=str(path.relative_to(FOLDER)).replace("\\", "/")))

    store = ProjectStore(FOLDER)
    if store.exists():
        saved, _ = store.load_entries()
        by_rel = {entry.rel: entry.boxes for entry in saved}
        for entry in entries:
            if entry.rel in by_rel:
                entry.boxes = by_rel[entry.rel]

    hits, stats = scan_anomalies(entries, include_manual=True)
    rank_of = {(hit.entry_index, hit.box_index): rank for rank, hit in enumerate(hits, 1)}

    print(f"scored {stats['boxes']} boxes")
    print("\naudited images (VLM suspicious boxes):")
    for name in AUDITED:
        for index, entry in enumerate(entries):
            if entry.name != name:
                continue
            print(f"  {name}")
            for position, box in enumerate(entry.boxes):
                key = (index, position)
                conf = "manual" if box.conf is None else f"{box.conf * 100:.0f}%"
                hit = hits[rank_of[key] - 1]
                print(f"    {box.label:7s} {conf:>7s}  rank {rank_of[key]:3d}/159  score {hit.score:5.2f}  {' | '.join(hit.reasons)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
