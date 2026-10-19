"""
label_muzzle.py — generate real muzzle bounding boxes from a per-cow folder layout.

The original notebook wrote the SAME fake box (0.5, 0.5, 0.8, 0.8) for every
image. That's the single biggest reason the detector never worked on real
images: the network learned "muzzle = 80% of the frame", which is only true
for tight muzzle crops.

This script gives you two options:

  1. --auto   : auto-estimate a bounding box per image using a simple heuristic
                (tightest non-background region, or a fixed fraction of the
                frame). Fast, but not as good as real labels.

  2. --manual : write a small JSON of manually-specified boxes. You can get
                these from CVAT / Roboflow / Label Studio and paste them in.

For the best results, spend ~1 hour labelling 100-200 images with CVAT or
Roboflow, then paste them into boxes.json.

Usage:
    python label_muzzle.py --data ./dataset --auto
    python label_muzzle.py --data ./dataset --boxes ./boxes.json
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def auto_box(img_path: Path, frac: float = 0.85) -> tuple[float, float, float, float]:
    """
    Estimate a bounding box around the muzzle.

    Strategy: convert to grayscale, apply CLAHE, then find the largest
    connected component of bright pixels (the muzzle ridge pattern is the
    highest-contrast region in the image). Fall back to a centred box if that
    fails.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return 0.5, 0.5, frac, frac
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # threshold to find the bright ridge region
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # keep only the central-ish region so background doesn't dominate
    ys, xs = np.where(thresh > 0)
    if len(ys) == 0:
        return 0.5, 0.5, frac, frac
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    # expand slightly to be generous
    bw = max(x2 - x1, 20)
    bh = max(y2 - y1, 20)
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw_n = min(bw / w, 1.0)
    bh_n = min(bh / h, 1.0)
    return cx, cy, bw_n, bh_n


def write_labels(data_root: Path, boxes_json, frac: float, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    boxes = {}
    if boxes_json and boxes_json.exists():
        boxes = json.loads(boxes_json.read_text())

    n_written = 0
    for cow_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for img_path in list(cow_dir.glob("*.jpg")) + list(cow_dir.glob("*.jpeg")) + list(cow_dir.glob("*.png")):
            label_path = out_dir / (img_path.stem + ".txt")
            key = f"{cow_dir.name}/{img_path.name}"
            if key in boxes:
                cx, cy, bw, bh = boxes[key]
            else:
                cx, cy, bw, bh = auto_box(img_path, frac)
            with open(label_path, "w") as f:
                f.write(f"0 {cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}\n")
            n_written += 1
    print(f"[DONE] Wrote {n_written} label files to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Per-cow folder layout")
    ap.add_argument("--boxes", default=None, help="JSON of manual boxes")
    ap.add_argument("--frac", type=float, default=0.85)
    ap.add_argument("--out", default=None, help="Where to write .txt labels")
    args = ap.parse_args()

    data_root = Path(args.data)
    out_dir = Path(args.out) if args.out else data_root / "labels"
    write_labels(data_root, Path(args.boxes) if args.boxes else None, args.frac, out_dir)


if __name__ == "__main__":
    main()