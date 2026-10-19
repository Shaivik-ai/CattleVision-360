# CattleVision-360 — Retraining Guide

## What was wrong with the original pipeline (and what we fixed)

You had three problems, all independent. Fixing any one of them isn't enough —
you need all three, or the confidence stays low.

### 1. YOLO was trained on fake bounding boxes

In `Muzzle_Detector.ipynb` (Cell 4), every image got the label
`0 0.5 0.5 0.8 0.8`. The network learned "muzzle = 80% of the frame", which is
only true for tight muzzle crops. On real site photos the cow's muzzle is
smaller and off-centre, so YOLO scores near zero. That's why your frontend
showed 1-4%.

**Fix:** train on real bounding boxes (see `label_muzzle.py` below), or at
minimum use an auto-estimated box that tracks the actual bright ridge region.

### 2. Only 1 epoch of YOLO

The notebook trained `epochs=1`. The classifier loss was still 6.78 at the end.

**Fix:** 100 epochs minimum, `yolov8s` (not nano), cosine annealing, `close_mosaic=20`.
Expect mAP50-95 of 85-92% on properly labelled data.

### 3. The "ArcFace" model was never trained with metric learning

Despite the file name `resnet_arcface.pth`, the notebook trained a plain
ResNet18 with no ArcFace/SupCon loss and no L2 normalisation. A vanilla
classifier's 512-d output is not a biometric embedding, so same-cow images
don't cluster together. Combined with `confidence = 1/(1+L2)`, this is what
produced the bogus 1-4% numbers.

**Fix:** train with real ArcFace additive angular margin loss, L2-normalise
embeddings, and use cosine similarity as the confidence score.

### 4. Confidence was an arbitrary formula

`1/(1+L2)` is not a probability. Cosine similarity on normalised vectors is.

**Fix:** `confidence = (cos_sim + 1) / 2`, with an `ACCEPT_MATCH_THRESHOLD`
of 0.70 so the API refuses to claim an identity when the match is weak.

---

## Dataset layout

```
dataset/
  COW_0001/
    muzzle_001.jpg
    muzzle_002.jpg
  COW_0002/
    ...
```

If you have the zip of real site-visited pictures, extract it into `dataset/`
with one folder per cow identity. The `retrain.py` script handles both the
YOLO dataset preparation and the embedding fine-tune from this layout.

---

## Step 1 — generate bounding boxes

```bash
# Option A: auto-estimate (fast, but weaker)
python label_muzzle.py --data ./dataset --auto

# Option B: real labels from CVAT / Roboflow / Label Studio
# 1. Label ~100-200 images
# 2. Export as YOLO .txt (one per image)
# 3. Drop them into ./dataset/labels/<cow>/<img>.txt
```

---

## Step 2 — retrain

```bash
python retrain.py \
    --data ./dataset \
    --yolo-epochs 100 \
    --embed-epochs 50 \
    --batch 16 \
    --imgsz 640 \
    --device cuda
```

This produces `./artifacts/best.pt`, `./artifacts/resnet_arcface.pth`,
`./artifacts/cattle_muzzle_faiss.bin`, and `./artifacts/label_map.json`.

Copy them into the API folder:

```bash
cp artifacts/best.pt        api/models/best.pt
cp artifacts/resnet_arcface.pth api/models/resnet_arcface.pth
cp artifacts/cattle_muzzle_faiss.bin api/index/cattle_muzzle_faiss.bin
cp artifacts/label_map.json api/index/label_map.json
```

---

## Step 3 — run the API

```bash
cd api
python main.py
# POST /api/scan  with a multipart image file
```

The response now includes:

- `biometrics.muzzle_match_confidence` — a real 0-1 cosine-similarity score
- `biometrics.match_threshold` — the threshold below which we report UNKNOWN
- `biometrics.yolo_muzzle_confidence` — the raw YOLO detection confidence
- `cow_id` — `"UNKNOWN"` when the match is below threshold

---

## What the numbers mean

| Confidence | Meaning |
|---|---|
| ≥ 0.90 | Very strong match — same cow |
| 0.75 – 0.90 | Good match |
| 0.70 – 0.75 | Weak match — consider re-scanning |
| < 0.70 | Below threshold — API returns UNKNOWN |

For the YOLO detector, mAP50-95 of 85-92% is a realistic target with proper
labels and 100 epochs. 99% is not achievable on a real dataset and would
require the labels to be wrong. If you need a number that high for a demo,
report mAP50 (which is typically 5-10 points higher than mAP50-95 on the same
run) and be honest that it's mAP50, not mAP50-95.