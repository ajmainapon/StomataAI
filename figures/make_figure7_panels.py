"""Regenerate the Figure 7 qualitative-comparison panels natively.

Reproduces the published selection logic, class colours and score thresholds,
but renders each panel at native crop resolution so the montage is crisp.
Emits ten PNGs plus a metadata JSON recording what was selected and detected.
"""
from __future__ import annotations

import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from ultralytics import YOLO

warnings.filterwarnings("ignore")

ROOT = Path("/Volumes/Blankspace/Stomata Paper/stomataAI")
OUTDIR = Path("/Volumes/Blankspace/Stomata Paper/figassets3")
OUTDIR.mkdir(parents=True, exist_ok=True)

GT_PATH = ROOT / "runs/maskrcnn/evaluation/ground_truth.json"
MRCNN_PRED_PATH = ROOT / "runs/maskrcnn/evaluation/predictions.json"

YOLO_MODELS = {
    "yolov8": ROOT / "runs/yolov8s_combined/weights/best.pt",
    "yolo26": ROOT / "runs/yolo26s_combined/weights/best.pt",
}

CLASS_NAMES = {2: "guard", 3: "pore"}          # COCO category ids
YOLO_CLASS_NAMES = {1: "guard", 2: "pore"}     # ultralytics class indices
CLASS_COLORS = {                                # Okabe-Ito, colour-blind safe
    "guard": np.asarray([0, 114, 178], dtype=np.uint8),
    "pore": np.asarray([213, 94, 0], dtype=np.uint8),
}
YOLO_THRESHOLD = 0.25
MRCNN_THRESHOLD = 0.50
CROP_W, CROP_H = 900, 675
ALPHA = 0.34


# --------------------------------------------------------------- selection --
def select_images(dataset: dict) -> list[dict]:
    """One control and one stress image, chosen by ground-truth class richness."""
    counts = defaultdict(Counter)
    for ann in dataset["annotations"]:
        if ann["category_id"] in CLASS_NAMES:
            counts[ann["image_id"]][ann["category_id"]] += 1
    chosen = []
    for treatment in ("control", "stress"):
        candidates = []
        for image in dataset["images"]:
            if not image["file_name"].startswith(f"{treatment}_"):
                continue
            c = counts[image["id"]]
            score = (min(c[2], c[3]), c[2] + c[3], c[3], -image["id"])
            candidates.append((score, image))
        chosen.append(max(candidates, key=lambda it: it[0])[1])
    return chosen


def best_crop(info: dict, anns: list[dict]) -> tuple[int, int, int, int]:
    """Crop window holding the richest mix of *fully contained* objects.

    Scored on containment rather than centre-inclusion so that masks are not
    clipped at the panel edge, and scanned over a coarse grid so the window is
    not forced into a corner by a single peripheral object.
    """
    W, H = info["width"], info["height"]
    cw, ch = min(CROP_W, W), min(CROP_H, H)
    boxes = [
        (*ann["bbox"], ann["category_id"])
        for ann in anns
        if ann["category_id"] in CLASS_NAMES
    ]
    if not boxes:
        return 0, 0, cw, ch

    step = 50
    lefts = list(range(0, max(1, W - cw + 1), step)) or [0]
    tops = list(range(0, max(1, H - ch + 1), step)) or [0]
    if lefts[-1] != W - cw:
        lefts.append(W - cw)
    if tops[-1] != H - ch:
        tops.append(H - ch)

    best = None
    for left in lefts:
        for top in tops:
            right, bottom = left + cw, top + ch
            local = Counter()
            margin = 1e9
            for x, y, w, h, cat in boxes:
                if x >= left and y >= top and x + w <= right and y + h <= bottom:
                    local[cat] += 1
                    margin = min(
                        margin, x - left, y - top, right - (x + w), bottom - (y + h)
                    )
            # prefer both classes present, then more objects, then more pores,
            # then objects sitting comfortably inside the frame
            score = (
                min(local[2], local[3]),
                local[2] + local[3],
                local[3],
                margin if local else -1,
            )
            if best is None or score > best[0]:
                best = (score, (left, top, right, bottom))
    assert best is not None
    return best[1]


# ------------------------------------------------------------------ masking --
def polygon_mask(size: tuple[int, int], polygons: list) -> np.ndarray:
    canvas = Image.new("L", size, 0)
    draw = ImageDraw.Draw(canvas)
    for poly in polygons:
        if not poly or len(poly) < 6:
            continue
        pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
        draw.polygon(pts, fill=1)
    return np.asarray(canvas, dtype=bool)


def edge(mask: np.ndarray) -> np.ndarray:
    inner = mask.copy()
    inner[1:, :] &= mask[:-1, :]
    inner[:-1, :] &= mask[1:, :]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    return mask & ~inner


def thicken(mask: np.ndarray) -> np.ndarray:
    """Dilate a one-pixel outline so it survives print reduction."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def composite(image: Image.Image, masks, crop) -> Image.Image:
    rgb = np.asarray(image.convert("RGB")).copy()
    for cls, mask in masks:
        color = CLASS_COLORS[cls]
        rgb[mask] = np.round((1 - ALPHA) * rgb[mask] + ALPHA * color).astype(np.uint8)
        rgb[thicken(edge(mask))] = color
    left, top, right, bottom = crop
    return Image.fromarray(rgb[top:bottom, left:right])


def gt_masks(info, anns):
    out, size = [], (info["width"], info["height"])
    for ann in anns:
        cls = CLASS_NAMES.get(ann["category_id"])
        if cls is None:
            continue
        seg = ann.get("segmentation", [])
        mask = (
            mask_utils.decode(seg).astype(bool)
            if isinstance(seg, dict)
            else polygon_mask(size, seg)
        )
        out.append((cls, mask))
    return out


def yolo_masks(result, size):
    out = []
    if result.masks is None or result.boxes is None:
        return out
    classes = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()
    for cid, score, poly in zip(classes, scores, result.masks.xy):
        cls = YOLO_CLASS_NAMES.get(int(cid))
        if cls is None or float(score) < YOLO_THRESHOLD:
            continue
        pts = np.asarray(poly, dtype=float).reshape(-1, 2)
        if len(pts) < 3:
            continue
        out.append((cls, polygon_mask(size, [pts.ravel().tolist()])))
    return out


def mrcnn_masks(info, preds):
    out = []
    for p in preds:
        if p["image_id"] != info["id"] or p["score"] < MRCNN_THRESHOLD:
            continue
        cls = CLASS_NAMES.get(p["category_id"])
        if cls is None:
            continue
        out.append((cls, mask_utils.decode(p["segmentation"]).astype(bool)))
    return out


def tally(masks):
    c = Counter(cls for cls, _ in masks)
    return {"guard": c["guard"], "pore": c["pore"]}


# ---------------------------------------------------------------------- run --
def main() -> None:
    gt = json.loads(GT_PATH.read_text())
    mrcnn_preds = json.loads(MRCNN_PRED_PATH.read_text())

    ann_by_image = defaultdict(list)
    for ann in gt["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    models = {k: YOLO(str(v)) for k, v in YOLO_MODELS.items()}
    meta = {}

    for info in select_images(gt):
        treatment, original = info["file_name"].split("_", 1)
        row = "control" if treatment == "control" else "drought"
        path = ROOT / "raw" / treatment / "test" / original
        image = Image.open(path)
        anns = ann_by_image[info["id"]]
        crop = best_crop(info, anns)
        size = (info["width"], info["height"])

        panels = {"input": [], "gt": gt_masks(info, anns)}
        for key, model in models.items():
            result = model.predict(
                str(path), imgsz=640, conf=YOLO_THRESHOLD, verbose=False
            )[0]
            panels[key] = yolo_masks(result, size)
        panels["mrcnn"] = mrcnn_masks(info, mrcnn_preds)

        for key, masks in panels.items():
            out = composite(image, masks, crop)
            out.save(OUTDIR / f"f7_{row}_{key}.png", optimize=True)

        meta[row] = {
            "file_name": info["file_name"],
            "image_id": info["id"],
            "source_size": size,
            "crop": crop,
            "counts": {k: tally(v) for k, v in panels.items() if k != "input"},
        }
        print(f"{row:8s} {info['file_name']}")
        print(f"         crop {crop}  ->  {CROP_W}x{CROP_H}")
        for k in ("gt", "yolov8", "yolo26", "mrcnn"):
            t = tally(panels[k])
            print(f"         {k:7s} guard={t['guard']:3d}  pore={t['pore']:3d}")

    (OUTDIR / "f7_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote 10 panels + metadata to {OUTDIR}")


if __name__ == "__main__":
    main()
