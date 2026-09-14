"""Figure 7 for v2: qualitative instance-segmentation comparison.

Rebuilds the full montage (5 columns x 2 treatment rows) from the v2 weights
and the v2 locked test set. Mask R-CNN uses its validation-selected checkpoint.
"""
from __future__ import annotations

import json
import os
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")
os.environ.setdefault("YOLO_VERBOSE", "False")

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pycocotools import mask as mask_utils  # noqa: E402
from ultralytics import YOLO  # noqa: E402

V2 = "/home/rmedu2026/stomataAI/v2"
OUT = f"{V2}/figures"
MRCNN = f"{V2}/runs/maskrcnn_orig800_s42"
YOLOW = {
    "YOLOv8s-seg": f"{V2}/runs/yolov8s_640_s42/weights/best.pt",
    "YOLO26s-seg": f"{V2}/runs/yolo26s_640_s42/weights/best.pt",
}

CLASS_NAMES = {2: "guard", 3: "pore"}        # COCO ids in the eval ground truth
YOLO_CLS = {1: "guard", 2: "pore"}
COLORS = {"guard": np.array([0, 114, 178], np.uint8),
          "pore": np.array([213, 94, 0], np.uint8)}
YOLO_T, MRCNN_T, ALPHA = 0.25, 0.50, 0.34
CROP_W, CROP_H = 900, 675


def poly_mask(size, polys):
    c = Image.new("L", size, 0)
    d = ImageDraw.Draw(c)
    for p in polys:
        if not p or len(p) < 6:
            continue
        d.polygon([(p[i], p[i + 1]) for i in range(0, len(p) - 1, 2)], fill=1)
    return np.asarray(c, bool)


def edge(m):
    i = m.copy()
    i[1:, :] &= m[:-1, :]; i[:-1, :] &= m[1:, :]
    i[:, 1:] &= m[:, :-1]; i[:, :-1] &= m[:, 1:]
    return m & ~i


def thicken(m):
    o = m.copy()
    o[1:, :] |= m[:-1, :]; o[:-1, :] |= m[1:, :]
    o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]
    return o


def composite(img, masks, crop):
    rgb = np.asarray(img.convert("RGB")).copy()
    for cls, m in masks:
        c = COLORS[cls]
        rgb[m] = np.round((1 - ALPHA) * rgb[m] + ALPHA * c).astype(np.uint8)
        rgb[thicken(edge(m))] = c
    l, t, r, b = crop
    return rgb[t:b, l:r]


def best_crop(info, anns):
    W, H = info["width"], info["height"]
    cw, ch = min(CROP_W, W), min(CROP_H, H)
    boxes = [(*a["bbox"], a["category_id"]) for a in anns
             if a["category_id"] in CLASS_NAMES]
    if not boxes:
        return 0, 0, cw, ch
    best = None
    for left in list(range(0, max(1, W - cw + 1), 50)) + [W - cw]:
        for top in list(range(0, max(1, H - ch + 1), 50)) + [H - ch]:
            r, b = left + cw, top + ch
            loc, margin = Counter(), 1e9
            for x, y, w, h, cat in boxes:
                if x >= left and y >= top and x + w <= r and y + h <= b:
                    loc[cat] += 1
                    margin = min(margin, x - left, y - top, r - (x + w), b - (y + h))
            s = (min(loc[2], loc[3]), loc[2] + loc[3], loc[3], margin if loc else -1)
            if best is None or s > best[0]:
                best = (s, (left, top, r, b))
    return best[1]


def main():
    sel = json.load(open(f"{MRCNN}/selection/selection_valid.json"))
    ckpt = sel["best_checkpoint"]
    gt = json.load(open(f"{MRCNN}/selection/ground_truth_test.json"))
    mpred = json.load(open(f"{MRCNN}/selection/pred_test_{ckpt}.json"))
    print(f"Mask R-CNN checkpoint: {ckpt}")

    ann_by = defaultdict(list)
    for a in gt["annotations"]:
        ann_by[a["image_id"]].append(a)

    # pick the richest image per treatment
    chosen = {}
    for treat, pref in (("control", "control_"), ("drought", "stress_")):
        cand = []
        for im in gt["images"]:
            if not im["file_name"].startswith(pref):
                continue
            c = Counter(a["category_id"] for a in ann_by[im["id"]])
            cand.append(((min(c[2], c[3]), c[2] + c[3], c[3], -im["id"]), im))
        chosen[treat] = max(cand, key=lambda x: x[0])[1]

    models = {k: YOLO(v) for k, v in YOLOW.items()}
    panels = {}
    for treat, info in chosen.items():
        tdir = "control" if treat == "control" else "stress"
        fname = info["file_name"].split("_", 1)[1]
        path = f"{V2}/data/raw/{tdir}/test/{fname}"
        img = Image.open(path)
        size = (info["width"], info["height"])
        anns = ann_by[info["id"]]
        crop = best_crop(info, anns)
        print(f"{treat}: {info['file_name']}  crop={crop}")

        row = {"Input": []}
        gm = []
        for a in anns:
            cls = CLASS_NAMES.get(a["category_id"])
            if cls is None:
                continue
            seg = a.get("segmentation", [])
            m = (mask_utils.decode(seg).astype(bool) if isinstance(seg, dict)
                 else poly_mask(size, seg))
            gm.append((cls, m))
        row["Ground truth"] = gm

        for name, model in models.items():
            r = model.predict(path, imgsz=640, conf=YOLO_T, verbose=False)[0]
            ms = []
            if r.masks is not None:
                for cid, sc, poly in zip(r.boxes.cls.cpu().numpy().astype(int),
                                         r.boxes.conf.cpu().numpy(), r.masks.xy):
                    cls = YOLO_CLS.get(int(cid))
                    if cls is None or float(sc) < YOLO_T:
                        continue
                    pts = np.asarray(poly, float).reshape(-1, 2)
                    if len(pts) >= 3:
                        ms.append((cls, poly_mask(size, [pts.ravel().tolist()])))
            row[name] = ms

        mm = []
        for p in mpred:
            if p["image_id"] != info["id"] or p["score"] < MRCNN_T:
                continue
            cls = CLASS_NAMES.get(p["category_id"])
            if cls:
                mm.append((cls, mask_utils.decode(p["segmentation"]).astype(bool)))
        row["Mask R-CNN"] = mm

        panels[treat] = {k: composite(img, v, crop) for k, v in row.items()}
        for k, v in row.items():
            if k != "Input":
                c = Counter(x for x, _ in v)
                print(f"    {k:14s} guard={c['guard']:3d} pore={c['pore']:3d}")

    cols = ["Input", "Ground truth", "YOLOv8s-seg", "YOLO26s-seg", "Mask R-CNN"]
    rows = [("control", "Control"), ("drought", "Drought\nstress")]
    fig, axes = plt.subplots(2, 5, figsize=(17.5, 6.0))
    for i, (key, rlabel) in enumerate(rows):
        for j, c in enumerate(cols):
            ax = axes[i, j]
            ax.imshow(panels[key][c])
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_edgecolor("0.75"); s.set_linewidth(0.8)
            if i == 0:
                ax.set_title(c, fontsize=14, fontweight="bold", pad=9)
            if j == 0:
                ax.set_ylabel(rlabel, fontsize=13, fontweight="bold", labelpad=10)
    fig.subplots_adjust(wspace=0.03, hspace=0.04, bottom=0.10)
    fig.legend(handles=[mpatches.Patch(color="#0072B2", label="Guard-cell mask"),
                        mpatches.Patch(color="#D55E00", label="Pore mask")],
               loc="lower center", ncol=2, frameon=False, fontsize=13,
               bbox_to_anchor=(0.5, 0.005))
    fig.savefig(f"{OUT}/fig7_qualitative.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{OUT}/fig7_qualitative.pdf", bbox_inches="tight")
    print(f"wrote {OUT}/fig7_qualitative.png / .pdf")


if __name__ == "__main__":
    main()
