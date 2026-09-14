"""Figure 10 for v2: manual (ground-truth) versus AI stomatal counts.

The manual count for an image is its number of annotated guard-cell complexes;
the AI count is the number of guard-cell instances the model predicts. Both are
per image, so the pairing is exact. Restricted to the locked test set, which the
model never saw during training or checkpoint selection.
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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

V2 = "/home/rmedu2026/stomataAI/v2"
OUT = f"{V2}/figures"
MODEL = f"{V2}/runs/yolo26s_640_s42/weights/best.pt"
CONF = 0.25
GUARD_YOLO, GUARD_COCO = 1, 1        # class index in YOLO / category id in export

BLUE, INK = "#1f77b4", "#191c21"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})


IMAGES_PER_PLANT = {"control": 16, "stress": 10}   # capture order, confirmed by the authors


def plant_of(fname, treatment):
    """Images were captured plant by plant in numerical order."""
    stem = fname.split("_jpg.rf.")[0]
    n = int(stem)
    return (n - 1) // IMAGES_PER_PLANT[treatment] + 1


def manual_counts(treatment):
    """Guard-cell annotations per image = the human count."""
    p = f"{V2}/data/raw/{treatment}/test/_annotations.coco.json"
    d = json.load(open(p))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    per = defaultdict(int)
    for a in d["annotations"]:
        if cats[a["category_id"]] == "guard":
            per[a["image_id"]] += 1
    return [(os.path.basename(i["file_name"]), per[i["id"]]) for i in d["images"]]


def ai_counts(treatment, files):
    model = YOLO(MODEL)
    out = []
    for f in files:
        path = f"{V2}/data/raw/{treatment}/test/{f}"
        r = model.predict(path, imgsz=640, conf=CONF, verbose=False)[0]
        n = 0
        if r.boxes is not None:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            n = int((cls == GUARD_YOLO).sum())
        out.append(n)
    return out


def fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return slope, intercept, r2


def main():
    K = 7.1107          # count -> density on the source scale (anchored to 90.17)
    per_image, per_plant = {}, {}

    for treat, label in (("control", "Control"), ("stress", "Drought stress")):
        pairs = manual_counts(treat)
        files = [f for f, _ in pairs]
        man = [c for _, c in pairs]
        ai = ai_counts(treat, files)
        per_image[label] = (np.array(man, float), np.array(ai, float))

        groups = defaultdict(lambda: [[], []])
        for f, m, a in zip(files, man, ai):
            g = groups[plant_of(f, treat)]
            g[0].append(m); g[1].append(a)
        plants = sorted(groups)
        pm = np.array([np.mean(groups[p][0]) for p in plants]) * K
        pa = np.array([np.mean(groups[p][1]) for p in plants]) * K
        per_plant[label] = (pm, pa)
        sizes = [len(groups[p][0]) for p in plants]
        print(f"{label:<15} images n={len(man):3d} | plants n={len(plants):3d}  "
              f"images/plant {min(sizes)}-{max(sizes)} (mean {np.mean(sizes):.1f})")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    panels = [("A", "Stomatal count", "Control",        per_image, "count"),
              ("B", "Stomatal count", "Drought stress", per_image, "count"),
              ("C", "Stomatal density", "Control",        per_plant, "density"),
              ("D", "Stomatal density", "Drought stress", per_plant, "density")]
    for ax, (letter, kind, label, source, unit) in zip(axes.ravel(), panels):
        x, y = source[label]
        s, b, r2 = fit(x, y)
        lim = max(x.max(), y.max()) * 1.12
        ax.plot([0, lim], [0, lim], "--", color="0.45", lw=1.3,
                label="Identity (AI = manual)")
        ax.scatter(x, y, s=44, color=BLUE, alpha=0.78, edgecolor="white", lw=0.6)
        xs = np.linspace(0, lim, 50)
        ax.plot(xs, s * xs + b, color=BLUE, lw=2.0)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        if unit == "count":
            ax.set_xlabel("Manual counting"); ax.set_ylabel("AI counting")
        else:
            ax.set_xlabel("Manual stomatal density")
            ax.set_ylabel("AI stomatal density")
        ax.set_title(f"({letter}) {kind} \u2014 {label.lower()}", loc="left")
        ax.text(0.97, 0.06, f"y = {s:.4f}x + {b:.4f}\n$R^2$ = {r2:.4f}\nn = {len(x)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5)
        ax.legend(loc="upper left", fontsize=10)
        print(f"  ({letter}) {kind:<17} {label:<15} n={len(x):3d}  "
              f"y={s:.4f}x+{b:.4f}  R2={r2:.4f}")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig10_manual_vs_ai.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{OUT}/fig10_manual_vs_ai.pdf", bbox_inches="tight")
    print(f"\nwrote {OUT}/fig10_manual_vs_ai.png / .pdf")


if __name__ == "__main__":
    main()
