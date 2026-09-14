"""Regenerate the model figures on the v2 dataset.

  fig4  training / validation loss and validation mask AP, both YOLO models
  fig5  normalised detection confusion matrices
  fig6  mask precision-recall and F1-confidence curves
  fig8  mask AP by treatment subset, mean +/- SD across seeds

Curves and matrices are inherently per-run and use seed 42; fig8 aggregates
the three seeds. Written as PDF (vector) plus 600 dpi PNG.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import statistics
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("YOLO_VERBOSE", "False")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402

V2 = "/home/rmedu2026/stomataAI/v2"
OUT = f"{V2}/figures"
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GREEN, INK = "#1f77b4", "#d95f02", "#2f7350", "#191c21"
MODELS = {"YOLOv8s-seg": "yolov8s", "YOLO26s-seg": "yolo26s"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 110,
})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def read_results(sub, seed=42):
    rows = list(csv.DictReader(open(f"{V2}/runs/{sub}_640_s{seed}/results.csv")))
    def col(frag, exclude=None):
        k = next(k for k in rows[0]
                 if frag in k and (exclude is None or exclude not in k))
        return np.array([float(r[k]) for r in rows])
    return {
        "epoch": col("epoch"),
        "train_seg": col("train/seg_loss"),
        "val_seg": col("val/seg_loss"),
        "map50": col("mAP50(M)", exclude="95"),
        "map": col("mAP50-95(M)"),
    }


# ---------------------------------------------------------------- figure 4 --
def figure4():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, (label, sub) in enumerate(MODELS.items()):
        d = read_results(sub)
        best = int(d["epoch"][int(np.argmax(d["map"]))])
        bi = int(np.argmax(d["map"]))

        ax = axes[row, 0]
        ax.plot(d["epoch"], d["train_seg"], color=BLUE, lw=1.6, label="Training loss")
        ax.plot(d["epoch"], d["val_seg"], color=ORANGE, lw=1.6, label="Validation loss")
        ax.axvline(best, color="0.35", ls="--", lw=1.2)
        ax.plot(best, d["val_seg"][bi], "D", color=INK, ms=8, zorder=5)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(f"{'AC'[row]}  {label}: segmentation loss", loc="left")
        ax.legend(loc="upper right")

        ax = axes[row, 1]
        ax.plot(d["epoch"], d["map50"], color=BLUE, lw=1.6, label="Mask mAP@0.50")
        ax.plot(d["epoch"], d["map"], color=ORANGE, lw=1.6, label="Mask mAP@0.50:0.95")
        ax.axvline(best, color="0.35", ls="--", lw=1.2)
        ax.plot(best, d["map50"][bi], "D", color=INK, ms=8, zorder=5)
        ax.plot(best, d["map"][bi], "D", color=INK, ms=8, zorder=5)
        ax.set_ylim(0, 0.75)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Average precision")
        ax.set_title(f"{'BD'[row]}  {label}: validation mask AP", loc="left")
        ax.annotate(f"Selected: epoch {best}", xy=(best, 0.70),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=10, color="0.3", va="top")
        ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig4_training_curves")


# ------------------------------------------------------- val for 5 and 6 ----
def subset_yaml():
    d = f"{V2}/data/yolo/images/test"
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jpg"))
    lst = f"{OUT}/_test.txt"
    open(lst, "w").write("\n".join(files) + "\n")
    cfg = f"{OUT}/_test.yaml"
    yaml.safe_dump({"path": f"{V2}/data/yolo", "train": lst, "val": lst,
                    "test": lst, "names": {0: "stomata", 1: "guard", 2: "pore"}},
                   open(cfg, "w"), sort_keys=False)
    return cfg


def run_val(sub, cfg, device):
    w = f"{V2}/runs/{sub}_640_s42/weights/best.pt"
    with contextlib.redirect_stdout(io.StringIO()):
        # plots=True is required: Ultralytics only populates confusion_matrix
        # when plotting is enabled, otherwise it stays all-zero.
        r = YOLO(w).val(data=cfg, split="test", imgsz=640, batch=1, device=device,
                        plots=True, verbose=False, project=OUT,
                        name=f"_val_{sub}", exist_ok=True)
    return r


# ---------------------------------------------------------------- figure 5 --
def figure5(results):
    labels = ["Guard cell", "Pore", "Background"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    fig.subplots_adjust(wspace=0.42)
    im = None
    for ax, (label, r) in zip(axes, results.items()):
        m = np.array(r.confusion_matrix.matrix, dtype=float)
        keep = [1, 2, 3] if m.shape[0] == 4 else list(range(m.shape[0]))
        m = m[np.ix_(keep, keep)]
        col = m.sum(axis=0, keepdims=True)
        norm = np.divide(m, col, out=np.zeros_like(m), where=col > 0)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                v = norm[i, j]
                ax.text(j, i, f"{v:.2f}".rstrip("0").rstrip(".") if v else "0",
                        ha="center", va="center", fontsize=13,
                        fontweight="bold" if v > 0.5 else "normal",
                        color="white" if v > 0.55 else INK)
        idx = list(results).index(label)
        ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("True class")
        if idx == 0:                      # only the left panel carries the y-label,
            ax.set_ylabel("Predicted class")   # otherwise it overlaps panel A
        ax.set_title(f"{'AB'[idx]}  {label}", loc="left", pad=10)
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.82, pad=0.03,
                 label="Proportion within true class")
    save(fig, "fig5_confusion_matrices")


# ---------------------------------------------------------------- figure 6 --
def figure6(results):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, (label, r) in enumerate(results.items()):
        curves = {c[2] + "|" + c[3]: c for c in r.seg.curves_results}
        pr = next(c for k, c in curves.items() if k.startswith("Recall|Precision"))
        f1 = next(c for k, c in curves.items() if "F1" in k)
        names = [r.names[i] for i in r.seg.ap_class_index]

        ax = axes[row, 0]
        x, ys = pr[0], np.atleast_2d(pr[1])
        for y, n, c in zip(ys, names, (BLUE, ORANGE)):
            ap = r.seg.maps[r.seg.ap_class_index[names.index(n)]]
            disp = "Guard cell" if n == "guard" else "Pore"
            ax.plot(x, y, color=c, lw=1.8,
                    label=f"{disp} (AP50={r.seg.ap50[names.index(n)]:.3f})")
        ax.plot(x, ys.mean(0), color=INK, lw=2.6,
                label=f"Macro (mAP50={r.seg.map50:.3f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_title(f"{'AC'[row]}  {label}: mask precision-recall", loc="left")
        ax.legend(loc="lower left")

        ax = axes[row, 1]
        x, ys = f1[0], np.atleast_2d(f1[1])
        for y, n, c in zip(ys, names, (BLUE, ORANGE)):
            disp = "Guard cell" if n == "guard" else "Pore"
            ax.plot(x, y, color=c, lw=1.8, label=disp)
        macro = ys.mean(0)
        bi = int(np.argmax(macro))
        ax.plot(x, macro, color=INK, lw=2.6,
                label=f"Macro ({macro[bi]:.2f} at confidence {x[bi]:.3f})")
        ax.set_xlabel("Confidence threshold"); ax.set_ylabel("F1")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_title(f"{'BD'[row]}  {label}: mask F1-confidence", loc="left")
        ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig6_pr_f1_curves")


# ---------------------------------------------------------------- figure 8 --
def figure8():
    path = f"{V2}/tables/v2_tables_seeds.json"
    if not os.path.exists(path):
        print("  fig8 skipped: tables json not present")
        return
    raw = json.load(open(path))
    subs = ["control", "stress", "combined"]
    disp = ["Control", "Drought stress", "Combined"]
    models = ["YOLOv8s-seg", "YOLO26s-seg", "Mask R-CNN"]
    cols = [BLUE, ORANGE, GREEN]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, key, title in zip(axes, ("mask_mAP50", "mask_mAP"),
                              ("A  Mask mAP@0.50", "B  Mask mAP@0.50:0.95")):
        w, xs = 0.26, np.arange(len(subs))
        for i, (m, c) in enumerate(zip(models, cols)):
            means, sds = [], []
            for s in subs:
                vals = [v[key] for v in raw[m][s].values() if v.get(key) is not None]
                means.append(statistics.mean(vals) if vals else 0)
                sds.append(statistics.stdev(vals) if len(vals) > 1 else 0)
            b = ax.bar(xs + (i - 1) * w, means, w, yerr=sds, capsize=3,
                       color=c, label=m,
                       error_kw={"lw": 1.1, "ecolor": "0.25"})
            for rect, mu in zip(b, means):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + 0.012, f"{mu:.3f}",
                        ha="center", va="bottom", fontsize=9, rotation=90)
        ax.set_xticks(xs, disp)
        ax.set_ylim(0, 0.85); ax.set_ylabel("Average precision")
        ax.set_title(title, loc="left")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, 1.16), ncol=3)
    fig.tight_layout()
    save(fig, "fig8_ap_by_treatment")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    want = set(a.only.split(",")) if a.only else {"4", "5", "6", "8"}

    if "4" in want:
        print("figure 4"); figure4()
    if want & {"5", "6"}:
        cfg = subset_yaml()
        res = {k: run_val(v, cfg, a.device) for k, v in MODELS.items()}
        if "5" in want:
            print("figure 5"); figure5(res)
        if "6" in want:
            print("figure 6"); figure6(res)
    if "8" in want:
        print("figure 8"); figure8()


if __name__ == "__main__":
    main()
