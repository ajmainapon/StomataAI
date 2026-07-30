"""Summarise validation-selected metrics for each YOLO seed run."""
import csv
import glob
import os
import statistics

BASE = "/home/rmedu2026/stomataAI/runs/seedstudy"

rows_out = []
for d in sorted(glob.glob(os.path.join(BASE, "*/"))):
    path = os.path.join(d, "results.csv")
    if not os.path.exists(path):
        continue
    rows = [r for r in csv.DictReader(open(path))]
    if not rows:
        continue
    k9 = next(k for k in rows[0] if "mAP50-95(M)" in k)
    k5 = next(k for k in rows[0] if "mAP50(M)" in k and "95" not in k)
    b9 = next(k for k in rows[0] if "mAP50-95(B)" in k)
    best = max(rows, key=lambda r: float(r[k9]))
    rows_out.append(
        {
            "run": os.path.basename(d.rstrip("/")),
            "epochs": len(rows),
            "best_epoch": int(float(best["epoch"])),
            "mask_mAP50_95": float(best[k9]),
            "mask_mAP50": float(best[k5]),
            "box_mAP50_95": float(best[b9]),
        }
    )

w = max(len(r["run"]) for r in rows_out)
print(f"{'run':<{w}}  epochs  best_ep   mask@.5:.95   mask@.50   box@.5:.95")
for r in rows_out:
    print(
        f"{r['run']:<{w}}  {r['epochs']:>6}  {r['best_epoch']:>7}   "
        f"{r['mask_mAP50_95']:>11.4f}   {r['mask_mAP50']:>8.4f}   {r['box_mAP50_95']:>10.4f}"
    )

print()
for family in ("yolov8s", "yolo26s"):
    vals = [r["mask_mAP50_95"] for r in rows_out if r["run"].startswith(family)]
    if len(vals) >= 2:
        print(
            f"{family}: n={len(vals)}  mean mask mAP@0.50:0.95 = {statistics.mean(vals):.4f}"
            f"  SD = {statistics.stdev(vals):.4f}"
            f"  range = {min(vals):.4f}-{max(vals):.4f}"
        )
    elif vals:
        print(f"{family}: n=1  {vals[0]:.4f}  (SD needs >=2 seeds)")
