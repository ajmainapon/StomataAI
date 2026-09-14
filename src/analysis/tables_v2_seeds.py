"""Tables 2 and 3 for v2 as mean +/- SD across seeds.

YOLO models are evaluated per treatment subset for every available seed.
Mask R-CNN reuses each seed's cached Phase-D test predictions from its
validation-selected checkpoint, re-scored with COCOeval on each subset.
Re-runnable: seeds that are not yet trained are simply skipped.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import statistics
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("YOLO_VERBOSE", "False")

import yaml  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402
from ultralytics import YOLO  # noqa: E402

V2 = "/home/rmedu2026/stomataAI/v2"
YOLODATA = f"{V2}/data/yolo"
OUT = f"{V2}/tables"
os.makedirs(OUT, exist_ok=True)

SEEDS = (42, 43, 44)
SUBSETS = ("control", "stress", "combined")
MODELS = ("YOLOv8s-seg", "YOLO26s-seg", "Mask R-CNN")
METRICS = ("box_mAP50", "box_mAP", "mask_mAP50", "mask_mAP", "guard", "pore")


def subset_yaml(subset):
    d = os.path.join(YOLODATA, "images/test")
    files = sorted(
        os.path.join(d, f) for f in os.listdir(d)
        if subset == "combined" or f.startswith(f"{subset}_")
    )
    lst = os.path.join(OUT, f"{subset}_test.txt")
    open(lst, "w").write("\n".join(files) + "\n")
    cfg = os.path.join(OUT, f"{subset}.yaml")
    yaml.safe_dump(
        {"path": YOLODATA, "train": lst, "val": lst, "test": lst,
         "names": {0: "stomata", 1: "guard", 2: "pore"}},
        open(cfg, "w"), sort_keys=False)
    return cfg


def yolo_eval(weights, subset, tag, device):
    cfg = subset_yaml(subset)
    with contextlib.redirect_stdout(io.StringIO()):
        r = YOLO(weights).val(data=cfg, split="test", imgsz=640, batch=1,
                              device=device, plots=False, verbose=False,
                              project=OUT, name=f"tmp_{tag}", exist_ok=True)
    idx = list(r.seg.ap_class_index)
    pc = {r.names[c]: float(v) for c, v in zip(idx, r.seg.maps[idx])}
    return {"box_mAP50": float(r.box.map50), "box_mAP": float(r.box.map),
            "mask_mAP50": float(r.seg.map50), "mask_mAP": float(r.seg.map),
            "guard": pc.get("guard"), "pore": pc.get("pore")}


def mrcnn_eval(run, subset):
    sel = f"{run}/selection"
    v = json.load(open(f"{sel}/selection_valid.json"))
    best = v["best_checkpoint"]
    gt_all = json.load(open(f"{sel}/ground_truth_test.json"))
    preds = json.load(open(f"{sel}/pred_test_{best}.json"))
    keep = {i["id"] for i in gt_all["images"]
            if subset == "combined" or i["file_name"].startswith(f"{subset}_")}
    gt = dict(gt_all)
    gt["images"] = [i for i in gt_all["images"] if i["id"] in keep]
    gt["annotations"] = [a for a in gt_all["annotations"] if a["image_id"] in keep]
    gp, dp = f"{OUT}/_g.json", f"{OUT}/_d.json"
    json.dump(gt, open(gp, "w"))
    json.dump([p for p in preds if p["image_id"] in keep], open(dp, "w"))
    with contextlib.redirect_stdout(io.StringIO()):
        c = COCO(gp); d = c.loadRes(dp)

    def ev(kind, cats=None):
        e = COCOeval(c, d, kind)
        if cats:
            e.params.catIds = cats
        with contextlib.redirect_stdout(io.StringIO()):
            e.evaluate(); e.accumulate(); e.summarize()
        return float(e.stats[1]), float(e.stats[0])

    b50, b = ev("bbox"); m50, m = ev("segm")
    return {"box_mAP50": b50, "box_mAP": b, "mask_mAP50": m50, "mask_mAP": m,
            "guard": ev("segm", [2])[1], "pore": ev("segm", [3])[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    raw = {m: {s: {} for s in SUBSETS} for m in MODELS}
    for seed in SEEDS:
        for model, sub in (("YOLOv8s-seg", "yolov8s"), ("YOLO26s-seg", "yolo26s")):
            w = f"{V2}/runs/{sub}_640_s{seed}/weights/best.pt"
            if not os.path.exists(w):
                continue
            for s in SUBSETS:
                raw[model][s][seed] = yolo_eval(w, s, f"{sub}{seed}{s}", args.device)
        run = f"{V2}/runs/maskrcnn_orig800_s{seed}"
        if os.path.exists(f"{run}/selection/selection_valid.json"):
            for s in SUBSETS:
                raw["Mask R-CNN"][s][seed] = mrcnn_eval(run, s)

    def agg(model, subset, key):
        vals = [v[key] for v in raw[model][subset].values() if v.get(key) is not None]
        if not vals:
            return None, None, 0
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return statistics.mean(vals), sd, len(vals)

    def cell(model, subset, key):
        m, sd, n = agg(model, subset, key)
        if m is None:
            return "     --   "
        return f"{m:.3f}±{sd:.3f}" if n > 1 else f"{m:.3f} (n=1)"

    disp = {"control": "Control", "stress": "Drought stress", "combined": "Combined"}
    print("=" * 96)
    print("TABLE 2  box and mask AP by treatment subset, mean +/- SD across seeds (v2)")
    print("=" * 96)
    print(f"{'Test subset':<16}{'Model':<14}{'Box@0.50':>14}{'Box@.50:.95':>14}"
          f"{'Mask@0.50':>14}{'Mask@.50:.95':>14}   n")
    for s in SUBSETS:
        for m in MODELS:
            n = agg(m, s, "mask_mAP")[2]
            print(f"{disp[s]:<16}{m:<14}{cell(m,s,'box_mAP50'):>14}"
                  f"{cell(m,s,'box_mAP'):>14}{cell(m,s,'mask_mAP50'):>14}"
                  f"{cell(m,s,'mask_mAP'):>14}   {n}")

    print()
    print("=" * 96)
    print("TABLE 3  class-specific mask AP@0.50:0.95, mean +/- SD across seeds (v2)")
    print("=" * 96)
    print(f"{'Model':<14}{'Treatment':<16}{'Guard-cell':>14}{'Pore':>14}{'Macro':>14}   n")
    for m in MODELS:
        for s in ("control", "stress"):
            n = agg(m, s, "pore")[2]
            print(f"{m:<14}{disp[s]:<16}{cell(m,s,'guard'):>14}"
                  f"{cell(m,s,'pore'):>14}{cell(m,s,'mask_mAP'):>14}   {n}")

    json.dump(raw, open(f"{OUT}/v2_tables_seeds.json", "w"), indent=2)
    print(f"\nwrote {OUT}/v2_tables_seeds.json")


if __name__ == "__main__":
    main()
