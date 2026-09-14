"""Select the Mask R-CNN checkpoint by validation mask mAP@0.50:0.95.

Applies the same model-selection rule already used for the YOLO models, which
were chosen by best validation mask mAP@0.50:0.95. Requires no retraining --
the epoch checkpoints saved during the original run are re-scored on the
validation split.

Usage:
  python src/select_maskrcnn_checkpoint.py --split valid --out runs/maskrcnn/selection
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms.functional import pil_to_tensor

ROOT = Path("/home/rmedu2026/stomataAI/raw")
NAME_TO_ID = {"stomata": 1, "guard": 2, "pore": 3}
ID_TO_NAME = {v: k for k, v in NAME_TO_ID.items()}


def build_model(weights: Path, min_size: int, max_size: int):
    model = maskrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, min_size=min_size, max_size=max_size
    )
    bf = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(bf, 4)
    mf = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mf, 256, 4)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    # epoch checkpoints wrap the weights; the final export does not
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    return model


def load_records(split: str):
    records, gt_images, gt_anns = [], [], []
    img_id = ann_id = 1
    for treatment in ("control", "stress"):
        folder = ROOT / treatment / split
        source = json.loads((folder / "_annotations.coco.json").read_text())
        cat_map = {c["id"]: NAME_TO_ID[c["name"]] for c in source["categories"]}
        by_image = {}
        for ann in source["annotations"]:
            by_image.setdefault(ann["image_id"], []).append(ann)
        for info in source["images"]:
            records.append((img_id, folder / Path(info["file_name"]).name))
            gt_images.append(
                {
                    "id": img_id,
                    "file_name": f"{treatment}_{Path(info['file_name']).name}",
                    "width": info["width"],
                    "height": info["height"],
                }
            )
            for ann in by_image.get(info["id"], []):
                gt_anns.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_map[ann["category_id"]],
                        "segmentation": ann["segmentation"],
                        "bbox": ann["bbox"],
                        "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                        "iscrowd": ann.get("iscrowd", 0),
                    }
                )
                ann_id += 1
            img_id += 1
    dataset = {
        "images": gt_images,
        "annotations": gt_anns,
        "categories": [{"id": i, "name": ID_TO_NAME[i]} for i in sorted(ID_TO_NAME)],
        "info": {"description": f"combined control and stress {split} split"},
        "licenses": [],
    }
    return records, dataset


def encode(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _load(rec):
    image_id, path = rec
    return image_id, pil_to_tensor(Image.open(path).convert("RGB")).float() / 255.0


def _stream(records, workers: int = 8, depth: int = 16):
    """Decode ahead by at most `depth` images while inference consumes them.

    Full-resolution JPEG decode dominates this loop, so overlapping it with GPU
    work roughly halves the sweep time. The window is bounded because each
    decoded image is about 59 MB as float32.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        it = iter(records)
        for _ in range(depth):
            try:
                pending.append(pool.submit(_load, next(it)))
            except StopIteration:
                break
        while pending:
            fut = pending.popleft()
            try:
                pending.append(pool.submit(_load, next(it)))
            except StopIteration:
                pass
            yield fut.result()


def predict(model, records, device):
    preds = []
    model.eval()
    with torch.inference_mode():
        for image_id, image in _stream(records):
            out = model([image.to(device)])[0]
            for box, label, score, mask in zip(
                out["boxes"].cpu().numpy(),
                out["labels"].cpu().numpy(),
                out["scores"].cpu().numpy(),
                out["masks"].cpu().numpy()[:, 0],
            ):
                x0, y0, x1, y1 = box.tolist()
                preds.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "segmentation": encode(mask >= 0.5),
                        "score": float(score),
                    }
                )
    return preds


def summarize(gt, dt, iou_type, cat_ids=None):
    ev = COCOeval(gt, dt, iou_type)
    if cat_ids is not None:
        ev.params.catIds = cat_ids
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {
        "AP50_95": float(ev.stats[0]),
        "AP50": float(ev.stats[1]),
        "AP75": float(ev.stats[2]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/home/rmedu2026/stomataAI/runs/maskrcnn")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--out", default="/home/rmedu2026/stomataAI/runs/maskrcnn/selection")
    ap.add_argument("--min-size", type=int, default=800)
    ap.add_argument("--max-size", type=int, default=1333)
    ap.add_argument("--glob", default="checkpoint_*.pt")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  split={args.split}  size={args.min_size}/{args.max_size}", flush=True)

    records, dataset = load_records(args.split)
    gt_path = out / f"ground_truth_{args.split}.json"
    gt_path.write_text(json.dumps(dataset))
    gt = COCO(str(gt_path))
    print(f"{len(records)} images, {len(dataset['annotations'])} annotations", flush=True)

    ckpts = sorted(Path(args.ckpt_dir).glob(args.glob))
    results = {}
    for ckpt in ckpts:
        model = build_model(ckpt, args.min_size, args.max_size).to(device)
        preds = predict(model, records, device)
        del model
        torch.cuda.empty_cache()
        pred_path = out / f"pred_{args.split}_{ckpt.stem}.json"
        pred_path.write_text(json.dumps(preds))
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt.loadRes(str(pred_path))
        entry = {"n_predictions": len(preds)}
        for kind in ("bbox", "segm"):
            entry[kind] = summarize(gt, dt, kind)
            for cid, name in ID_TO_NAME.items():
                if name == "stomata":
                    continue
                entry.setdefault(f"{kind}_per_class", {})[name] = summarize(
                    gt, dt, kind, [cid]
                )
        results[ckpt.stem] = entry
        s = entry["segm"]
        pc = entry["segm_per_class"]
        print(
            f"{ckpt.stem}:  mask mAP@0.50:0.95 = {s['AP50_95']:.4f}   "
            f"mask mAP@0.50 = {s['AP50']:.4f}   "
            f"guard = {pc['guard']['AP50_95']:.4f}   pore = {pc['pore']['AP50_95']:.4f}",
            flush=True,
        )

    best = max(results, key=lambda k: results[k]["segm"]["AP50_95"])
    payload = {
        "split": args.split,
        "min_size": args.min_size,
        "max_size": args.max_size,
        "selection_metric": "segm AP50_95",
        "best_checkpoint": best,
        "results": results,
    }
    (out / f"selection_{args.split}.json").write_text(json.dumps(payload, indent=2))
    print(f"\nBEST by validation mask mAP@0.50:0.95 -> {best} "
          f"({results[best]['segm']['AP50_95']:.4f})", flush=True)


if __name__ == "__main__":
    main()
