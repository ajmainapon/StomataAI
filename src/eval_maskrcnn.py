import json
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
OUT = Path("/home/rmedu2026/stomataAI/runs/maskrcnn/evaluation")
WEIGHTS = Path("/home/rmedu2026/stomataAI/runs/maskrcnn/maskrcnn_resnet50_fpn_final.pt")
NAME_TO_ID = {"stomata": 1, "guard": 2, "pore": 3}
ID_TO_NAME = {v: k for k, v in NAME_TO_ID.items()}


def build_model():
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, 4)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, 4)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    return model


def load_test_records():
    records = []
    gt_images, gt_annotations = [], []
    global_image_id = 1
    global_ann_id = 1
    for treatment in ("control", "stress"):
        folder = ROOT / treatment / "test"
        source = json.loads((folder / "_annotations.coco.json").read_text())
        category_map = {c["id"]: NAME_TO_ID[c["name"]] for c in source["categories"]}
        anns_by_image = {}
        for ann in source["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)
        for info in source["images"]:
            image_path = folder / Path(info["file_name"]).name
            records.append((global_image_id, image_path))
            gt_images.append({
                "id": global_image_id,
                "file_name": f"{treatment}_{Path(info['file_name']).name}",
                "width": info["width"],
                "height": info["height"],
            })
            for ann in anns_by_image.get(info["id"], []):
                copied = {
                    "id": global_ann_id,
                    "image_id": global_image_id,
                    "category_id": category_map[ann["category_id"]],
                    "segmentation": ann["segmentation"],
                    "bbox": ann["bbox"],
                    "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                    "iscrowd": ann.get("iscrowd", 0),
                }
                gt_annotations.append(copied)
                global_ann_id += 1
            global_image_id += 1
    dataset = {
        "images": gt_images,
        "annotations": gt_annotations,
        "categories": [{"id": i, "name": ID_TO_NAME[i]} for i in sorted(ID_TO_NAME)],
        "info": {"description": "Combined control and stress held-out test set"},
        "licenses": [],
    }
    return records, dataset


def encode_mask(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def predict(model, records, device):
    predictions = []
    model.eval()
    with torch.inference_mode():
        for index, (image_id, image_path) in enumerate(records, 1):
            image = pil_to_tensor(Image.open(image_path).convert("RGB")).float() / 255.0
            output = model([image.to(device)])[0]
            boxes = output["boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            masks = output["masks"].cpu().numpy()[:, 0]
            for box, label, score, mask in zip(boxes, labels, scores, masks):
                x0, y0, x1, y1 = box.tolist()
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "segmentation": encode_mask(mask >= 0.5),
                    "score": float(score),
                })
            print(f"evaluated {index}/{len(records)} images", flush=True)
    return predictions


def summarize(gt, dt, iou_type, cat_ids=None):
    evaluator = COCOeval(gt, dt, iou_type)
    if cat_ids is not None:
        evaluator.params.catIds = cat_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {"AP50_95": float(evaluator.stats[0]), "AP50": float(evaluator.stats[1]), "AP75": float(evaluator.stats[2])}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 2_000_000_000 else "cpu")
    print(f"device={device}", flush=True)
    records, dataset = load_test_records()
    (OUT / "ground_truth.json").write_text(json.dumps(dataset))
    model = build_model().to(device)
    predictions = predict(model, records, device)
    (OUT / "predictions.json").write_text(json.dumps(predictions))
    gt = COCO(str(OUT / "ground_truth.json"))
    dt = gt.loadRes(str(OUT / "predictions.json"))
    results = {"images": len(records), "predictions": len(predictions), "overall": {}, "per_class": {}}
    for kind in ("bbox", "segm"):
        results["overall"][kind] = summarize(gt, dt, kind)
        for category_id, name in ID_TO_NAME.items():
            results["per_class"].setdefault(name, {})[kind] = summarize(gt, dt, kind, [category_id])
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
