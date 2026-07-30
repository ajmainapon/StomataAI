import json
from pathlib import Path

import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

ROOT = Path("/home/rmedu2026/stomataAI")
DATA = ROOT / "baseline_data"
OUT = ROOT / "runs" / "treatment_evaluation"
NAMES = {0: "stomata", 1: "guard", 2: "pore"}
MODELS = {
    "yolov8s": ROOT / "runs/yolov8s_combined/weights/best.pt",
    "yolo26s": ROOT / "runs/yolo26s_combined/weights/best.pt",
}


def subset_config(treatment):
    files = sorted((DATA / "images/test").glob(f"{treatment}_*"))
    image_list = OUT / f"{treatment}_test.txt"
    image_list.write_text("\n".join(str(p) for p in files) + "\n")
    config = OUT / f"{treatment}.yaml"
    config.write_text(yaml.safe_dump({"train": str(image_list), "val": str(image_list), "test": str(image_list), "names": NAMES}, sort_keys=False))
    return config, len(files)


def yolo_results():
    collected = {}
    for treatment in ("control", "stress"):
        config, count = subset_config(treatment)
        collected[treatment] = {"images": count, "models": {}}
        for model_name, weights in MODELS.items():
            metrics = YOLO(weights).val(
                data=config,
                split="test",
                imgsz=640,
                batch=1,
                device="cpu",
                workers=4,
                plots=False,
                project=OUT,
                name=f"{model_name}_{treatment}",
            )
            item = {k: float(v) for k, v in metrics.results_dict.items()}
            item["box_per_class_mAP50_95"] = {NAMES[i]: float(v) for i, v in enumerate(metrics.box.maps)}
            item["mask_per_class_mAP50_95"] = {NAMES[i]: float(v) for i, v in enumerate(metrics.seg.maps)}
            collected[treatment]["models"][model_name] = item
    return collected


def coco_summary(gt, dt, kind, category_ids=None):
    evaluator = COCOeval(gt, dt, kind)
    if category_ids is not None:
        evaluator.params.catIds = category_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {"mAP50_95": float(evaluator.stats[0]), "mAP50": float(evaluator.stats[1]), "AP75": float(evaluator.stats[2])}


def maskrcnn_results():
    source_dir = ROOT / "runs/maskrcnn/evaluation"
    source_gt = json.loads((source_dir / "ground_truth.json").read_text())
    source_predictions = json.loads((source_dir / "predictions.json").read_text())
    results = {}
    for treatment in ("control", "stress"):
        images = [i for i in source_gt["images"] if i["file_name"].startswith(f"{treatment}_")]
        ids = {i["id"] for i in images}
        annotations = [a for a in source_gt["annotations"] if a["image_id"] in ids]
        predictions = [p for p in source_predictions if p["image_id"] in ids]
        subset = {
            "images": images,
            "annotations": annotations,
            "categories": source_gt["categories"],
            "info": source_gt.get("info", {}),
            "licenses": [],
        }
        gt_path = OUT / f"maskrcnn_{treatment}_ground_truth.json"
        pred_path = OUT / f"maskrcnn_{treatment}_predictions.json"
        gt_path.write_text(json.dumps(subset))
        pred_path.write_text(json.dumps(predictions))
        gt = COCO(str(gt_path))
        dt = gt.loadRes(str(pred_path))
        result = {"images": len(images), "instances": len(annotations), "bbox": {}, "mask": {}, "per_class": {}}
        result["bbox"] = coco_summary(gt, dt, "bbox")
        result["mask"] = coco_summary(gt, dt, "segm")
        for category in source_gt["categories"]:
            category_id, name = category["id"], category["name"]
            result["per_class"][name] = {
                "bbox": coco_summary(gt, dt, "bbox", [category_id]),
                "mask": coco_summary(gt, dt, "segm", [category_id]),
            }
        results[treatment] = result
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {"yolo": yolo_results(), "maskrcnn": maskrcnn_results()}
    path = OUT / "metrics_by_treatment.json"
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    print(f"saved={path}", flush=True)


if __name__ == "__main__":
    main()
