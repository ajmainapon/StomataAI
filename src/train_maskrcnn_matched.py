"""Mask R-CNN training with configurable input resolution and augmentation.

Differences from the original maskrcnn_train.py, each addressing a documented
limitation of the published benchmark:

  * --min-size/--max-size expose the torchvision resize, which previously fell
    back to the 800/1333 default and gave Mask R-CNN roughly three times the
    pixel area of the 640-pixel YOLO models.
  * horizontal flip and colour jitter approximate the YOLO augmentation
    (fliplr 0.5, hsv_h 0.015, hsv_s 0.7, hsv_v 0.4). Mosaic has no torchvision
    equivalent and is not reproduced.
  * a checkpoint is written every epoch so the model can be selected on
    validation mask mAP@0.50:0.95, the rule already used for the YOLO models.
  * --seed is exposed so the run can be repeated for uncertainty estimates.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import ColorJitter
from torchvision.transforms.functional import pil_to_tensor

ROOT = Path("/home/rmedu2026/stomataAI/raw")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NAMES = {"stomata": 1, "guard": 2, "pore": 3}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CocoSegDataset(Dataset):
    def __init__(self, treatment, split, augment=False, jitter=True):
        self.root = ROOT / treatment / split
        data = json.loads((self.root / "_annotations.coco.json").read_text())
        self.images = data["images"]
        self.categories = {c["id"]: NAMES[c["name"]] for c in data["categories"]}
        self.anns = {}
        for ann in data["annotations"]:
            self.anns.setdefault(ann["image_id"], []).append(ann)
        self.augment = augment
        self.jitter = (
            ColorJitter(brightness=0.4, saturation=0.7, hue=0.015)
            if (augment and jitter)
            else None
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        path = self.root / Path(info["file_name"]).name
        image = pil_to_tensor(Image.open(path).convert("RGB")).float() / 255.0
        h, w = image.shape[-2:]

        masks, boxes, labels, areas = [], [], [], []
        for ann in self.anns.get(info["id"], []):
            seg = ann.get("segmentation")
            if isinstance(seg, list) and seg:
                polys = seg if isinstance(seg[0], list) else [seg]
                rles = mask_utils.frPyObjects(polys, h, w)
                mask = mask_utils.decode(mask_utils.merge(rles))
            elif isinstance(seg, dict):
                mask = mask_utils.decode(seg)
            else:
                continue
            if mask.ndim == 3:
                mask = mask.max(axis=2)
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            ys, xs = torch.where(mask > 0)
            if len(xs) == 0:
                continue
            masks.append(mask)
            boxes.append(
                [xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1]
            )
            labels.append(self.categories[ann["category_id"]])
            areas.append(float(mask.sum()))

        if masks:
            masks = torch.stack(masks)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
        else:
            masks = torch.zeros((0, h, w), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)

        # ---- augmentation (train split only) ----------------------------------
        if self.augment:
            if self.jitter is not None:
                image = self.jitter(image)
            if random.random() < 0.5:                      # matches YOLO fliplr 0.5
                image = torch.flip(image, dims=[-1])
                if len(masks):
                    masks = torch.flip(masks, dims=[-1])
                    boxes = boxes.clone()
                    x0 = boxes[:, 0].clone()
                    boxes[:, 0] = w - boxes[:, 2]
                    boxes[:, 2] = w - x0

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": areas,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        return image, target


def collate(batch):
    return tuple(zip(*batch))


def build_model(min_size: int, max_size: int):
    model = maskrcnn_resnet50_fpn(
        weights="DEFAULT", min_size=min_size, max_size=max_size
    )
    bf = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(bf, 4)
    mf = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mf, 256, 4)
    return model


@torch.no_grad()
def validation_loss(model, loader) -> float:
    """Mean total loss over the validation split (model kept in train mode so
    the loss heads stay active; no gradients are taken)."""
    model.train()
    total, n = 0.0, 0
    for images, targets in loader:
        images = [x.to(DEVICE) for x in images]
        targets = [
            {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in t.items()}
            for t in targets
        ]
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=DEVICE.type == "cuda"):
            losses = model(images, targets)
            total += float(sum(losses.values()).detach().cpu())
        n += 1
    return total / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-size", type=int, default=480)
    ap.add_argument("--max-size", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=12,
                    help="dataloader workers; the pipeline is input-bound")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    augment = not args.no_augment

    train = ConcatDataset(
        [
            CocoSegDataset("control", "train", augment=augment),
            CocoSegDataset("stress", "train", augment=augment),
        ]
    )
    val = ConcatDataset(
        [CocoSegDataset("control", "valid"), CocoSegDataset("stress", "valid")]
    )
    # The original script used num_workers=4 on a 24-core host, which left the
    # GPU idle ~76% of the time (measured). Full-resolution JPEG decode plus
    # polygon rasterisation dominates, so widen the input pipeline.
    train_loader = DataLoader(
        train, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        collate_fn=collate, pin_memory=True,
        persistent_workers=args.workers > 0, prefetch_factor=2,
    )
    val_workers = max(2, args.workers // 2)
    val_loader = DataLoader(
        val, batch_size=1, shuffle=False, num_workers=val_workers,
        collate_fn=collate, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    model = build_model(args.min_size, args.max_size).to(DEVICE)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    log = (out / "train.log").open("w")
    print(
        f"seed={args.seed} size={args.min_size}/{args.max_size} "
        f"batch={args.batch} epochs={args.epochs} augment={augment}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, targets in train_loader:
            images = [x.to(DEVICE) for x in images]
            targets = [
                {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in t.items()}
                for t in targets
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=DEVICE.type == "cuda"):
                losses = model(images, targets)
                loss = sum(losses.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().cpu())
        vl = validation_loss(model, val_loader)
        msg = (
            f"epoch={epoch} train_loss={running / max(1, len(train_loader)):.5f} "
            f"val_loss={vl:.5f}"
        )
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
        torch.save(
            {"epoch": epoch, "model": model.state_dict()},
            out / f"checkpoint_{epoch:03d}.pt",
        )

    torch.save(model.state_dict(), out / "final.pt")
    log.close()


if __name__ == "__main__":
    main()
