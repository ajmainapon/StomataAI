import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

ROOT = Path("/home/rmedu2026/stomataAI/raw")
OUT = Path("/home/rmedu2026/stomataAI/runs/maskrcnn")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NAMES = {"stomata": 1, "guard": 2, "pore": 3}

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CocoSegDataset(Dataset):
    def __init__(self, treatment, split):
        self.treatment = treatment
        self.root = ROOT / treatment / split
        data = json.loads((self.root / "_annotations.coco.json").read_text())
        self.images = data["images"]
        self.categories = {c["id"]: NAMES[c["name"]] for c in data["categories"]}
        self.anns = {}
        for ann in data["annotations"]:
            self.anns.setdefault(ann["image_id"], []).append(ann)

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
                if mask.ndim == 3:
                    mask = mask.max(axis=2)
            elif isinstance(seg, dict):
                mask = mask_utils.decode(seg)
                if mask.ndim == 3:
                    mask = mask.max(axis=2)
            else:
                continue
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            ys, xs = torch.where(mask > 0)
            if len(xs) == 0:
                continue
            x0, x1 = xs.min().item(), xs.max().item() + 1
            y0, y1 = ys.min().item(), ys.max().item() + 1
            masks.append(mask)
            boxes.append([x0, y0, x1, y1])
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
        target = {"boxes": boxes, "labels": labels, "masks": masks,
                  "image_id": torch.tensor([idx]), "area": areas,
                  "iscrowd": torch.zeros((len(labels),), dtype=torch.int64)}
        return image, target

def collate(batch):
    return tuple(zip(*batch))

def build_model():
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden, 4)
    return model

def main():
    seed_all()
    train = ConcatDataset([CocoSegDataset("control", "train"), CocoSegDataset("stress", "train")])
    val = ConcatDataset([CocoSegDataset("control", "valid"), CocoSegDataset("stress", "valid")])
    train_loader = DataLoader(train, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate, pin_memory=True)
    model = build_model().to(DEVICE)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "train.log"
    with log.open("w") as f:
        for epoch in range(1, 31):
            model.train()
            running = 0.0
            for images, targets in train_loader:
                images = [x.to(DEVICE) for x in images]
                targets = [{k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                    losses = model(images, targets)
                    loss = sum(losses.values())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running += float(loss.detach().cpu())
            msg = f"epoch={epoch} train_loss={running / max(1, len(train_loader)):.5f}"
            print(msg, flush=True)
            f.write(msg + "\n")
            if epoch % 5 == 0 or epoch == 1:
                torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, OUT / f"checkpoint_{epoch:03d}.pt")
        torch.save(model.state_dict(), OUT / "maskrcnn_resnet50_fpn_final.pt")

if __name__ == "__main__":
    main()
