"""Build the v2 dataset tree from the new Roboflow COCO export.

Produces two views of the same data:

  v2/data/raw/{control,stress}/{train,valid,test}/   COCO, consumed by Mask R-CNN
  v2/data/yolo/{images,labels}/{train,valid,test}/   YOLO-seg, treatment-prefixed

Class indices follow the COCO category ids in the export (0 stomata, 1 guard,
2 pore), which match the v1 data.yaml, so the trained-model class mapping is
unchanged.
"""
from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path

SRC = Path("/home/rmedu2026/stomataAI/new_export")
V2 = Path("/home/rmedu2026/stomataAI/v2")
RAW = V2 / "data" / "raw"
YOLO = V2 / "data" / "yolo"

TREATMENTS = ("control", "stress")
SPLITS = ("train", "valid", "test")


def largest_polygon(seg):
    """YOLO-seg takes one polygon per instance; keep the largest ring."""
    if isinstance(seg, dict):          # RLE -- not expected from this export
        return None
    polys = [p for p in seg if isinstance(p, list) and len(p) >= 6]
    if not polys:
        return None
    return max(polys, key=len)


def main() -> None:
    for split in SPLITS:
        (YOLO / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = Counter()
    skipped_multi = 0
    skipped_nopoly = 0

    for t in TREATMENTS:
        for split in SPLITS:
            src_dir = SRC / t / split
            ann_path = src_dir / "_annotations.coco.json"
            if not ann_path.exists():
                print(f"!! missing {ann_path}")
                continue

            # ---- COCO view (verbatim copy, for Mask R-CNN) ----------------
            dst_dir = RAW / t / split
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in src_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, dst_dir / f.name)

            # ---- YOLO view -------------------------------------------------
            data = json.loads(ann_path.read_text())
            by_image = {}
            for a in data["annotations"]:
                by_image.setdefault(a["image_id"], []).append(a)

            for info in data["images"]:
                w, h = info["width"], info["height"]
                name = os.path.basename(info["file_name"])
                new_name = f"{t}_{name}"
                shutil.copy2(src_dir / name, YOLO / "images" / split / new_name)

                lines = []
                for a in by_image.get(info["id"], []):
                    seg = a.get("segmentation")
                    if isinstance(seg, list) and len(seg) > 1:
                        skipped_multi += 1
                    poly = largest_polygon(seg)
                    if poly is None:
                        skipped_nopoly += 1
                        continue
                    coords = []
                    for i in range(0, len(poly) - 1, 2):
                        x = min(max(poly[i] / w, 0.0), 1.0)
                        y = min(max(poly[i + 1] / h, 0.0), 1.0)
                        coords += [f"{x:.6f}", f"{y:.6f}"]
                    lines.append(f"{a['category_id']} " + " ".join(coords))
                    stats[(t, split, a["category_id"])] += 1

                label = YOLO / "labels" / split / (Path(new_name).stem + ".txt")
                label.write_text("\n".join(lines) + ("\n" if lines else ""))

    # ---- data.yaml -----------------------------------------------------------
    (YOLO / "data.yaml").write_text(
        f"path: {YOLO}\n"
        "train: images/train\n"
        "val: images/valid\n"
        "test: images/test\n"
        "names:\n"
        "  0: stomata\n"
        "  1: guard\n"
        "  2: pore\n"
    )

    # ---- report ---------------------------------------------------------------
    names = {0: "stomata", 1: "guard", 2: "pore"}
    print(f"{'split':<18}{'images':>8}{'labels':>8}{'guard':>8}{'pore':>8}")
    total = Counter()
    for split in SPLITS:
        imgs = len(list((YOLO / "images" / split).glob("*.jpg")))
        labs = len(list((YOLO / "labels" / split).glob("*.txt")))
        g = sum(v for (t, s, c), v in stats.items() if s == split and c == 1)
        p = sum(v for (t, s, c), v in stats.items() if s == split and c == 2)
        print(f"{split:<18}{imgs:>8}{labs:>8}{g:>8}{p:>8}")
        total["images"] += imgs
        total["guard"] += g
        total["pore"] += p
    print(f"{'TOTAL':<18}{total['images']:>8}{'':>8}{total['guard']:>8}{total['pore']:>8}")
    if skipped_multi:
        print(f"note: {skipped_multi} annotations had multiple polygons; largest kept")
    if skipped_nopoly:
        print(f"WARNING: {skipped_nopoly} annotations had no usable polygon and were dropped")
    print(f"\nwrote {YOLO/'data.yaml'}")


if __name__ == "__main__":
    main()
