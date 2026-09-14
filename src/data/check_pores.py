"""Compare annotation density on newly-added vs previously-present images."""
import json
import os
from collections import Counter, defaultdict


def stem(f):
    f = os.path.basename(f)
    return f.split("_jpg.rf.")[0] if "_jpg.rf." in f else f


def load(root, t, s):
    p = os.path.join(root, t, s, "_annotations.coco.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    per_image = defaultdict(Counter)
    for a in d["annotations"]:
        per_image[a["image_id"]][cats[a["category_id"]]] += 1
    return {stem(i["file_name"]): per_image[i["id"]] for i in d["images"]}


print(f"{'split':<16}{'group':<26}{'imgs':>6}{'guard':>8}{'pore':>7}{'g:p':>8}")
print("-" * 72)
for t in ("control", "stress"):
    for s in ("train", "valid", "test"):
        new = load("new_export", t, s)
        old_any = {}
        for os_ in ("train", "valid", "test"):
            o = load("raw", t, os_)
            if o:
                old_any.update(o)
        if new is None:
            continue
        groups = {
            "already annotated before": [k for k in new if k in old_any],
            "NEW to this export": [k for k in new if k not in old_any],
        }
        for label, keys in groups.items():
            if not keys:
                continue
            g = sum(new[k]["guard"] for k in keys)
            p = sum(new[k]["pore"] for k in keys)
            ratio = f"{g / p:.1f}:1" if p else "  inf"
            print(f"{t + '/' + s:<16}{label:<26}{len(keys):>6}{g:>8}{p:>7}{ratio:>8}")

# images that previously had zero annotations -- did they gain any?
print()
print("Images that were background-only in the OLD export:")
print(f"{'':<16}{'':<26}{'imgs':>6}{'guard':>8}{'pore':>7}")
for t in ("control", "stress"):
    old_bg = set()
    for s in ("train", "valid", "test"):
        o = load("raw", t, s)
        if o:
            old_bg |= {k for k, v in o.items() if sum(v.values()) == 0}
    new_all = {}
    for s in ("train", "valid", "test"):
        n = load("new_export", t, s)
        if n:
            new_all.update(n)
    present = [k for k in old_bg if k in new_all]
    g = sum(new_all[k]["guard"] for k in present)
    p = sum(new_all[k]["pore"] for k in present)
    still_empty = sum(1 for k in present if sum(new_all[k].values()) == 0)
    print(f"{t:<16}{'now annotated':<26}{len(present):>6}{g:>8}{p:>7}"
          f"   still empty: {still_empty}")

# remaining background-only images in the new export
print()
print("Background-only images REMAINING in the new export:")
for t in ("control", "stress"):
    for s in ("train", "valid", "test"):
        n = load("new_export", t, s)
        if not n:
            continue
        empty = [k for k, v in n.items() if sum(v.values()) == 0]
        if empty:
            print(f"  {t}/{s}: {len(empty)}  e.g. {sorted(empty)[:4]}")
