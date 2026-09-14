"""Sanity-check a new Roboflow COCO export against the previous one."""
import json
import os
from collections import Counter

NEW = "new_export"
OLD = "raw"

print("=" * 78)
print("NEW EXPORT")
print("=" * 78)
hdr = ("split", "images", "annots", "guard", "pore", "bg-only")
print(f"{hdr[0]:<16}{hdr[1]:>8}{hdr[2]:>8}{hdr[3]:>8}{hdr[4]:>8}{hdr[5]:>9}")

tot = Counter()
cats_seen = set()
new_files = {}
for t in ("control", "stress"):
    for s in ("train", "valid", "test"):
        p = os.path.join(NEW, t, s, "_annotations.coco.json")
        if not os.path.exists(p):
            print(f"{t}/{s:<10} MISSING _annotations.coco.json")
            continue
        d = json.load(open(p))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        cats_seen.update(cats.values())
        per = Counter(cats[a["category_id"]] for a in d["annotations"])
        with_ann = {a["image_id"] for a in d["annotations"]}
        bg = len(d["images"]) - len(with_ann)
        print(
            f"{t + '/' + s:<16}{len(d['images']):>8}{len(d['annotations']):>8}"
            f"{per.get('guard', 0):>8}{per.get('pore', 0):>8}{bg:>9}"
        )
        tot["images"] += len(d["images"])
        tot["ann"] += len(d["annotations"])
        tot["guard"] += per.get("guard", 0)
        tot["pore"] += per.get("pore", 0)
        tot["bg"] += bg
        new_files[(t, s)] = {os.path.basename(i["file_name"]) for i in d["images"]}

print(f"{'TOTAL':<16}{tot['images']:>8}{tot['ann']:>8}{tot['guard']:>8}"
      f"{tot['pore']:>8}{tot['bg']:>9}")
print(f"categories present: {sorted(cats_seen)}")

# ---------------------------------------------------------------- old ------
print()
print("=" * 78)
print("OLD EXPORT (for comparison)")
print("=" * 78)
print(f"{hdr[0]:<16}{hdr[1]:>8}{hdr[2]:>8}{hdr[3]:>8}{hdr[4]:>8}{hdr[5]:>9}")
otot = Counter()
old_files = {}
for t in ("control", "stress"):
    for s in ("train", "valid", "test"):
        p = os.path.join(OLD, t, s, "_annotations.coco.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        per = Counter(cats[a["category_id"]] for a in d["annotations"])
        with_ann = {a["image_id"] for a in d["annotations"]}
        bg = len(d["images"]) - len(with_ann)
        print(
            f"{t + '/' + s:<16}{len(d['images']):>8}{len(d['annotations']):>8}"
            f"{per.get('guard', 0):>8}{per.get('pore', 0):>8}{bg:>9}"
        )
        otot["images"] += len(d["images"])
        otot["ann"] += len(d["annotations"])
        otot["guard"] += per.get("guard", 0)
        otot["pore"] += per.get("pore", 0)
        otot["bg"] += bg
        old_files[(t, s)] = {os.path.basename(i["file_name"]) for i in d["images"]}
print(f"{'TOTAL':<16}{otot['images']:>8}{otot['ann']:>8}{otot['guard']:>8}"
      f"{otot['pore']:>8}{otot['bg']:>9}")

# --------------------------------------------------- augmentation check ----
print()
print("=" * 78)
print("CHECKS")
print("=" * 78)


def stem(f):
    # Roboflow names: <orig>_jpg.rf.<hash>.jpg  -> take the part before _jpg.rf.
    return f.split("_jpg.rf.")[0] if "_jpg.rf." in f else f


for t in ("control", "stress"):
    allf = []
    for s in ("train", "valid", "test"):
        allf += list(new_files.get((t, s), []))
    stems = Counter(stem(f) for f in allf)
    dupes = {k: v for k, v in stems.items() if v > 1}
    print(f"{t}: {len(allf)} files, {len(stems)} unique source stems, "
          f"{len(dupes)} stems appearing more than once"
          f"{'  <-- augmented copies or duplicates' if dupes else '  (no augmentation)'}")
    if dupes:
        for k in list(dupes)[:5]:
            print(f"    {k} x{dupes[k]}")

# --------------------------------------------------- split membership ------
print()
for t in ("control", "stress"):
    for s in ("train", "valid", "test"):
        n = {stem(f) for f in new_files.get((t, s), set())}
        o = {stem(f) for f in old_files.get((t, s), set())}
        if not o:
            continue
        print(f"{t}/{s:<6} old={len(o):>4} new={len(n):>4}  kept={len(n & o):>4}  "
              f"added={len(n - o):>4}  removed={len(o - n):>4}")

# ------------------------------------------------------------ leakage -----
print()
for t in ("control", "stress"):
    tr = {stem(f) for f in new_files.get((t, "train"), set())}
    va = {stem(f) for f in new_files.get((t, "valid"), set())}
    te = {stem(f) for f in new_files.get((t, "test"), set())}
    print(f"{t}: train∩valid={len(tr & va)}  train∩test={len(tr & te)}  "
          f"valid∩test={len(va & te)}"
          f"{'   OK' if not (tr & va or tr & te or va & te) else '   <-- LEAKAGE'}")
