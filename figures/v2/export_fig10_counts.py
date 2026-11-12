"""Save the per-image and per-plant counts behind Figure 10 / the agreement table.

make_fig10.py computes these counts but only plots them. This script reuses its
exact functions (same model, confidence 0.25, plant assignment and K = 7.1107)
and writes four CSVs in the column format read by SAT_v2/source_figures/make_figures.py.

Run on the training server from the repo root:
    python figures/v2/export_fig10_counts.py
Output: /home/rmedu2026/stomataAI/v2/figures/fig10_counts/*.csv
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_fig10 as m  # noqa: E402

K = 7.1107
OUT = f"{m.OUT}/fig10_counts"
NAMES = {"control": "control", "stress": "drought"}


def main():
    os.makedirs(OUT, exist_ok=True)
    for treat, name in NAMES.items():
        pairs = m.manual_counts(treat)
        files = [f for f, _ in pairs]
        man = [c for _, c in pairs]
        ai = m.ai_counts(treat, files)

        with open(f"{OUT}/fig10_per_image_{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["annotated_count", "predicted_count", "file", "plant"])
            for f, a, p in zip(files, man, ai):
                w.writerow([a, p, f, m.plant_of(f, treat)])

        groups = defaultdict(lambda: [[], []])
        for f, a, p in zip(files, man, ai):
            g = groups[m.plant_of(f, treat)]
            g[0].append(a); g[1].append(p)
        with open(f"{OUT}/fig10_per_plant_{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["annotated_density_relative", "predicted_density_relative", "plant", "n_images"])
            for pl in sorted(groups):
                a, p = groups[pl]
                w.writerow([round(sum(a) / len(a) * K, 4), round(sum(p) / len(p) * K, 4), pl, len(a)])

        print(f"{name:<8} images {len(man):3d}  plants {len(groups):3d}  "
              f"annotated {sum(man)}  predicted {sum(ai)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
