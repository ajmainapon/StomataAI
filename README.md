# StomataAI

Code for **"Stomatal Phenotyping in Chickpea (*Cicer arietinum* L.) under
Drought Stress Using Deep-Learning Instance-Segmentation Models"** (manuscript
under review).

Three instance-segmentation architectures — YOLOv8s-seg, YOLO26s-seg and
Mask R-CNN (ResNet-50 FPN) — are trained three times each on identical
image-level splits and evaluated on a common held-out test set, with average
precision reported separately for guard-cell complexes and stomatal pores.

## Dataset (v2)

Nail-polish impressions of the abaxial epidermis of chickpea variety
BARI Chola-5, imaged at 400× (2560 × 1920 px). Plants were grown under a
well-watered control or with water withheld for 15 days (drought stress).

| Treatment | Split | Images | Guard-cell | Pore | Background-only |
|---|---|---:|---:|---:|---:|
| Control | train / valid / test | 223 / 48 / 47 | 2,740 / 501 / 596 | 858 / 244 / 204 | 3 / 9 / 2 |
| Drought | train / valid / test | 136 / 29 / 29 | 1,265 / 254 / 275 | 342 / 68 / 77 | 0 / 0 / 0 |
| **Total** | **359 / 77 / 76** | **512** | **5,631** | **1,793** | **14** |

600 images were acquired; 512 are present in the annotated Roboflow export
(COCO instance segmentation). The export also declares a `stomata` category
with zero annotations, which is excluded from macro AP.

**Imagery, annotations and trained weights are not redistributed here.** They
are excluded by `.gitignore`; individual Mask R-CNN checkpoints also exceed
GitHub's 100 MB per-file limit.

## Results reported in the manuscript

Pooled held-out test set (76 images, 1,152 instances), mean ± SD over seeds
42, 43 and 44. Source: [`results/v2/v2_tables_seeds.json`](results/v2/v2_tables_seeds.json).

| Model | Box mAP@0.50 | Box mAP@0.50:0.95 | Mask mAP@0.50 | Mask mAP@0.50:0.95 |
|---|---|---|---|---|
| YOLOv8s-seg | 0.640 ± 0.023 | 0.472 ± 0.005 | 0.583 ± 0.014 | 0.381 ± 0.010 |
| YOLO26s-seg | **0.649 ± 0.005** | **0.480 ± 0.009** | **0.605 ± 0.011** | 0.393 ± 0.002 |
| Mask R-CNN | 0.617 ± 0.011 | 0.444 ± 0.007 | 0.572 ± 0.009 | **0.440 ± 0.008** |

Class-specific mask AP@0.50:0.95 ranges from 0.701 to 0.817 for guard-cell
complexes and from 0.045 to 0.172 for pores across models and treatments:
pore segmentation, not the choice of architecture, is the limiting factor.
The JSON holds every per-seed, per-treatment value behind Tables 2 and 3.

## Layout

```
configs/
  data_v2.yaml                     YOLO dataset config (v2, used in the paper)
  data.yaml                        YOLO dataset config (v1, superseded)
src/
  data/build_v2_dataset.py         builds COCO + YOLO views from the Roboflow export
  data/check_export.py             sanity-checks a new export against the previous one
  data/check_pores.py              annotation density on added vs existing images
  train_maskrcnn_matched.py        Mask R-CNN training (resolution/augmentation flags,
                                   per-epoch checkpoints, configurable seed)
  select_maskrcnn_checkpoint.py    selects each Mask R-CNN run on validation mask AP
  analysis/tables_v2_seeds.py      Tables 2-3: per-treatment AP, mean ± SD over seeds
  train_maskrcnn_original.py       v1 Mask R-CNN training, as originally run
  eval_maskrcnn.py                 v1 COCOeval on the test set
  evaluate_by_treatment.py         v1 control / drought subset metrics
  collect_yolo_seeds.py            v1 seed-variance summary
scripts/
  run_queue_v2.sh                  the full v2 run queue (phases A0-D)
  run_queue_v1.sh                  v1 run queue (superseded)
figures/
  v2/make_fig4_5_6_8.py            training curves, confusion matrices, PR/F1, AP by treatment
  v2/make_fig7.py                  qualitative comparison montage
  v2/make_fig10.py                 manual vs AI counts (per image) and density (per plant)
  Figure1_architecture.tex         workflow figure (TikZ)
  Figure7_qualitative.tex, make_figure7_panels.py   v1 qualitative montage
results/
  v2/                              results reported in the manuscript
  v1/                              superseded v1 results, kept for provenance
```

## Environment

```bash
conda create -n stomataai python=3.12 -y
conda activate stomataai
pip install -r requirements.txt
```

Reported runs used torch 2.11.0+cu130, torchvision 0.26.0+cu130 and
ultralytics 8.4.104 on a single RTX 4060 (8 GB).

## Reproducing the manuscript results

1. Build the dataset views from the Roboflow export:

   ```bash
   python src/data/build_v2_dataset.py
   ```

2. Train and select every model. The queue skips completed steps, so it is
   safe to interrupt and restart:

   ```bash
   bash scripts/run_queue_v2.sh
   ```

   The runs reported in the paper are Phase B (`yolov8s_640_s{42,43,44}`,
   `yolo26s_640_s{42,43,44}`; 640 px, up to 100 epochs, patience 30, best
   validation mask mAP@0.50:0.95) and Phase A0 (`maskrcnn_orig800_s{42,43,44}`;
   800/1333 px, no augmentation, 30 epochs, AdamW lr 2e-4, checkpoint selected
   on validation mask mAP@0.50:0.95 in Phase D). Phases A and C
   (matched-resolution and 1280 px runs) were exploratory and are not
   reported.

3. Tables and figures:

   ```bash
   python src/analysis/tables_v2_seeds.py --device 0
   python figures/v2/make_fig4_5_6_8.py
   python figures/v2/make_fig7.py
   python figures/v2/make_fig10.py
   ```

   Training curves, confusion matrices, PR/F1 curves, the qualitative montage
   and the count comparison are per-run and use seed 42; Figure 8 and Tables
   2-3 aggregate all three seeds.

> **Paths are hard-coded** to the machine the reported runs were executed on
> (`/home/rmedu2026/stomataAI`). Scripts are committed as they ran, for
> provenance. Edit the path constants at the top of each file, and `path:` in
> `configs/data_v2.yaml`, before running elsewhere.

## Methodological notes

**Checkpoint selection.** Every model, in every run, is evaluated at the
checkpoint with the best validation mask mAP@0.50:0.95. For Mask R-CNN this
lands at epochs 4, 14 and 17 of 30 for the three seeds, well before the end of
training, so final-epoch evaluation would understate it.

**Seeds.** Each architecture is trained with seeds 42, 43 and 44. Differences
smaller than the reported standard deviations should not be read as
architectural differences; on pooled mask mAP@0.50:0.95 only the Mask R-CNN
advantage clearly exceeds run-to-run variation.

**Counting.** Figure 10 compares annotated and predicted guard-cell counts on
the test set (seed-42 YOLO26s-seg, confidence 0.25): R² = 0.437 (control) and
0.688 (drought) per image, 0.774 and 0.734 per plant. The model overcounts in
both treatments (739 vs 596 and 393 vs 275), so these are associations, not
evidence of interchangeability with manual counting.

## v1 (superseded)

An earlier benchmark used a 506-image export (359 / 99 / 48 split), a single
training run per model and final-epoch Mask R-CNN evaluation. Its scripts and
results are kept under `results/v1/`, `scripts/run_queue_v1.sh` and the v1
entries in `src/` for provenance only. **None of its numbers appear in the
manuscript.**

## Not yet in this repository

Per-run training logs (`results.csv`), Mask R-CNN checkpoint-selection records
and the Figure 10 source data for the v2 runs are still on the training host
and will be added under `results/v2/`.

## Citation

Nashiha NJ, Apon AI, Afrin F, Arrafi MA, Hasan MM, Rahman MA, Kashem MA,
Arman SE, Hossain MZ. *Stomatal phenotyping in chickpea (Cicer arietinum L.)
under drought stress using deep-learning instance-segmentation models.*
Manuscript under review.

Correspondence: zabed@du.ac.bd, shifatearman@du.ac.bd
