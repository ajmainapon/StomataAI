# StomataAI

Code for **"Benchmarking Deep-Learning Instance-Segmentation Models for Stomatal
Phenotyping under Drought Stress in Chickpea (*Cicer arietinum* L.)"**.

Three instance-segmentation architectures — YOLOv8s-seg, YOLO26s-seg and
Mask R-CNN (ResNet-50 FPN) — are trained on identical image-level splits and
evaluated on a common locked test set, with average precision reported
separately for guard-cell complexes and stomatal pores.

## Dataset

Nail-polish impressions of the abaxial epidermis of BARI Chola-5 chickpea,
imaged at 400× (2560 × 1920 px). Plants were grown under regular irrigation
(control) or with water withheld for 15 days (drought stress).

| | images | guard-cell instances | pore instances |
|---|---|---|---|
| train | 359 | — | — |
| validation | 99 | — | — |
| test (locked) | 48 | 453 | 223 |
| **total** | **506** | **4,816** | **1,789** |

600 images were acquired; 506 were present in the annotated export. Annotation
was done in Roboflow and exported in COCO instance-segmentation format.

**The imagery and annotations are not redistributed in this repository.** They
are excluded by `.gitignore` pending publication. Expected layout:

```
raw/{control,stress}/{train,valid,test}/    images + _annotations.coco.json
baseline_data/{images,labels}/{train,valid,test}/    YOLO-format mirror
baseline_data/data.yaml                     see configs/data.yaml
```

Trained weights are also excluded — individual Mask R-CNN checkpoints exceed
GitHub's 100 MB per-file limit.

## Layout

```
configs/data.yaml                     YOLO dataset config
src/
  train_maskrcnn_original.py          Mask R-CNN training as originally run
  train_maskrcnn_matched.py           matched resolution + augmentation, per-epoch
                                      checkpoints, configurable seed
  eval_maskrcnn.py                    COCOeval on the locked test set
  select_maskrcnn_checkpoint.py       checkpoint selection by validation mask AP
  evaluate_by_treatment.py            control / drought subset metrics
  collect_yolo_seeds.py               seed-variance summary from results.csv
scripts/run_queue.sh                  sequential run queue (phases A–D)
figures/
  make_figure7_panels.py              regenerates the qualitative comparison panels
  Figure1_architecture.tex            workflow figure (TikZ)
  Figure7_qualitative.tex             qualitative comparison montage (TikZ)
results/                              metrics and per-epoch logs (small files only)
```

## Environment

```bash
conda create -n stomataai python=3.12 -y
conda activate stomataai
pip install -r requirements.txt
```

Reported runs used torch 2.11.0+cu130, torchvision 0.26.0+cu130 and
ultralytics 8.4.104 on a single RTX 4060 (8 GB).

## Reproducing

YOLO models:

```bash
yolo segment train model=yolov8s-seg.pt data=configs/data.yaml \
    epochs=100 patience=30 imgsz=640 seed=42 \
    optimizer=auto lr0=0.01 weight_decay=0.0005 amp=True
```

Mask R-CNN, matched to the YOLO input resolution and augmentation:

```bash
python src/train_maskrcnn_matched.py --seed 42 \
    --min-size 480 --max-size 640 --epochs 30 --batch 2 --workers 12 \
    --out runs/maskrcnn_m640_s42
```

Select its checkpoint on validation, then score that checkpoint on the test set:

```bash
python src/select_maskrcnn_checkpoint.py --ckpt-dir runs/maskrcnn_m640_s42 \
    --split valid --out runs/maskrcnn_m640_s42/selection
python src/select_maskrcnn_checkpoint.py --ckpt-dir runs/maskrcnn_m640_s42 \
    --split test  --out runs/maskrcnn_m640_s42/selection
```

`scripts/run_queue.sh` runs the whole sequence and skips completed steps, so it
is safe to interrupt and restart.

> **Paths are hard-coded** to the machine the reported runs were executed on
> (`/home/rmedu2026/stomataAI`). The scripts are committed as they ran, for
> provenance. Edit the `ROOT` / `DATA` constants at the top of each file before
> running elsewhere.

## Methodological notes

Three asymmetries in the original benchmark were identified after the fact and
are addressed by the scripts here. They are worth understanding before comparing
any numbers across runs.

**Input resolution.** `train_maskrcnn_original.py` builds the model without a
`min_size`/`max_size` override, so torchvision's 800/1333 default applies. A
2560 × 1920 micrograph is resized to about 1067 × 800, whereas the YOLO models
see roughly 640 × 480 of content inside a 640-pixel letterbox — close to three
times the pixel area. `train_maskrcnn_matched.py` exposes these parameters.

**Augmentation.** The original Mask R-CNN dataset applied none, while the YOLO
runs used the framework defaults (mosaic, HSV jitter, horizontal flip).
`train_maskrcnn_matched.py` adds horizontal flip and colour jitter. Mosaic has
no torchvision equivalent and is not reproduced, so the pipelines are closer but
not identical.

**Checkpoint selection.** The original script created a validation loader and
never used it; the final-epoch checkpoint was evaluated. The YOLO models were
selected on best validation mask mAP@0.50:0.95. Applying that same rule to the
original Mask R-CNN checkpoints selects **epoch 10, not epoch 30** — validation
performance peaks there and declines thereafter — which raises locked-test mask
mAP@0.50:0.95 from **0.4284 to 0.4441**. See
`results/maskrcnn_original/selection_valid.json`.

**Input throughput.** The original training loop used `num_workers=4` on a
24-core host, leaving the GPU idle roughly 76% of the time (measured). Raising
it to 12 cut epoch wall time from 157 s to 101 s with no change to the model.

## Status

A seed-replication and resolution study is in progress; `results/queue.log`
records what has completed. Numbers under `results/` are per-run metrics, not
final aggregates, and the seed study is not yet complete — do not read the
current `yolo_seeds/` values as published results.

## Citation

Nashiha NJ, Afrin F, Arrafi MA, Rahman MA, Kashem MA, Bhuiyan MSE, Hossain MZ.
*Benchmarking deep-learning instance-segmentation models for stomatal
phenotyping under drought stress in chickpea (Cicer arietinum L.)* (manuscript
in preparation).

Correspondence: zabed@du.ac.bd, shifatearman@du.ac.bd
