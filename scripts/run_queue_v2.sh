#!/usr/bin/env bash
# v2 run queue -- every experiment repeated on the re-annotated dataset.
#
#  Phase A0  Mask R-CNN in the ORIGINAL published configuration (800/1333, no
#            augmentation) -- the configuration reported in the manuscript;
#            its checkpoint is chosen on validation in Phase D
#  Phase A   Mask R-CNN at matched resolution 480/640 with augmentation, 3 seeds
#  Phase B   YOLOv8s-seg and YOLO26s-seg at 640, 3 seeds each
#  Phase C   resolution sweep at 1280 for all three architectures
#  Phase D   validation-based checkpoint selection for every Mask R-CNN run
#
# Everything writes under v2/. v1 artefacts are never touched.
# Steps are skipped if already complete, so this is safe to interrupt.

set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"   # scripts live in this repo; runs write under the data root
cd /home/rmedu2026/stomataAI
source ~/miniconda3/etc/profile.d/conda.sh
conda activate stomataai

export STOMATA_RAW=/home/rmedu2026/stomataAI/v2/data/raw
DATA=/home/rmedu2026/stomataAI/v2/data/yolo/data.yaml
RUNS=/home/rmedu2026/stomataAI/v2/runs
LOGS=v2/logs
mkdir -p "$LOGS" "$RUNS"

stamp () { date "+%Y-%m-%d %H:%M:%S"; }
note  () { echo "[$(stamp)] $*" | tee -a "$LOGS/queue.log"; }

note "v2 queue started on $(hostname)"
note "raw=$STOMATA_RAW"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOGS/queue.log"

# --------------------------------------------------------------- Phase A0 --
OUT=$RUNS/maskrcnn_orig800_s42
if [ ! -f "$OUT/final.pt" ]; then
  note "Phase A0: Mask R-CNN 800/1333, no augmentation, seed 42 (published config)"
  python "$REPO/src/train_maskrcnn_matched.py" \
      --seed 42 --min-size 800 --max-size 1333 --no-augment \
      --epochs 30 --batch 2 --workers 12 --out "$OUT" \
      > "$LOGS/mrcnn_orig800_s42.log" 2>&1 \
    && note "  ok" || note "  FAILED"
else note "skip $(basename $OUT) (done)"; fi

# ---------------------------------------------------------------- Phase A --
for SEED in 42 43 44; do
  OUT=$RUNS/maskrcnn_m640_s${SEED}
  if [ -f "$OUT/final.pt" ]; then note "skip $(basename $OUT) (done)"; continue; fi
  note "Phase A: Mask R-CNN 480/640 augmented, seed $SEED"
  python "$REPO/src/train_maskrcnn_matched.py" \
      --seed "$SEED" --min-size 480 --max-size 640 \
      --epochs 30 --batch 2 --workers 12 --out "$OUT" \
      > "$LOGS/mrcnn_m640_s${SEED}.log" 2>&1 \
    && note "  ok" || note "  FAILED"
done

# ---------------------------------------------------------------- Phase B --
for SEED in 42 43 44; do
  for M in yolov8s-seg yolo26s-seg; do
    TAG="${M//-seg/}_640_s${SEED}"
    if [ -d "$RUNS/$TAG/weights" ]; then note "skip $TAG (done)"; continue; fi
    note "Phase B: $M imgsz 640, seed $SEED"
    yolo segment train model="${M}.pt" data="$DATA" \
        epochs=100 patience=30 imgsz=640 batch=8 seed="$SEED" \
        optimizer=auto lr0=0.01 weight_decay=0.0005 amp=True workers=12 \
        project="$RUNS" name="$TAG" exist_ok=True plots=True \
        > "$LOGS/${TAG}.log" 2>&1 \
      && note "  ok" || note "  FAILED -> $TAG"
  done
done

# ---------------------------------------------------------------- Phase C --
for M in yolov8s-seg yolo26s-seg; do
  TAG="${M//-seg/}_1280_s42"
  if [ -d "$RUNS/$TAG/weights" ]; then note "skip $TAG (done)"; continue; fi
  note "Phase C: $M imgsz 1280, seed 42"
  yolo segment train model="${M}.pt" data="$DATA" \
      epochs=100 patience=30 imgsz=1280 batch=2 seed=42 \
      optimizer=auto lr0=0.01 weight_decay=0.0005 amp=True workers=12 \
      project="$RUNS" name="$TAG" exist_ok=True plots=True \
      > "$LOGS/${TAG}.log" 2>&1 \
    && note "  ok" || note "  FAILED -> $TAG"
done

OUT=$RUNS/maskrcnn_m1280_s42
if [ ! -f "$OUT/final.pt" ]; then
  note "Phase C: Mask R-CNN 960/1280 augmented, seed 42"
  python "$REPO/src/train_maskrcnn_matched.py" \
      --seed 42 --min-size 960 --max-size 1280 \
      --epochs 30 --batch 1 --workers 12 --out "$OUT" \
      > "$LOGS/mrcnn_m1280_s42.log" 2>&1 \
    && note "  ok" || note "  FAILED"
else note "skip $(basename $OUT) (done)"; fi

# ---------------------------------------------------------------- Phase D --
for RUN in "$RUNS"/maskrcnn_*; do
  [ -d "$RUN" ] || continue
  case "$RUN" in
    *m1280*)   MIN=960; MAX=1280 ;;
    *orig800*) MIN=800; MAX=1333 ;;
    *)         MIN=480; MAX=640  ;;
  esac
  for SPLIT in valid test; do
    [ -f "$RUN/selection/selection_${SPLIT}.json" ] && { note "skip select $(basename $RUN)/$SPLIT"; continue; }
    note "Phase D: select/evaluate $(basename "$RUN") on $SPLIT"
    python "$REPO/src/select_maskrcnn_checkpoint.py" \
        --ckpt-dir "$RUN" --split "$SPLIT" --out "$RUN/selection" \
        --min-size "$MIN" --max-size "$MAX" \
        > "$LOGS/select_$(basename "$RUN")_${SPLIT}.log" 2>&1 \
      && note "  ok" || note "  FAILED"
  done
done

note "v2 queue finished"
