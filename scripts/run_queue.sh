#!/usr/bin/env bash
# Sequential run queue addressing the "unmatched training setups" limitation.
#
#  Phase A  Mask R-CNN at matched resolution (480/640) with flip + colour
#           augmentation and per-epoch checkpoints, seeds 42/43/44
#  Phase B  YOLOv8s-seg and YOLO26s-seg, seeds 42/43/44, fixed batch for
#           run-to-run comparability
#  Phase C  resolution sweep at 1280 to test the pore-resolution hypothesis
#  Phase D  validation-based checkpoint selection for every Mask R-CNN run
#
# Each step logs to logs/queue/ and failures do not abort the queue.

set -u
cd /home/rmedu2026/stomataAI
source ~/miniconda3/etc/profile.d/conda.sh
conda activate stomataai

DATA=/home/rmedu2026/stomataAI/baseline_data/data.yaml
LOGS=logs/queue
mkdir -p "$LOGS"

stamp () { date "+%Y-%m-%d %H:%M:%S"; }
note  () { echo "[$(stamp)] $*" | tee -a "$LOGS/queue.log"; }

note "queue started on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOGS/queue.log"

# ---------------------------------------------------------------- Phase A --
for SEED in 42 43 44; do
  OUT=runs/maskrcnn_m640_s${SEED}
  if [ -f "$OUT/final.pt" ]; then note "skip $OUT (done)"; continue; fi
  note "Phase A: Mask R-CNN 480/640 augmented, seed $SEED"
  python remote_mrcnn_train_v2.py \
      --seed "$SEED" --min-size 480 --max-size 640 \
      --epochs 30 --batch 2 --out "$OUT" \
      > "$LOGS/mrcnn_m640_s${SEED}.log" 2>&1 \
    && note "  ok -> $OUT" || note "  FAILED -> $OUT"
done

# ---------------------------------------------------------------- Phase B --
for SEED in 42 43 44; do
  for M in yolov8s-seg yolo26s-seg; do
    TAG="${M//-seg/}_640_s${SEED}"
    if [ -d "runs/seedstudy/$TAG/weights" ]; then note "skip $TAG (done)"; continue; fi
    note "Phase B: $M imgsz 640, seed $SEED"
    yolo segment train model="${M}.pt" data="$DATA" \
        epochs=100 patience=30 imgsz=640 batch=8 seed="$SEED" \
        optimizer=auto lr0=0.01 weight_decay=0.0005 amp=True workers=12 \
        project=/home/rmedu2026/stomataAI/runs/seedstudy name="$TAG" exist_ok=True plots=True \
        > "$LOGS/${TAG}.log" 2>&1 \
      && note "  ok -> runs/seedstudy/$TAG" || note "  FAILED -> $TAG"
  done
done

# ---------------------------------------------------------------- Phase C --
for M in yolov8s-seg yolo26s-seg; do
  TAG="${M//-seg/}_1280_s42"
  if [ -d "runs/seedstudy/$TAG/weights" ]; then note "skip $TAG (done)"; continue; fi
  note "Phase C: $M imgsz 1280, seed 42"
  yolo segment train model="${M}.pt" data="$DATA" \
      epochs=100 patience=30 imgsz=1280 batch=2 seed=42 \
      optimizer=auto lr0=0.01 weight_decay=0.0005 amp=True workers=12 \
      project=/home/rmedu2026/stomataAI/runs/seedstudy name="$TAG" exist_ok=True plots=True \
      > "$LOGS/${TAG}.log" 2>&1 \
    && note "  ok -> runs/seedstudy/$TAG" || note "  FAILED -> $TAG"
done

OUT=runs/maskrcnn_m1280_s42
if [ ! -f "$OUT/final.pt" ]; then
  note "Phase C: Mask R-CNN 960/1280 augmented, seed 42"
  python remote_mrcnn_train_v2.py \
      --seed 42 --min-size 960 --max-size 1280 \
      --epochs 30 --batch 1 --out "$OUT" \
      > "$LOGS/mrcnn_m1280_s42.log" 2>&1 \
    && note "  ok -> $OUT" || note "  FAILED -> $OUT"
fi

# ---------------------------------------------------------------- Phase D --
for RUN in runs/maskrcnn_m640_s42 runs/maskrcnn_m640_s43 runs/maskrcnn_m640_s44 \
           runs/maskrcnn_m1280_s42; do
  [ -d "$RUN" ] || continue
  case "$RUN" in
    *m1280*) MIN=960; MAX=1280 ;;
    *)       MIN=480; MAX=640  ;;
  esac
  for SPLIT in valid test; do
    note "Phase D: select/evaluate $RUN on $SPLIT"
    python remote_mrcnn_select.py \
        --ckpt-dir "$RUN" --split "$SPLIT" --out "$RUN/selection" \
        --min-size "$MIN" --max-size "$MAX" \
        > "$LOGS/select_$(basename "$RUN")_${SPLIT}.log" 2>&1 \
      && note "  ok" || note "  FAILED"
  done
done

note "queue finished"
