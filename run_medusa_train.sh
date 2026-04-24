#!/bin/bash
# ============================================================
# Medusa end-to-end training pipeline:
#   1. Discover all HF-saved datasets under $DATASET_ROOT.
#   2. Launch `collect_medusa_data.py` in parallel, one shard per GPU,
#      mixing all discovered datasets round-robin (up to NUM_SAMPLES).
#   3. Wait for all shards to finish.
#   4. Run `train_medusa_v2.py` on the shards to produce the heads.
#
# Defaults target the jinchao remote and can be overridden via env vars.
#
# Usage:
#   bash run_medusa_train.sh                                 # defaults
#   GPUS="0 2 3"  NUM_SAMPLES=50000  bash run_medusa_train.sh
#   DATASET_ROOT=/my/path bash run_medusa_train.sh
#   DATASETS_INCLUDE="IMDA_PART1 IMDA_PART2" bash run_medusa_train.sh   # substring filter
# ============================================================
set -e
export PYTHONUNBUFFERED=1

# ── Paths / env ────────────────────────────────────────────────────────────────
PYTHON_PATH=${PYTHON_PATH:-/home/jinchao/miniconda3/envs/audiobench_quant/bin/python}
WORKDIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
MODEL=${MODEL:-/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B}
DATASET_ROOT=${DATASET_ROOT:-/home/jinchao/runtao/meralion_datasets/ASR}

# ── Knobs ──────────────────────────────────────────────────────────────────────
NUM_SAMPLES=${NUM_SAMPLES:-30000}                 # total across all datasets
NUM_SAMPLES_PER_DATASET=${NUM_SAMPLES_PER_DATASET:-0}   # 0 = no cap per dataset
START_IDX=${START_IDX:-30}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}

NUM_HEADS=${NUM_HEADS:-4}
NUM_LAYERS=${NUM_LAYERS:-1}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}
LR=${LR:-1e-3}
EPOCHS=${EPOCHS:-3}

SHARDS_DIR=${SHARDS_DIR:-$WORKDIR/medusa_shards}
HEADS_OUT=${HEADS_OUT:-$WORKDIR/medusa_heads_best.pt}

FORCE=${FORCE:-0}                 # 1 = re-collect shards even if present

# ── GPU selection ─────────────────────────────────────────────────────────────
# GPUS defaults to auto: pick the top-N GPUs by free VRAM where N comes
# from NUM_SHARDS (default 3).  Override GPUS="0 2 3" to pin.
NUM_SHARDS=${NUM_SHARDS:-3}
if [ -z "$GPUS" ]; then
    if ! command -v nvidia-smi >/dev/null; then
        echo "ERROR: nvidia-smi not found and GPUS unset"; exit 1
    fi
    GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
           | sort -t, -k2 -n -r | head -n "$NUM_SHARDS" | cut -d, -f1 | tr '\n' ' ')
    GPUS=$(echo $GPUS)   # trim
fi
# Read into an array
read -r -a GPU_ARR <<< "$GPUS"
NUM_SHARDS=${#GPU_ARR[@]}
if [ "$NUM_SHARDS" -lt 1 ]; then
    echo "ERROR: no GPUs available"; exit 1
fi

# ── Discover datasets ─────────────────────────────────────────────────────────
if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: DATASET_ROOT not a directory: $DATASET_ROOT"; exit 1
fi
ALL_DATASETS=()
for d in "$DATASET_ROOT"/*/; do
    [ -d "$d" ] || continue
    # A HuggingFace save_to_disk artefact contains state.json or dataset_info.json.
    if [ -f "$d/state.json" ] || [ -f "$d/dataset_info.json" ]; then
        ALL_DATASETS+=("${d%/}")
    fi
done

# Optional substring filter
DATASETS=()
if [ -n "$DATASETS_INCLUDE" ]; then
    for d in "${ALL_DATASETS[@]}"; do
        match=0
        for key in $DATASETS_INCLUDE; do
            if [[ "$d" == *"$key"* ]]; then match=1; break; fi
        done
        [ $match -eq 1 ] && DATASETS+=("$d")
    done
else
    DATASETS=("${ALL_DATASETS[@]}")
fi

if [ "${#DATASETS[@]}" -lt 1 ]; then
    echo "ERROR: no HF datasets found under $DATASET_ROOT"
    echo "       looked for subdirs containing state.json / dataset_info.json"
    exit 1
fi

# ── Print config ──────────────────────────────────────────────────────────────
echo "========================================"
echo "  Config"
echo "========================================"
echo "  MODEL               : $MODEL"
echo "  DATASET_ROOT        : $DATASET_ROOT"
echo "  datasets found      : ${#DATASETS[@]}"
for d in "${DATASETS[@]}"; do echo "    - ${d##*/}"; done
echo "  NUM_SAMPLES (total) : $NUM_SAMPLES"
echo "  per-dataset cap     : $NUM_SAMPLES_PER_DATASET  (0 = none)"
echo "  GPUs                : ${GPU_ARR[*]}  (${NUM_SHARDS} shards)"
echo "  SHARDS_DIR          : $SHARDS_DIR"
echo "  HEADS_OUT           : $HEADS_OUT"
echo "  FORCE               : $FORCE"

mkdir -p "$SHARDS_DIR"

# ════════════════════════════════════════════════════════════════════════════
echo
echo "========================================"
echo "  Step 1/2: collect shards ($NUM_SHARDS × GPU)"
echo "========================================"

PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu="${GPU_ARR[$i]}"
    shard_out="$SHARDS_DIR/medusa_data_shard_${i}.pt"
    log="$SHARDS_DIR/shard_${i}.log"
    if [ "$FORCE" != "1" ] && [ -s "$shard_out" ]; then
        echo "  [skip] $shard_out exists"; continue
    fi
    echo "  launching shard $i on GPU $gpu  → $shard_out"
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON_PATH" -u "$WORKDIR/collect_medusa_data.py" \
        --model "$MODEL" \
        --datasets "${DATASETS[@]}" \
        --num_samples "$NUM_SAMPLES" \
        --num_samples_per_dataset "$NUM_SAMPLES_PER_DATASET" \
        --start_idx "$START_IDX" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --shard_id "$i" --num_shards "$NUM_SHARDS" \
        --output_shard "$shard_out" \
        > "$log" 2>&1 &
    PIDS+=($!)
done

if [ "${#PIDS[@]}" -gt 0 ]; then
    echo "  waiting for ${#PIDS[@]} shard process(es) …"
    FAILED=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "  pid $pid FAILED"
            FAILED=$((FAILED + 1))
        fi
    done
    if [ "$FAILED" -gt 0 ]; then
        echo "$FAILED shard(s) failed.  See $SHARDS_DIR/shard_*.log"
        exit 1
    fi
    echo "  all shards done."
fi

# ════════════════════════════════════════════════════════════════════════════
echo
echo "========================================"
echo "  Step 2/2: train Medusa heads"
echo "========================================"

SHARD_FILES=("$SHARDS_DIR"/medusa_data_shard_*.pt)
if [ ! -s "${SHARD_FILES[0]}" ]; then
    echo "ERROR: no shard files in $SHARDS_DIR"; exit 1
fi

# Pick one GPU for training (first of the GPU_ARR by default).
TRAIN_GPU=${TRAIN_GPU:-${GPU_ARR[0]}}
echo "  training on GPU $TRAIN_GPU  → $HEADS_OUT"

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$PYTHON_PATH" -u "$WORKDIR/train_medusa_v2.py" \
    --model "$MODEL" \
    --data_shards "${SHARD_FILES[@]}" \
    --num_heads "$NUM_HEADS" --num_layers "$NUM_LAYERS" \
    --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" \
    --lr "$LR" --epochs "$EPOCHS" \
    --output "${HEADS_OUT%.pt}_final.pt" \
    --output_best "$HEADS_OUT" \
    2>&1 | tee "$WORKDIR/medusa_train.log"

echo
echo "========================================"
echo "  Done"
echo "========================================"
echo "  heads (best val) → $HEADS_OUT"
echo "  training log     → $WORKDIR/medusa_train.log"
