#!/bin/bash
# ============================================================
# EAGLE draft-model end-to-end pipeline:
#   1. (optional) auto-discover datasets + collect shards via
#      collect_medusa_data.py if $SHARDS_DIR has none (reuses
#      medusa_data_shard_*.pt).
#   2. Train EAGLE on those shards.
#   3. Bench EAGLE vs bf16 baseline on an eval dataset.
#
# Matches the defaults used by the other bench scripts on the jinchao
# remote; every variable is env-overridable.
#
# Usage:
#   bash run_eagle_train.sh                                  # defaults
#   GPUS="0 2 3"  NUM_SAMPLES=30000  bash run_eagle_train.sh
#   EPOCHS=5 LR=2e-4 bash run_eagle_train.sh
#   SKIP_COLLECT=1 bash run_eagle_train.sh                   # reuse shards only
#   SKIP_BENCH=1 bash run_eagle_train.sh                     # train only
# ============================================================
set -e
export PYTHONUNBUFFERED=1

PYTHON_PATH=${PYTHON_PATH:-/home/jinchao/miniconda3/envs/audiobench_quant/bin/python}
WORKDIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
MODEL=${MODEL:-/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B}
DATASET_ROOT=${DATASET_ROOT:-/home/jinchao/runtao/meralion_datasets/ASR}
BENCH_DATASET=${BENCH_DATASET:-$DATASET_ROOT/imda_part1_asr_test}

# ── Shard collection knobs ─────────────────────────────────────────────────────
SHARDS_DIR=${SHARDS_DIR:-$WORKDIR/medusa_shards}
NUM_SAMPLES=${NUM_SAMPLES:-30000}                       # total across datasets
NUM_SAMPLES_PER_DATASET=${NUM_SAMPLES_PER_DATASET:-0}
START_IDX=${START_IDX:-30}
MAX_NEW_TOKENS_COLLECT=${MAX_NEW_TOKENS_COLLECT:-128}

# ── EAGLE training knobs ──────────────────────────────────────────────────────
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-3e-4}
WARMUP_STEPS=${WARMUP_STEPS:-200}
EPOCHS=${EPOCHS:-3}
HIDDEN_LOSS_ALPHA=${HIDDEN_LOSS_ALPHA:-0.5}
SCHED_SAMPLING_MAX=${SCHED_SAMPLING_MAX:-0.5}
UNROLL_DEPTH=${UNROLL_DEPTH:-1}    # 1 = teacher-force + sched sampling.
                                   # >=2 = multi-step autoregressive unroll
                                   # (better K scaling, D× train cost)
NUM_LAYERS=${NUM_LAYERS:-1}        # EAGLE decoder layers (try 2 for capacity).
EVAL_EVERY=${EVAL_EVERY:-300}
LOG_EVERY=${LOG_EVERY:-50}

EAGLE_OUT=${EAGLE_OUT:-$WORKDIR/eagle_best.pt}

# ── Inference / bench knobs ────────────────────────────────────────────────────
K=${K:-4}
TREE_B=${TREE_B:-0}              # 0 = chain (FA2); >=2 = tree mode (eager attn)
BENCH_NUM_SAMPLES=${BENCH_NUM_SAMPLES:-20}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
BASELINE_OUT=${BASELINE_OUT:-$WORKDIR/gpu_bf16_nospec.json}
EAGLE_BENCH_OUT=${EAGLE_BENCH_OUT:-$WORKDIR/gpu_eagle_bench.json}

# ── Pipeline controls ──────────────────────────────────────────────────────────
FORCE=${FORCE:-0}                # re-collect + re-train + re-bench (everything)
FORCE_BENCH=${FORCE_BENCH:-0}    # just re-bench (re-runs both baseline and EAGLE);
                                 # useful for sweeping K without retraining
SKIP_COLLECT=${SKIP_COLLECT:-0}  # skip step 1
SKIP_TRAIN=${SKIP_TRAIN:-0}      # skip step 2 (reuse existing $EAGLE_OUT)
SKIP_BENCH=${SKIP_BENCH:-0}      # skip step 3

# ── GPU selection (auto-pick top-N by free VRAM) ──────────────────────────────
NUM_SHARDS=${NUM_SHARDS:-3}
if [ -z "$GPUS" ]; then
    if ! command -v nvidia-smi >/dev/null; then
        echo "ERROR: GPUS unset and nvidia-smi not found"; exit 1
    fi
    GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
           | sort -t, -k2 -n -r | head -n "$NUM_SHARDS" | cut -d, -f1 | tr '\n' ' ')
    GPUS=$(echo $GPUS)
fi
read -r -a GPU_ARR <<< "$GPUS"
NUM_SHARDS=${#GPU_ARR[@]}
[ "$NUM_SHARDS" -lt 1 ] && { echo "ERROR: no GPUs"; exit 1; }
TRAIN_GPU=${TRAIN_GPU:-${GPU_ARR[0]}}
BENCH_GPU=${BENCH_GPU:-${GPU_ARR[0]}}

echo "========================================"
echo "  Config"
echo "========================================"
printf '  %-22s %s\n' "MODEL:"            "$MODEL"
printf '  %-22s %s\n' "DATASET_ROOT:"     "$DATASET_ROOT"
printf '  %-22s %s\n' "BENCH_DATASET:"    "$BENCH_DATASET"
printf '  %-22s %s\n' "SHARDS_DIR:"       "$SHARDS_DIR"
printf '  %-22s %s\n' "EAGLE_OUT:"        "$EAGLE_OUT"
printf '  %-22s %s\n' "collect GPUs:"     "${GPU_ARR[*]}"
printf '  %-22s %s\n' "train GPU:"        "$TRAIN_GPU"
printf '  %-22s %s\n' "bench GPU:"        "$BENCH_GPU"
printf '  %-22s %s\n' "EPOCHS / LR:"      "$EPOCHS / $LR  (hidden_alpha=$HIDDEN_LOSS_ALPHA)"
printf '  %-22s %s\n' "K (draft depth):"  "$K"
printf '  %-22s %s\n' "FORCE:"            "$FORCE"

mkdir -p "$SHARDS_DIR"

# ════════════════════════════════════════════════════════════════════════════
# Step 1: collect shards (reuse medusa data — same format works for EAGLE)
# ════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_COLLECT" = "1" ]; then
    echo
    echo "[skip] collect shards (SKIP_COLLECT=1)"
else
    echo
    echo "========================================"
    echo "  Step 1/3: collect shards"
    echo "========================================"
    # Only collect if no shards exist (or FORCE).
    existing=$(ls "$SHARDS_DIR"/medusa_data_shard_*.pt 2>/dev/null | wc -l)
    if [ "$FORCE" != "1" ] && [ "$existing" -gt 0 ]; then
        echo "  [skip] $existing shard(s) already present in $SHARDS_DIR"
    else
        # Discover HF save_to_disk artefacts under DATASET_ROOT (recurse up to
        # depth 3 — handles both flat layouts and nested {corpus}/{part}/ ones).
        mapfile -t DATASETS < <(
            find -L "$DATASET_ROOT" -mindepth 1 -maxdepth 3 -type d \
                \( -exec test -f "{}/state.json" \; -o \
                   -exec test -f "{}/dataset_info.json" \; \) \
                -print 2>/dev/null | sort -u
        )
        if [ "${#DATASETS[@]}" -lt 1 ]; then
            echo "ERROR: no HF datasets under $DATASET_ROOT"; exit 1
        fi
        echo "  found ${#DATASETS[@]} datasets under $DATASET_ROOT:"
        for d in "${DATASETS[@]}"; do echo "    - $d"; done
        echo "  launching $NUM_SHARDS shards on GPUs ${GPU_ARR[*]}"
        PIDS=()
        for i in $(seq 0 $((NUM_SHARDS - 1))); do
            gpu="${GPU_ARR[$i]}"
            out="$SHARDS_DIR/medusa_data_shard_${i}.pt"
            log="$SHARDS_DIR/shard_${i}.log"
            if [ "$FORCE" != "1" ] && [ -s "$out" ]; then
                echo "  [skip] $out"; continue
            fi
            echo "  launch shard $i on GPU $gpu → $out"
            CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON_PATH" -u "$WORKDIR/collect_medusa_data.py" \
                --model "$MODEL" --datasets "${DATASETS[@]}" \
                --num_samples "$NUM_SAMPLES" \
                --num_samples_per_dataset "$NUM_SAMPLES_PER_DATASET" \
                --start_idx "$START_IDX" --max_new_tokens "$MAX_NEW_TOKENS_COLLECT" \
                --shard_id "$i" --num_shards "$NUM_SHARDS" \
                --output_shard "$out" > "$log" 2>&1 &
            PIDS+=($!)
        done
        if [ "${#PIDS[@]}" -gt 0 ]; then
            FAILED=0
            for pid in "${PIDS[@]}"; do
                wait "$pid" || FAILED=$((FAILED+1))
            done
            [ "$FAILED" -gt 0 ] && { echo "$FAILED shard(s) failed"; exit 1; }
        fi
    fi
fi

# ════════════════════════════════════════════════════════════════════════════
# Step 2: train EAGLE
# ════════════════════════════════════════════════════════════════════════════
echo
echo "========================================"
echo "  Step 2/3: train EAGLE"
echo "========================================"

SHARD_FILES=("$SHARDS_DIR"/medusa_data_shard_*.pt)
if [ ! -s "${SHARD_FILES[0]}" ]; then
    echo "ERROR: no shard files in $SHARDS_DIR"; exit 1
fi

if [ "$SKIP_TRAIN" = "1" ]; then
    echo "  [skip] SKIP_TRAIN=1 (reusing $EAGLE_OUT if present)"
elif [ "$FORCE" != "1" ] && [ -s "$EAGLE_OUT" ]; then
    echo "  [skip] $EAGLE_OUT already exists (FORCE=1 to retrain)"
else
    echo "  training on GPU $TRAIN_GPU → $EAGLE_OUT"
    export TOKENIZERS_PARALLELISM=false
    CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$PYTHON_PATH" -u "$WORKDIR/train_eagle.py" \
        --model "$MODEL" \
        --data_shards "${SHARD_FILES[@]}" \
        --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" \
        --lr "$LR" --warmup_steps "$WARMUP_STEPS" --epochs "$EPOCHS" \
        --hidden_loss_alpha "$HIDDEN_LOSS_ALPHA" \
        --sched_sampling_max "$SCHED_SAMPLING_MAX" \
        --unroll_depth "$UNROLL_DEPTH" \
        --num_layers "$NUM_LAYERS" \
        --eval_every "$EVAL_EVERY" --log_every "$LOG_EVERY" \
        --output "${EAGLE_OUT%.pt}_final.pt" \
        --output_best "$EAGLE_OUT" \
        2>&1 | tee "$WORKDIR/eagle_train.log"
fi

# ════════════════════════════════════════════════════════════════════════════
# Step 3: bench EAGLE vs bf16 baseline
# ════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_BENCH" = "1" ]; then
    echo
    echo "[skip] bench (SKIP_BENCH=1)"
    exit 0
fi

echo
echo "========================================"
echo "  Step 3/3: bench EAGLE vs bf16 baseline"
echo "========================================"
echo "  bench GPU : $BENCH_GPU"
echo "  dataset   : $BENCH_DATASET"
echo "  K / N     : $K / $BENCH_NUM_SAMPLES"

# BF16 baseline
if [ "$FORCE" != "1" ] && [ "$FORCE_BENCH" != "1" ] && [ -s "$BASELINE_OUT" ]; then
    echo "  [skip] baseline: $BASELINE_OUT exists  (FORCE_BENCH=1 to re-run)"
else
    echo "  [1/2] bf16 baseline …"
    CUDA_VISIBLE_DEVICES="$BENCH_GPU" "$PYTHON_PATH" -u "$WORKDIR/infer_gpu.py" \
        --model "$MODEL" --dataset "$BENCH_DATASET" \
        --num_samples "$BENCH_NUM_SAMPLES" --max_new_tokens "$MAX_NEW_TOKENS" \
        --quant bf16 --output "$BASELINE_OUT" \
        | tee "${BASELINE_OUT%.json}.log"
fi

# EAGLE
if [ "$FORCE" != "1" ] && [ "$FORCE_BENCH" != "1" ] && [ -s "$EAGLE_BENCH_OUT" ]; then
    echo "  [skip] eagle bench: $EAGLE_BENCH_OUT exists  (FORCE_BENCH=1 to re-run)"
else
    if [ "$TREE_B" != "0" ]; then
        echo "  [2/2] EAGLE-tree (K=$K, B=$TREE_B) …"
        CUDA_VISIBLE_DEVICES="$BENCH_GPU" "$PYTHON_PATH" -u "$WORKDIR/infer_gpu_eagle_tree.py" \
            --model "$MODEL" --eagle "$EAGLE_OUT" \
            --dataset "$BENCH_DATASET" \
            --num_samples "$BENCH_NUM_SAMPLES" --max_new_tokens "$MAX_NEW_TOKENS" \
            --K "$K" --B "$TREE_B" --output "$EAGLE_BENCH_OUT" \
            | tee "${EAGLE_BENCH_OUT%.json}.log"
    else
        echo "  [2/2] EAGLE chain (K=$K, quant=${VERIFIER_QUANT:-bf16}) …"
        EXTRA_ARGS=()
        if [ "${VERIFIER_QUANT:-bf16}" = "gptq_marlin" ]; then
            : "${GPTQ_MARLIN_MODEL:?GPTQ_MARLIN_MODEL must be set when VERIFIER_QUANT=gptq_marlin}"
            : "${BF16_MODEL:?BF16_MODEL (path to bf16 MERaLiON for speech_encoder) must be set}"
            EXTRA_ARGS+=( --quant gptq_marlin --model "$GPTQ_MARLIN_MODEL"
                          --bf16_path "$BF16_MODEL" )
        else
            EXTRA_ARGS+=( --model "$MODEL" )
        fi
        CUDA_VISIBLE_DEVICES="$BENCH_GPU" "$PYTHON_PATH" -u "$WORKDIR/infer_gpu_eagle.py" \
            "${EXTRA_ARGS[@]}" --eagle "$EAGLE_OUT" \
            --dataset "$BENCH_DATASET" \
            --num_samples "$BENCH_NUM_SAMPLES" --max_new_tokens "$MAX_NEW_TOKENS" \
            --K "$K" --output "$EAGLE_BENCH_OUT" \
            | tee "${EAGLE_BENCH_OUT%.json}.log"
    fi
fi

# Summary
echo
echo "========================================"
echo "  Summary"
echo "========================================"
"$PYTHON_PATH" - "$BASELINE_OUT" "$EAGLE_BENCH_OUT" <<'PYEOF'
import json, sys, os
with open(sys.argv[1]) as f: b = json.load(f)
with open(sys.argv[2]) as f: e = json.load(f)
lat_b = b.get('avg_latency_s', 0); tps_b = b.get('avg_decode_tps', 0); wer_b = b.get('wer', 0)
lat_e = e.get('avg_latency_s', 0); tps_e = e.get('avg_decode_tps', 0); wer_e = e.get('wer', 0)
acc   = e.get('avg_spec_accept_rate', 0)
vb    = b.get('gpu_mem_peak_gb'); ve = e.get('gpu_mem_peak_gb')
hdr = f"  {'':<22} {'Lat(s)':>8} {'TPS':>7} {'WER%':>6} {'VRAM(GB)':>10}"
print(hdr); print('  ' + '-' * (len(hdr)-2))
print(f"  {'bf16 baseline':<22} {lat_b:>8.3f} {tps_b:>7.2f} {wer_b*100:>5.2f}% "
      f"{(vb if vb is not None else 0):>10.2f}")
print(f"  {'bf16 + EAGLE (K='+str(e.get('K','?'))+')':<22} "
      f"{lat_e:>8.3f} {tps_e:>7.2f} {wer_e*100:>5.2f}% "
      f"{(ve if ve is not None else 0):>10.2f}")
print()
print(f"  latency speedup     : {lat_b / max(lat_e, 1e-6):.2f}x")
print(f"  throughput speedup  : {tps_e / max(tps_b, 1e-6):.2f}x")
print(f"  accept rate (EAGLE) : {acc:.1%}")
PYEOF
