#!/bin/bash
# ============================================================
# MERaLiON-2-10B pruning pipeline — 3 configs (scaled from 3B)
#
# 3B (26 layers)  →  10B (42 layers), same theoretical pruning ratio
#   mid3-22  →  mid5-36
#   mid3-23  →  mid5-37
#   mid4-23  →  mid6-37
#
# Steps per config:
#   1) Structured pruning (meralion.py)
#   2) LoRA post-training (post_training_meralion.py)
#   3) Merge LoRA (merge_lora.py)
#   4) CPU benchmark (infer_cpu.py)
# ============================================================
set -euo pipefail

export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/tmp/hf_home"

PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
BASE_MODEL="/home/jinchao/runtao/LLM_base_model/MERaLiON-2-10B-ASR"
DATASET="/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR"

CKPT_ROOT="meralion_checkpoints"
TUNE_ROOT="meralion_tune_log"
NUM_BENCH_SAMPLES=20

# Pruning common args
PRUNE_COMMON="--pruner_type taylor --taylor param_mix --block_wise --num_examples 20 --max_seq_len 256 --save_model --post_prune_eval"

# LoRA common args (smaller micro_batch for 10B memory)
LORA_ARGS="--lora_r 16 --lora_alpha 16 --learning_rate 5e-5 --num_epochs 3 --batch_size 8 --micro_batch_size 1 --lora_dropout 0.05"

cd "$WORKDIR"

# ── Configurations ────────────────────────────────────────────────────────
# Format: NAME:BLOCK_START:BLOCK_END:PRUNE_GPU:TRAIN_GPU0:TRAIN_GPU1
CONFIGS=(
    "v3-td50-mid5-36:5:36:0:0:1"
    "v3-td50-mid5-37:5:37:2:2:3"
    "v3-td50-mid6-37:6:37:4:4:5"
)

_run_one() {
    local NAME="$1" START="$2" END="$3" PGPU="$4" TGPU0="$5" TGPU1="$6"

    local CKPT_NAME="MERaLiON-2-10B-${NAME}"
    local CKPT_DIR="${CKPT_ROOT}/${CKPT_NAME}"
    local TUNE_DIR="${TUNE_ROOT}/${CKPT_NAME}-tune"

    echo ""
    echo "================================================================"
    echo "  10B Config: $NAME  (block $START→$END, td50)"
    echo "================================================================"

    # ── Step 1: Prune ─────────────────────────────────────────────────────
    if [ -d "$CKPT_DIR" ]; then
        echo "  [skip] pruned model already exists: $CKPT_DIR"
    else
        echo "  Step 1: pruning → $CKPT_DIR"
        CUDA_VISIBLE_DEVICES=$PGPU $PYTHON_PATH -u meralion.py \
            --base_model "$BASE_MODEL" \
            --pruning_ratio 0.5 \
            --text_attn_pruning_ratio 0.5 \
            --text_mlp_pruning_ratio 0.5 \
            --block_attention_layer_start $START \
            --block_attention_layer_end   $END \
            --block_mlp_layer_start       $START \
            --block_mlp_layer_end         $END \
            $PRUNE_COMMON \
            --save_ckpt_log_name "$CKPT_NAME" \
            --save_model_path    "$CKPT_DIR" \
            || { echo "  [FAIL] pruning $NAME"; return 1; }
    fi

    # ── Step 2: LoRA post-training (2-GPU DDP) ───────────────────────────
    if [ -d "$TUNE_DIR" ] && ls "${TUNE_DIR}"/model*.safetensors 2>/dev/null | grep -q .; then
        echo "  [skip] tune dir with merged weights exists: $TUNE_DIR"
    else
        echo "  Step 2: LoRA post-training → $TUNE_DIR"
        CUDA_VISIBLE_DEVICES=${TGPU0},${TGPU1} \
        torchrun --nproc_per_node=2 --master_port=$((29500 + TGPU0)) \
            $PYTHON_PATH -u post_training_meralion.py \
            --base_model "$CKPT_DIR" \
            --output_dir "$TUNE_DIR" \
            $LORA_ARGS \
            || { echo "  [FAIL] post-training $NAME"; return 1; }
    fi

    # ── Step 3: Merge LoRA ────────────────────────────────────────────────
    HAS_FULL=0
    ls "${TUNE_DIR}"/model*.safetensors 2>/dev/null | grep -q . && HAS_FULL=1
    ls "${TUNE_DIR}"/pytorch_model*.bin 2>/dev/null | grep -q . && HAS_FULL=1

    if [ "$HAS_FULL" = "0" ] && [ -f "${TUNE_DIR}/adapter_config.json" ]; then
        echo "  Step 3: merging LoRA → $TUNE_DIR"
        CUDA_VISIBLE_DEVICES=$PGPU $PYTHON_PATH -u merge_lora.py \
            --base    "$CKPT_DIR" \
            --adapter "$TUNE_DIR" \
            --output  "$TUNE_DIR" \
            || { echo "  [FAIL] merge $NAME"; return 1; }
    else
        echo "  [skip] merge: already has full weights or no adapter"
    fi

    # ── Step 4: CPU benchmark ─────────────────────────────────────────────
    local INT8_JSON="cpu_10b_int8_${NAME}.json"
    local INT4_JSON="cpu_10b_int4_${NAME}.json"

    echo "  Step 4a: CPU INT8 benchmark"
    $PYTHON_PATH -u infer_cpu.py \
        --model       "$TUNE_DIR" \
        --dataset     "$DATASET" \
        --num_samples "$NUM_BENCH_SAMPLES" \
        --output      "$INT8_JSON" \
        || echo "  [FAIL] INT8 benchmark"

    echo "  Step 4b: CPU INT4+compile benchmark"
    $PYTHON_PATH -u infer_cpu.py \
        --model       "$TUNE_DIR" \
        --dataset     "$DATASET" \
        --num_samples "$NUM_BENCH_SAMPLES" \
        --int4 \
        --output      "$INT4_JSON" \
        || echo "  [FAIL] INT4 benchmark"

    echo "  Done: $NAME"
}

# ── Run all configs ───────────────────────────────────────────────────────
for entry in "${CONFIGS[@]}"; do
    IFS=':' read -r NAME START END PGPU TGPU0 TGPU1 <<< "$entry"
    _run_one "$NAME" "$START" "$END" "$PGPU" "$TGPU0" "$TGPU1"
done

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  MERaLiON-2-10B pruning pipeline complete"
echo "================================================================"
echo "Results:"
for entry in "${CONFIGS[@]}"; do
    IFS=':' read -r NAME _ _ _ _ _ <<< "$entry"
    echo "  INT8: cpu_10b_int8_${NAME}.json"
    echo "  INT4: cpu_10b_int4_${NAME}.json"
done
