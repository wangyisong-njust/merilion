#!/bin/bash
# ============================================================
# Quantization speed benchmark for v3-td50-mid3-22
# BF16 baseline → W8A16 RTN → W4A16 AWQ
# ============================================================
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/tmp/hf_home"
PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"

cd $WORKDIR

GPU=1
NAME="v3-td50-mid3-22"
CKPT="meralion_checkpoints/MERaLiON-2-3B-$NAME"
ORIGINAL="/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B"
DATASET="/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR"
NUM_BENCH_SAMPLES=50

echo "[GPU $GPU] Quantization benchmark: $NAME"

# ── helper: skip if JSON already exists ──────────────────────────────────
run_cpu_if_missing() {
    local json="$1"; shift
    if [ -f "$json" ]; then echo "  [skip] $json already exists"; return 0; fi
    echo "  running → $json"
    "$PYTHON_PATH" -u infer_cpu.py --output "$json" "$@" \
        | tee "${json%.json}.log" \
        || { echo "[FAIL] $json"; return 1; }
}

run_gpu_if_missing() {
    local json="$1"; shift
    if [ -f "$json" ]; then echo "  [skip] $json already exists"; return 0; fi
    echo "  running → $json"
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_PATH" -u infer_gpu.py --output "$json" "$@" \
        | tee "${json%.json}.log" \
        || { echo "[FAIL] $json"; return 1; }
}

CUDA_VISIBLE_DEVICES=$GPU nohup bash -c "
echo '========== Quantize → W4A16 (AutoAWQ, calibrated) ==========' && \
$PYTHON_PATH -u quantize_pruned_awq.py \
    --model $CKPT \
    --dataset $DATASET && \
echo '' && \
echo '========== W4A16 AWQ benchmark ==========' && \
$PYTHON_PATH -u vllm_benchmark_pruned.py \
    --pruned ${CKPT}-W4A16-AWQ \
    --original $ORIGINAL \
    --dataset $DATASET \
    --num_samples $NUM_BENCH_SAMPLES \
    --output vllm_benchmark_${NAME}-W4A16-AWQ.json
" > quant_bench_${NAME}.log 2>&1 &

echo ""
echo "Submitted to GPU $GPU — monitor: tail -f quant_bench_${NAME}.log"
echo "Results: vllm_benchmark_${NAME}-{BF16,W8A16,W4A16-AWQ}.json"

# ── CPU benchmarks ────────────────────────────────────────────────────────
echo ""
echo "========== CPU pruned: INT8ao + torch.compile =========="
run_cpu_if_missing "cpu_int8ao_${NAME}.json" \
    --model "$CKPT" --dataset "$DATASET" \
    --num_samples "$NUM_BENCH_SAMPLES" --int8ao

echo ""
echo "========== CPU original: INT4 + compile =========="
run_cpu_if_missing "cpu_int4_original.json" \
    --model "$ORIGINAL" --dataset "$DATASET" \
    --num_samples "$NUM_BENCH_SAMPLES" --int4

echo ""
echo "========== CPU original: INT8 dynamic =========="
run_cpu_if_missing "cpu_int8_original.json" \
    --model "$ORIGINAL" --dataset "$DATASET" \
    --num_samples "$NUM_BENCH_SAMPLES" --trust_remote_code

echo ""
echo "========== CPU original: INT8ao + compile =========="
run_cpu_if_missing "cpu_int8ao_original.json" \
    --model "$ORIGINAL" --dataset "$DATASET" \
    --num_samples "$NUM_BENCH_SAMPLES" --trust_remote_code --int8ao

# ── GPU speculative decoding benchmarks ──────────────────────────────────
GAMMA=5

echo ""
echo "========== GPU pruned BF16: no-spec =========="
run_gpu_if_missing "gpu_bf16_${NAME}_nospec.json" \
    --model "$CKPT" --dataset "$DATASET" \
    --num_samples "$NUM_BENCH_SAMPLES" --quant bf16

echo ""
echo "========== GPU pruned BF16: +spec γ=${GAMMA} =========="
run_gpu_if_missing "gpu_bf16_${NAME}_spec${GAMMA}.json" \
    --model "$CKPT" --dataset "$DATASET" \
    --num_samples "$NUM_BENCH_SAMPLES" --quant bf16 \
    --speculative --gamma "$GAMMA"

echo ""
echo "========== GPU original BF16: no-spec =========="
run_gpu_if_missing "gpu_bf16_original_nospec.json" \
    --model "$ORIGINAL" --dataset "$DATASET" \
    --num_samples "$NUM_BENCH_SAMPLES" --quant bf16

echo ""
echo "========== GPU original BF16: +spec γ=${GAMMA} =========="
run_gpu_if_missing "gpu_bf16_original_spec${GAMMA}.json" \
    --model "$ORIGINAL" --dataset "$DATASET" \
    --num_samples "$NUM_BENCH_SAMPLES" --quant bf16 \
    --speculative --gamma "$GAMMA"
