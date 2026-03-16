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
# Pruned mid3-22: INT8ao + torch.compile
echo ""
echo "========== CPU pruned: INT8ao + torch.compile =========="
$PYTHON_PATH -u infer_cpu.py \
    --model       "$CKPT" \
    --dataset     "$DATASET" \
    --num_samples "$NUM_BENCH_SAMPLES" \
    --int8ao \
    --output      "cpu_int8ao_${NAME}.json" \
    | tee cpu_int8ao_${NAME}.log

# Original model — INT4 + compile, meralion2_bl path (model.generate)
echo ""
echo "========== CPU original: INT4 + compile [meralion2_bl path] =========="
$PYTHON_PATH -u infer_cpu.py \
    --model            "$ORIGINAL" \
    --dataset          "$DATASET" \
    --num_samples      "$NUM_BENCH_SAMPLES" \
    --int4 \
    --output           "cpu_int4_original.json" \
    | tee cpu_int4_original.log

# Original model — INT8 dynamic (no compile) vs INT8ao + compile
echo ""
echo "========== CPU original: INT8 dynamic (no compile) =========="
$PYTHON_PATH -u infer_cpu.py \
    --model            "$ORIGINAL" \
    --dataset          "$DATASET" \
    --num_samples      "$NUM_BENCH_SAMPLES" \
    --trust_remote_code \
    --output           "cpu_int8_original.json" \
    | tee cpu_int8_original.log

echo ""
echo "========== CPU original: INT8ao + compile =========="
$PYTHON_PATH -u infer_cpu.py \
    --model            "$ORIGINAL" \
    --dataset          "$DATASET" \
    --num_samples      "$NUM_BENCH_SAMPLES" \
    --trust_remote_code \
    --int8ao \
    --output           "cpu_int8ao_original.json" \
    | tee cpu_int8ao_original.log
