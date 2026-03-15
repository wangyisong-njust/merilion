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

GPU=0
NAME="v3-td50-mid3-22"
CKPT="meralion_checkpoints/MERaLiON-2-3B-$NAME"
ORIGINAL="/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B"
DATASET="/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR"
NUM_BENCH_SAMPLES=50

echo "[GPU $GPU] Quantization benchmark: $NAME"

CUDA_VISIBLE_DEVICES=$GPU nohup bash -c "
echo '========== BF16 baseline ==========' && \
$PYTHON_PATH -u vllm_benchmark_pruned.py \
    --pruned $CKPT \
    --original $ORIGINAL \
    --dataset $DATASET \
    --num_samples $NUM_BENCH_SAMPLES \
    --output vllm_benchmark_${NAME}-BF16.json && \
echo '' && \
echo '========== Quantize → W8A16 (llm-compressor RTN) ==========' && \
$PYTHON_PATH -u quantize_pruned.py \
    --model $CKPT \
    --scheme W8A16 && \
echo '' && \
echo '========== W8A16 benchmark ==========' && \
$PYTHON_PATH -u vllm_benchmark_pruned.py \
    --pruned ${CKPT}-W8A16-RTN \
    --original $ORIGINAL \
    --dataset $DATASET \
    --num_samples $NUM_BENCH_SAMPLES \
    --output vllm_benchmark_${NAME}-W8A16.json && \
echo '' && \
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
