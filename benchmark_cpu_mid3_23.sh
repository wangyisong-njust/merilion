#!/bin/bash
# CPU INT4 inference benchmark for v3-td50-mid3-23 (text 50%, layers 3-23)
# Uses the merged model (merge_lora.py output at TUNE_DIR).
# Compares FP32 baseline vs torchao INT4 + torch.compile.

PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"

NAME="v3-td50-mid3-23"
# Merged model: merge_lora.py --output $TUNE_DIR writes the full model here
MODEL="meralion_tune_log/MERaLiON-2-3B-${NAME}-tune"
DATASET="/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR"
NUM_SAMPLES=20

cd $WORKDIR

echo "========================================"
echo "CPU Benchmark: $NAME"
echo "Model (merged): $MODEL"
echo "Samples: $NUM_SAMPLES"
echo "========================================"

echo ""
echo "--- Step 1: FP32 baseline (no quant, no compile) ---"
$PYTHON_PATH -u infer_cpu.py \
    --model      $MODEL \
    --dataset    $DATASET \
    --num_samples $NUM_SAMPLES \
    --no_quant --no_compile \
    --output     cpu_fp32_${NAME}.json

echo ""
echo "--- Step 2: INT4 + torch.compile ---"
$PYTHON_PATH -u infer_cpu.py \
    --model      $MODEL \
    --dataset    $DATASET \
    --num_samples $NUM_SAMPLES \
    --output     cpu_int4_${NAME}.json

echo ""
echo "========================================"
echo "Summary:"
$PYTHON_PATH -c "
import json
fp32 = json.load(open('cpu_fp32_${NAME}.json'))
int4 = json.load(open('cpu_int4_${NAME}.json'))
speedup = fp32['avg_latency_s'] / int4['avg_latency_s']
print(f'  FP32   WER={fp32[\"wer\"]*100:.2f}%  latency={fp32[\"avg_latency_s\"]:.2f}s/sample')
print(f'  INT4   WER={int4[\"wer\"]*100:.2f}%  latency={int4[\"avg_latency_s\"]:.2f}s/sample')
print(f'  Speedup: {speedup:.2f}x')
"
echo "========================================"
