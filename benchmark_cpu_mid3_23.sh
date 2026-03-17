#!/bin/bash
# CPU inference benchmark for v3-td50-mid3-23 (text 50%, layers 3-23)
# Uses the merged model (merge_lora.py output at TUNE_DIR).
# Compares FP32 baseline vs INT8 dynamic vs W8A8 (INT8 GEMM via oneDNN).

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
echo "--- Step 2: INT8 dynamic (legacy torch.quantization) ---"
$PYTHON_PATH -u infer_cpu.py \
    --model      $MODEL \
    --dataset    $DATASET \
    --num_samples $NUM_SAMPLES \
    --output     cpu_int8_${NAME}.json

echo ""
echo "--- Step 3: W8A8 (INT8 weights + INT8 dynamic activations) + torch.compile ---"
$PYTHON_PATH -u infer_cpu.py \
    --model      $MODEL \
    --dataset    $DATASET \
    --num_samples $NUM_SAMPLES \
    --w8a8 \
    --output     cpu_w8a8_${NAME}.json

echo ""
echo "========================================"
echo "Summary:"
$PYTHON_PATH -c "
import json
fp32 = json.load(open('cpu_fp32_${NAME}.json'))
int8 = json.load(open('cpu_int8_${NAME}.json'))
w8a8 = json.load(open('cpu_w8a8_${NAME}.json'))
fmt = '  {:<6s} WER={:.2f}%  latency={:.2f}s/sample  RAM={:.0f}MB'
print(fmt.format('FP32',  fp32['wer']*100, fp32['avg_latency_s'], fp32.get('ram_mb',0)))
print(fmt.format('INT8',  int8['wer']*100, int8['avg_latency_s'], int8.get('ram_mb',0)))
print(fmt.format('W8A8',  w8a8['wer']*100, w8a8['avg_latency_s'], w8a8.get('ram_mb',0)))
print()
int8_sp = fp32['avg_latency_s'] / int8['avg_latency_s']
w8a8_sp = fp32['avg_latency_s'] / w8a8['avg_latency_s']
print(f'  INT8 vs FP32:  {int8_sp:.2f}x speedup   ΔWER: {(int8[\"wer\"]-fp32[\"wer\"])*100:+.2f}%')
print(f'  W8A8 vs FP32:  {w8a8_sp:.2f}x speedup   ΔWER: {(w8a8[\"wer\"]-fp32[\"wer\"])*100:+.2f}%')
"
echo "========================================"
