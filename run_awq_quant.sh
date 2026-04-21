#!/bin/bash
# ============================================================
# AWQ quantization for MERaLiON-2-3B text decoder
#
# Calibrates per-column scales on IMDA audio, applies W4A16
# INT4 group quantization, saves to MERaLiON-2-3B-AWQ4/.
#
# Load in infer_gpu.py with:  --model .../MERaLiON-2-3B-AWQ4 --quant awq4
# ============================================================
export PYTHONUNBUFFERED=1
PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
ORIGINAL="/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B"
DATASET="/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR"
SAVE="${WORKDIR}/MERaLiON-2-3B-AWQ4"
NUM_CALIB=64
GROUP_SIZE=64
ALPHA=0.5
GPU=0

export CUDA_VISIBLE_DEVICES=$GPU
cd "$WORKDIR"

echo "========================================"
echo "  AWQ quantization  MERaLiON-2-3B"
echo "  calib=${NUM_CALIB}  group=${GROUP_SIZE}  alpha=${ALPHA}"
echo "  save → ${SAVE}"
echo "========================================"

"$PYTHON_PATH" -u quantize_awq.py \
    --model      "$ORIGINAL" \
    --dataset    "$DATASET" \
    --save       "$SAVE" \
    --num_calib  "$NUM_CALIB" \
    --group_size "$GROUP_SIZE" \
    --alpha      "$ALPHA" \
    --device     "cuda" \
    | tee awq_quant.log

echo ""
echo "Quantization log → awq_quant.log"
echo "Load with:  python infer_gpu.py --model $SAVE --quant awq4 ..."
