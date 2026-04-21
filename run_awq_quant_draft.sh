#!/bin/bash
# ============================================================
# AWQ quantization for the pruned draft model (mid3-23)
#
# Calibrates per-column scales on IMDA audio, applies W4A16
# INT4 group quantization, saves to MERaLiON-2-3B-mid3-23-AWQ4/.
#
# Load in infer_gpu_spec_draft.py with:  --draft_quant awq4
# ============================================================
export PYTHONUNBUFFERED=1
PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
DRAFT="${WORKDIR}/meralion_tune_log/MERaLiON-2-3B-v3-td50-mid3-23-tune"
DATASET="/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR"
SAVE="${WORKDIR}/MERaLiON-2-3B-mid3-23-AWQ4-g32"
NUM_CALIB=256
GROUP_SIZE=32
ALPHA=0.5
GPU=0

export CUDA_VISIBLE_DEVICES=$GPU
cd "$WORKDIR"

echo "========================================"
echo "  AWQ quantization  mid3-23 draft"
echo "  calib=${NUM_CALIB}  group=${GROUP_SIZE}  alpha=${ALPHA}"
echo "  save → ${SAVE}"
echo "========================================"

"$PYTHON_PATH" -u quantize_awq.py \
    --model      "$DRAFT" \
    --dataset    "$DATASET" \
    --save       "$SAVE" \
    --num_calib  "$NUM_CALIB" \
    --group_size "$GROUP_SIZE" \
    --alpha      "$ALPHA" \
    --device     "cuda" \
    | tee awq_quant_draft.log

echo ""
echo "Quantization log → awq_quant_draft.log"
echo "Load with:  python infer_gpu_spec_draft.py --draft $SAVE --draft_quant awq4 ..."
