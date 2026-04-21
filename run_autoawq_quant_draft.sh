#!/bin/bash
# ============================================================
# AutoAWQ W4A16 quantization for the pruned draft model (mid3-23)
#
# Load with:  --draft_quant autoawq4
# ============================================================
export PYTHONUNBUFFERED=1
PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
DRAFT="${WORKDIR}/meralion_tune_log/MERaLiON-2-3B-v3-td50-mid3-23-tune"
DATASET="/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR"
SAVE="${WORKDIR}/MERaLiON-2-3B-mid3-23-AutoAWQ4"
NUM_CALIB=64
Q_GROUP_SIZE=128
GPU=0

export CUDA_VISIBLE_DEVICES=$GPU
cd "$WORKDIR"

echo "========================================"
echo "  AutoAWQ quantization  mid3-23 draft"
echo "  calib=${NUM_CALIB}  group=${Q_GROUP_SIZE}"
echo "  save → ${SAVE}"
echo "========================================"

"$PYTHON_PATH" -u quantize_autoawq.py \
    --model      "$DRAFT" \
    --dataset    "$DATASET" \
    --save       "$SAVE" \
    --num_calib  "$NUM_CALIB" \
    --q_group_size "$Q_GROUP_SIZE" \
    | tee autoawq_quant_draft.log

echo ""
echo "Log → autoawq_quant_draft.log"
echo "Load with:  python infer_gpu_spec_draft.py --draft $SAVE --draft_quant autoawq4 ..."
