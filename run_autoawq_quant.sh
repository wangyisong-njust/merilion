#!/bin/bash
# ============================================================
# AutoAWQ W4A16 quantization for MERaLiON-2-3B text decoder
#
# Processes layer-by-layer → much lower GPU memory than custom AWQ.
# Speech encoder and audio adapter remain in FP16.
#
# Load with:  python infer_gpu.py --model .../MERaLiON-2-3B-AutoAWQ4 --quant autoawq4
# ============================================================
export PYTHONUNBUFFERED=1
PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
ORIGINAL="/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B"
DATASET="/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR"
SAVE="${WORKDIR}/MERaLiON-2-3B-AutoAWQ4"
NUM_CALIB=64
Q_GROUP_SIZE=128
GPU=0

export CUDA_VISIBLE_DEVICES=$GPU
cd "$WORKDIR"

echo "========================================"
echo "  AutoAWQ quantization  MERaLiON-2-3B"
echo "  calib=${NUM_CALIB}  group=${Q_GROUP_SIZE}"
echo "  save → ${SAVE}"
echo "========================================"

"$PYTHON_PATH" -u quantize_autoawq.py \
    --model      "$ORIGINAL" \
    --dataset    "$DATASET" \
    --save       "$SAVE" \
    --num_calib  "$NUM_CALIB" \
    --q_group_size "$Q_GROUP_SIZE" \
    | tee autoawq_quant.log

echo ""
echo "Log → autoawq_quant.log"
echo "Load with:  python infer_gpu.py --model $SAVE --quant autoawq4 ..."
