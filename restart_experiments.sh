#!/bin/bash
# ============================================================
# MERaLiON-2-3B 全层统一剪枝实验 v2 (8x A100 并行)
# ============================================================
# 关键变更: 全层统一剪枝 (layer 0-26), 确保 vLLM 兼容
# 之前用 layer 4-23 保护首尾层, 导致不同层尺寸不同, vLLM 无法加载
# 现在用 layer 0-26 全层统一, LoRA 微调会自愈精度损失
#
# 已有 v1 结果 (5000 样本 PART1, vLLM 推理):
#   TextMLP-25:  WER 0.0312
#   TextBoth-25: WER 0.0326
#   (基线 BF16:  WER 0.0488)
# ============================================================

export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/tmp/hf_home"
PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"

cd $WORKDIR

# 清理旧进程
pkill -f post_training_meralion.py
pkill -f meralion.py
sleep 2

# 清理 HF Arrow 缓存
echo "Cleaning HF cache..."
rm -rf ~/.cache/huggingface/datasets/ 2>/dev/null
rm -rf /tmp/hf_cache_* 2>/dev/null

# 通用微调参数
LORA_ARGS="--lora_r 16 --lora_alpha 16 --learning_rate 5e-5 --num_epochs 3 --batch_size 8 --micro_batch_size 2"
PRUNE_COMMON="--pruner_type taylor --taylor param_mix --block_wise --num_examples 20 --max_seq_len 256 --save_model"

# 全层统一剪枝: Gemma2 有 26 层 (0-25), Whisper 有 32 层 (0-31)
TEXT_LAYERS="--block_attention_layer_start 0 --block_attention_layer_end 26 --block_mlp_layer_start 0 --block_mlp_layer_end 26"
WHISPER_LAYERS="--whisper_block_layer_start 0 --whisper_block_layer_end 32"

# ============================================================
# GPU 0: Text Decoder Attention Only (0.25)
# ============================================================
NAME="v2-TextAttn-25"
echo "[GPU 0] $NAME"
CUDA_VISIBLE_DEVICES=0 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.25 --text_attn_pruning_ratio 0.25 \
    $PRUNE_COMMON $TEXT_LAYERS \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}.log 2>&1 &

# ============================================================
# GPU 1: Text Decoder MLP Only (0.25)
# ============================================================
NAME="v2-TextMLP-25"
echo "[GPU 1] $NAME"
CUDA_VISIBLE_DEVICES=1 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.25 --text_mlp_pruning_ratio 0.25 \
    $PRUNE_COMMON $TEXT_LAYERS \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}.log 2>&1 &

# ============================================================
# GPU 2: Text Decoder Combined (Attn 0.25 + MLP 0.25)
# ============================================================
NAME="v2-TextBoth-25"
echo "[GPU 2] $NAME"
CUDA_VISIBLE_DEVICES=2 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.25 \
    --text_attn_pruning_ratio 0.25 --text_mlp_pruning_ratio 0.25 \
    $PRUNE_COMMON $TEXT_LAYERS \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}.log 2>&1 &

# ============================================================
# GPU 3: Text Decoder 0.25 + Whisper Attention 0.10
# ============================================================
NAME="v2-td25-wa10"
echo "[GPU 3] $NAME"
CUDA_VISIBLE_DEVICES=3 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.25 \
    --text_attn_pruning_ratio 0.25 --text_mlp_pruning_ratio 0.25 \
    --whisper_attn_pruning_ratio 0.10 --whisper_mlp_pruning_ratio 0.0 \
    $PRUNE_COMMON $TEXT_LAYERS $WHISPER_LAYERS \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}.log 2>&1 &

# ============================================================
# GPU 4: Text Decoder 0.25 + Whisper Attention 0.15
# ============================================================
NAME="v2-td25-wa15"
echo "[GPU 4] $NAME"
CUDA_VISIBLE_DEVICES=4 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.25 \
    --text_attn_pruning_ratio 0.25 --text_mlp_pruning_ratio 0.25 \
    --whisper_attn_pruning_ratio 0.15 --whisper_mlp_pruning_ratio 0.0 \
    $PRUNE_COMMON $TEXT_LAYERS $WHISPER_LAYERS \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}.log 2>&1 &

# ============================================================
# GPU 5: 全栈剪枝 — Text 0.25 + Whisper Attn 0.10 + Whisper MLP 0.10
# ============================================================
NAME="v2-td25-wa10-wm10"
echo "[GPU 5] $NAME"
CUDA_VISIBLE_DEVICES=5 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.25 \
    --text_attn_pruning_ratio 0.25 --text_mlp_pruning_ratio 0.25 \
    --whisper_attn_pruning_ratio 0.10 --whisper_mlp_pruning_ratio 0.10 \
    $PRUNE_COMMON $TEXT_LAYERS $WHISPER_LAYERS \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}.log 2>&1 &

# ============================================================
# GPU 6: 保守区 — Text Attn 0.125 + MLP 0.15
# ============================================================
NAME="v2-ta125-tm15"
echo "[GPU 6] $NAME"
CUDA_VISIBLE_DEVICES=6 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.15 \
    --text_attn_pruning_ratio 0.125 --text_mlp_pruning_ratio 0.15 \
    $PRUNE_COMMON $TEXT_LAYERS \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}.log 2>&1 &

# ============================================================
# GPU 7: 激进区 — Text Attn 0.25 + MLP 0.35
# ============================================================
NAME="v2-ta25-tm35"
echo "[GPU 7] $NAME"
CUDA_VISIBLE_DEVICES=7 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.35 \
    --text_attn_pruning_ratio 0.25 --text_mlp_pruning_ratio 0.35 \
    $PRUNE_COMMON $TEXT_LAYERS \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}.log 2>&1 &

echo ""
echo "=========================================="
echo "All 8 v2 experiments submitted to GPU 0-7"
echo "=========================================="
echo ""
echo "实验概览 (全层统一剪枝, vLLM 兼容):"
echo "  GPU 0: v2-TextAttn-25     — Attention 单独 0.25"
echo "  GPU 1: v2-TextMLP-25      — MLP 单独 0.25"
echo "  GPU 2: v2-TextBoth-25     — Attn+MLP 各 0.25"
echo "  GPU 3: v2-td25-wa10       — +Whisper Attn 0.10"
echo "  GPU 4: v2-td25-wa15       — +Whisper Attn 0.15"
echo "  GPU 5: v2-td25-wa10-wm10  — +Whisper Attn+MLP 0.10"
echo "  GPU 6: v2-ta125-tm15      — 保守区 (~10%)"
echo "  GPU 7: v2-ta25-tm35       — 激进区 (~30%)"
echo ""
echo "监控: tail -f tune_v2-*.log"
echo "WER:  grep 'Validation WER' tune_v2-*.log"
