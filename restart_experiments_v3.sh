#!/bin/bash
# ============================================================
# MERaLiON-2-3B v3 剪枝实验 (8x A100 并行)
# ============================================================
# 基于 v2 结果设计:
#   - v2-TextBoth-25: WER 0.0343, 加速 20.6% (Attn+MLP 各 25%)
#   - v2-ta25-tm35:   WER 0.0361, 加速 21.1% (Attn 25% + MLP 35%)
#   - 结论: text decoder 25% 剪枝已触及加速天花板 (~20%)
#           因为 Whisper encoder 占固定开销, 只剪 decoder 提升有限
#
# v3 策略:
#   1. 更激进的 text decoder 剪枝 (40-50%), 看 WER 退化多少
#   2. 加入 Whisper FFN 剪枝 (只剪 encoder_ffn_dim, 不剪 attention)
#      → vLLM 兼容, 减少 encoder 固定开销
#   3. text decoder + Whisper FFN 组合, 追求 30-40% 总体加速
#
# 注意: 不剪 Whisper attention (会改 d_model, vLLM 不兼容)
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

# 通用参数
LORA_ARGS="--lora_r 16 --lora_alpha 16 --learning_rate 5e-5 --num_epochs 3 --batch_size 8 --micro_batch_size 2"
PRUNE_COMMON="--pruner_type taylor --taylor param_mix --block_wise --num_examples 20 --max_seq_len 256 --save_model"

# 全层统一剪枝范围
TEXT_LAYERS="--block_attention_layer_start 0 --block_attention_layer_end 26 --block_mlp_layer_start 0 --block_mlp_layer_end 26"
WHISPER_LAYERS="--whisper_block_layer_start 0 --whisper_block_layer_end 32"

# ============================================================
# GPU 0: 更激进 text decoder — Attn 0.35 + MLP 0.45
# 目标: 探索 text decoder 更大剪枝空间
# ============================================================
NAME="v3-ta35-tm45"
echo "[GPU 0] $NAME"
CUDA_VISIBLE_DEVICES=0 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.45 \
    --text_attn_pruning_ratio 0.35 --text_mlp_pruning_ratio 0.45 \
    $PRUNE_COMMON $TEXT_LAYERS \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}.log 2>&1 &

# ============================================================
# GPU 1: 极限 text decoder — Attn 0.50 + MLP 0.50
# 目标: 找到 WER 崩溃点
# ============================================================
NAME="v3-TextBoth-50"
echo "[GPU 1] $NAME"
CUDA_VISIBLE_DEVICES=1 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.50 \
    --text_attn_pruning_ratio 0.50 --text_mlp_pruning_ratio 0.50 \
    $PRUNE_COMMON $TEXT_LAYERS \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}.log 2>&1 &

# ============================================================
# GPU 2: Whisper FFN 单独剪 25% (text decoder 不剪)
# 目标: 量化 Whisper FFN 剪枝的 WER 影响和加速效果
# ============================================================
NAME="v3-WhisperFFN-25"
echo "[GPU 2] $NAME"
CUDA_VISIBLE_DEVICES=2 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.25 \
    --text_attn_pruning_ratio 0.0 --text_mlp_pruning_ratio 0.0 \
    --whisper_attn_pruning_ratio 0.0 --whisper_mlp_pruning_ratio 0.25 \
    $PRUNE_COMMON $TEXT_LAYERS $WHISPER_LAYERS \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}.log 2>&1 &

# ============================================================
# GPU 3: Whisper FFN 单独剪 50% (text decoder 不剪)
# 目标: 探索 Whisper FFN 的剪枝极限
# ============================================================
NAME="v3-WhisperFFN-50"
echo "[GPU 3] $NAME"
CUDA_VISIBLE_DEVICES=3 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.50 \
    --text_attn_pruning_ratio 0.0 --text_mlp_pruning_ratio 0.0 \
    --whisper_attn_pruning_ratio 0.0 --whisper_mlp_pruning_ratio 0.50 \
    $PRUNE_COMMON $TEXT_LAYERS $WHISPER_LAYERS \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}.log 2>&1 &

# ============================================================
# GPU 4: 组合 — TextBoth 25% + WhisperFFN 25%
# 目标: 全栈剪枝 (vLLM 兼容), 追求 30%+ 加速
# ============================================================
NAME="v3-tb25-wf25"
echo "[GPU 4] $NAME"
CUDA_VISIBLE_DEVICES=4 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.25 \
    --text_attn_pruning_ratio 0.25 --text_mlp_pruning_ratio 0.25 \
    --whisper_attn_pruning_ratio 0.0 --whisper_mlp_pruning_ratio 0.25 \
    $PRUNE_COMMON $TEXT_LAYERS $WHISPER_LAYERS \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}.log 2>&1 &

# ============================================================
# GPU 5: 组合 — TextBoth 25% + WhisperFFN 50%
# 目标: 最大化 Whisper FFN 剪枝 + 稳定的 text 剪枝
# ============================================================
NAME="v3-tb25-wf50"
echo "[GPU 5] $NAME"
CUDA_VISIBLE_DEVICES=5 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.25 \
    --text_attn_pruning_ratio 0.25 --text_mlp_pruning_ratio 0.25 \
    --whisper_attn_pruning_ratio 0.0 --whisper_mlp_pruning_ratio 0.50 \
    $PRUNE_COMMON $TEXT_LAYERS $WHISPER_LAYERS \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}.log 2>&1 &

# ============================================================
# GPU 6: 激进组合 — TextAttn 35% + TextMLP 45% + WhisperFFN 25%
# 目标: 最大化总体加速 (text+whisper 都剪)
# ============================================================
NAME="v3-ta35-tm45-wf25"
echo "[GPU 6] $NAME"
CUDA_VISIBLE_DEVICES=6 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.45 \
    --text_attn_pruning_ratio 0.35 --text_mlp_pruning_ratio 0.45 \
    --whisper_attn_pruning_ratio 0.0 --whisper_mlp_pruning_ratio 0.25 \
    $PRUNE_COMMON $TEXT_LAYERS $WHISPER_LAYERS \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}.log 2>&1 &

# ============================================================
# GPU 7: 极限全栈 — TextBoth 50% + WhisperFFN 50%
# 目标: 找到整体崩溃点, 模型还能用吗?
# ============================================================
NAME="v3-tb50-wf50"
echo "[GPU 7] $NAME"
CUDA_VISIBLE_DEVICES=7 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.50 \
    --text_attn_pruning_ratio 0.50 --text_mlp_pruning_ratio 0.50 \
    --whisper_attn_pruning_ratio 0.0 --whisper_mlp_pruning_ratio 0.50 \
    $PRUNE_COMMON $TEXT_LAYERS $WHISPER_LAYERS \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}.log 2>&1 &

echo ""
echo "=========================================="
echo "All 8 v3 experiments submitted to GPU 0-7"
echo "=========================================="
echo ""
echo "实验概览 (v3: 更激进 text + Whisper FFN 剪枝):"
echo "  GPU 0: v3-ta35-tm45          — Text Attn 35% + MLP 45%"
echo "  GPU 1: v3-TextBoth-50        — Text Attn+MLP 各 50% (极限)"
echo "  GPU 2: v3-WhisperFFN-25      — 仅 Whisper FFN 25%"
echo "  GPU 3: v3-WhisperFFN-50      — 仅 Whisper FFN 50%"
echo "  GPU 4: v3-tb25-wf25          — Text 25% + Whisper FFN 25%"
echo "  GPU 5: v3-tb25-wf50          — Text 25% + Whisper FFN 50%"
echo "  GPU 6: v3-ta35-tm45-wf25     — Text 激进 + Whisper FFN 25%"
echo "  GPU 7: v3-tb50-wf50          — 全部 50% (极限)"
echo ""
echo "监控: tail -f tune_v3-*.log"
echo "WER:  grep 'Validation WER' tune_v3-*.log"
