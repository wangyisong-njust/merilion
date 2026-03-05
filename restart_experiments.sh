#!/bin/bash
# ============================================================
# MERaLiON-2-3B 分组件差异化剪枝实验 (8x A100 并行)
# ============================================================
#
# 已有结果 (baseline):
#   原始 BF16:                WER 0.0488, 6.50GB, 1.00x
#   Text Decoder 19.23%:      WER 0.0527, 5.88GB, 1.34x  ← 甜点
#   Text Decoder 19.23%+INT8: WER 0.0532, 4.30GB, 1.47x  ← 最佳性价比
#   Text Decoder 38.46%:      WER 0.0619, 5.29GB, 1.39x  ← 太激进
#
# 本轮实验设计 (8 GPU 并行):
#   GPU 0-1: 分离实验 — Attention vs MLP 各自的独立贡献
#   GPU 2:   复现基线 — 用修复后代码复现 19.23% 结果
#   GPU 3-4: Whisper 探索 — 在甜点基础上叠加 whisper attention 剪枝
#   GPU 5:   全栈剪枝 — 包含 whisper MLP
#   GPU 6:   保守区 — ~10% 剪枝率, 最小 WER 损失
#   GPU 7:   激进区 — ~25-30% 剪枝率, 找到质量悬崖
# ============================================================

export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1          # 禁止任何网络请求（服务器无网）
export TRANSFORMERS_OFFLINE=1         # 同上，transformers 也不联网
export HF_HOME="/tmp/hf_home"         # HF 缓存放 /tmp，不占主盘
PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"

cd $WORKDIR

# 清理旧进程
pkill -f post_training_meralion.py
pkill -f meralion.py
sleep 2

# 清理之前残留的 HF Arrow 缓存（可能几百GB）
echo "Cleaning HF cache..."
rm -rf ~/.cache/huggingface/datasets/ 2>/dev/null
rm -rf /tmp/hf_cache_* 2>/dev/null

# 通用微调参数
LORA_ARGS="--lora_r 16 --lora_alpha 16 --learning_rate 5e-5 --num_epochs 3 --batch_size 8 --micro_batch_size 2"
PRUNE_COMMON="--pruner_type taylor --taylor param_mix --block_wise --num_examples 20 --max_seq_len 256 --save_model"

# ============================================================
# GPU 0: Text Decoder Attention Only (0.25)
#   目的: 隔离 KV cache 减少的独立贡献
#   预期: 参数减少不多, 但推理时 KV cache 减 25%
# ============================================================
NAME="TextAttn-25"
echo "[GPU 0] $NAME"
CUDA_VISIBLE_DEVICES=0 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.25 --text_attn_pruning_ratio 0.25 \
    $PRUNE_COMMON \
    --block_attention_layer_start 4 --block_attention_layer_end 23 \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}_v12.log 2>&1 &

# ============================================================
# GPU 1: Text Decoder MLP Only (0.25)
#   目的: 隔离 MLP 剪枝的独立贡献 (MLP 是最大参数组件)
# ============================================================
NAME="TextMLP-25"
echo "[GPU 1] $NAME"
CUDA_VISIBLE_DEVICES=1 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.25 --text_mlp_pruning_ratio 0.25 \
    $PRUNE_COMMON \
    --block_mlp_layer_start 4 --block_mlp_layer_end 23 \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}_v12.log 2>&1 &

# ============================================================
# GPU 2: Text Decoder Combined (Attn 0.25 + MLP 0.25)
#   目的: 用修复后的代码复现已有 19.23% 剪枝结果作为 baseline
#   这对应已有表格中 WER=0.0527 的实验
# ============================================================
NAME="TextBoth-25"
echo "[GPU 2] $NAME"
CUDA_VISIBLE_DEVICES=2 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.25 \
    --text_attn_pruning_ratio 0.25 --text_mlp_pruning_ratio 0.25 \
    $PRUNE_COMMON \
    --block_mlp_layer_start 4 --block_mlp_layer_end 23 \
    --block_attention_layer_start 4 --block_attention_layer_end 23 \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}_v12.log 2>&1 &

# ============================================================
# GPU 3: Text Decoder 0.25 + Whisper Attention 0.10
#   目的: 在甜点基础上叠加保守的 whisper attention 剪枝
#   Whisper 20 heads × 0.10 = 剪 2 heads, 保留 18 heads
# ============================================================
NAME="td25-wa10"
echo "[GPU 3] $NAME"
CUDA_VISIBLE_DEVICES=3 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.25 \
    --text_attn_pruning_ratio 0.25 --text_mlp_pruning_ratio 0.25 \
    --whisper_attn_pruning_ratio 0.10 --whisper_mlp_pruning_ratio 0.0 \
    $PRUNE_COMMON \
    --block_mlp_layer_start 4 --block_mlp_layer_end 23 \
    --block_attention_layer_start 4 --block_attention_layer_end 23 \
    --whisper_block_layer_start 4 --whisper_block_layer_end 28 \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}_v12.log 2>&1 &

# ============================================================
# GPU 4: Text Decoder 0.25 + Whisper Attention 0.15
#   目的: 稍激进的 whisper attention 剪枝, 与 GPU 3 对比
#   Whisper 20 heads × 0.15 = 剪 3 heads, 保留 17 heads
# ============================================================
NAME="td25-wa15"
echo "[GPU 4] $NAME"
CUDA_VISIBLE_DEVICES=4 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.25 \
    --text_attn_pruning_ratio 0.25 --text_mlp_pruning_ratio 0.25 \
    --whisper_attn_pruning_ratio 0.15 --whisper_mlp_pruning_ratio 0.0 \
    $PRUNE_COMMON \
    --block_mlp_layer_start 4 --block_mlp_layer_end 23 \
    --block_attention_layer_start 4 --block_attention_layer_end 23 \
    --whisper_block_layer_start 4 --whisper_block_layer_end 28 \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}_v12.log 2>&1 &

# ============================================================
# GPU 5: 全栈剪枝 — Text Decoder 0.25 + Whisper Attn 0.10 + Whisper MLP 0.10
#   目的: 验证 whisper MLP 是否可以轻度剪枝
#   post_training 会自动检测并添加 fc1/fc2 到 LoRA targets
# ============================================================
NAME="td25-wa10-wm10"
echo "[GPU 5] $NAME"
CUDA_VISIBLE_DEVICES=5 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.25 \
    --text_attn_pruning_ratio 0.25 --text_mlp_pruning_ratio 0.25 \
    --whisper_attn_pruning_ratio 0.10 --whisper_mlp_pruning_ratio 0.10 \
    $PRUNE_COMMON \
    --block_mlp_layer_start 4 --block_mlp_layer_end 23 \
    --block_attention_layer_start 4 --block_attention_layer_end 23 \
    --whisper_block_layer_start 4 --whisper_block_layer_end 28 \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}_v12.log 2>&1 &

# ============================================================
# GPU 6: 保守区 — Text Attn 0.125 + MLP 0.15
#   目的: 最小可行剪枝, 追求 WER 几乎不变
#   Attn 0.125 = 剪 1/8 KV heads; MLP 0.15 = 温和剪枝
#   预期: ~8-10% 参数减少, WER 极小劣化
# ============================================================
NAME="ta125-tm15"
echo "[GPU 6] $NAME"
CUDA_VISIBLE_DEVICES=6 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.15 \
    --text_attn_pruning_ratio 0.125 --text_mlp_pruning_ratio 0.15 \
    $PRUNE_COMMON \
    --block_mlp_layer_start 4 --block_mlp_layer_end 23 \
    --block_attention_layer_start 4 --block_attention_layer_end 23 \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}_v12.log 2>&1 &

# ============================================================
# GPU 7: 激进区 — Text Attn 0.25 + MLP 0.35
#   目的: 找到质量悬崖, 在 19.23% 和 38.46% 之间探索
#   预期: ~25-30% 参数减少, WER 在 0.055-0.060 之间
# ============================================================
NAME="ta25-tm35"
echo "[GPU 7] $NAME"
CUDA_VISIBLE_DEVICES=7 nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.35 \
    --text_attn_pruning_ratio 0.25 --text_mlp_pruning_ratio 0.35 \
    $PRUNE_COMMON \
    --block_mlp_layer_start 4 --block_mlp_layer_end 23 \
    --block_attention_layer_start 4 --block_attention_layer_end 23 \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
    --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
    $LORA_ARGS
" > tune_${NAME}_v12.log 2>&1 &

echo ""
echo "=========================================="
echo "All 8 experiments submitted to GPU 0-7"
echo "=========================================="
echo ""
echo "实验概览:"
echo "  GPU 0: TextAttn-25       — Attention 单独 0.25"
echo "  GPU 1: TextMLP-25        — MLP 单独 0.25"
echo "  GPU 2: TextBoth-25       — 复现 19.23% baseline"
echo "  GPU 3: td25-wa10         — +Whisper Attn 0.10"
echo "  GPU 4: td25-wa15         — +Whisper Attn 0.15"
echo "  GPU 5: td25-wa10-wm10    — +Whisper Attn+MLP 0.10"
echo "  GPU 6: ta125-tm15        — 保守区 (~10%)"
echo "  GPU 7: ta25-tm35         — 激进区 (~25-30%)"
echo ""
echo "监控: tail -f tune_*_v12.log"
echo "WER:  grep 'Validation WER' tune_*_v12.log"
