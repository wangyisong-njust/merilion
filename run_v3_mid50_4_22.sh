#!/bin/bash
# ============================================================
# MERaLiON-2-3B Middle-block pruning experiment v3
# ============================================================
# Strategy: Prune 50% of text decoder layers 4-22 (both attn + MLP)
# Layers 0-3 and 22-25 remain unpruned (protected head/tail)
#
# NOTE: This produces non-uniform layer sizes. Use the custom
#       PrunedGemma2Model for vLLM inference (see pruned_gemma2_vllm.py).
# ============================================================

export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/tmp/hf_home"
PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"

cd $WORKDIR

# Common args
PRUNE_COMMON="--pruner_type taylor --taylor param_mix --block_wise --num_examples 20 --max_seq_len 256 --save_model"
LORA_ARGS="--lora_r 16 --lora_alpha 16 --learning_rate 5e-5 --num_epochs 3 --batch_size 8 --micro_batch_size 2"

# Middle-block pruning: layers 4-22 for both attention and MLP
TEXT_LAYERS="--block_attention_layer_start 4 --block_attention_layer_end 22 --block_mlp_layer_start 4 --block_mlp_layer_end 22"

# ============================================================
# GPU 2: Text Decoder mid-block 50% (layers 4-22, attn+mlp)
# ============================================================
GPU=2
NAME="v3-td50-mid4-22"
echo "[GPU $GPU] $NAME — text decoder attn+mlp 0.5, layers 4-22"

CUDA_VISIBLE_DEVICES=$GPU nohup bash -c "
$PYTHON_PATH -u meralion.py \
    --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B \
    --pruning_ratio 0.5 \
    --text_attn_pruning_ratio 0.5 --text_mlp_pruning_ratio 0.5 \
    $PRUNE_COMMON $TEXT_LAYERS \
    --post_prune_eval \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path meralion_checkpoints/MERaLiON-2-3B-$NAME
# && \
# $PYTHON_PATH -u post_training_meralion.py \
#     --base_model meralion_checkpoints/MERaLiON-2-3B-$NAME \
#     --output_dir meralion_tune_log/MERaLiON-2-3B-$NAME-tune \
#     $LORA_ARGS
" > tune_${NAME}.log 2>&1 &

# ============================================================
# vLLM inference test for pruned model (non-uniform layers)
# ============================================================
# After pruning completes, test vLLM loading with PrunedGemma2Model:
#
#   CUDA_VISIBLE_DEVICES=$GPU $PYTHON_PATH -c "
#   import sys; sys.path.insert(0, '$WORKDIR/vllm_inference')
#   from vllm import LLM, ModelRegistry
#   from meralion2_vllm_pruned import MERaLiON2PrunedForConditionalGeneration
#   ModelRegistry.register_model('MERaLiON2ForConditionalGeneration',
#                                MERaLiON2PrunedForConditionalGeneration)
#   llm = LLM(model='meralion_checkpoints/MERaLiON-2-3B-$NAME',
#             tokenizer='meralion_checkpoints/MERaLiON-2-3B-$NAME',
#             limit_mm_per_prompt={'audio': 1}, trust_remote_code=True)
#   print('vLLM loaded pruned model successfully!')
#   "

echo ""
echo "=========================================="
echo "Experiment submitted: $NAME on GPU $GPU"
echo "=========================================="
echo ""
echo "Config:"
echo "  Pruning ratio:  0.5 (50%) for both attn and MLP"
echo "  Layer range:    4-22 (protected: 0-3 head, 22-25 tail)"
echo "  Post-prune eval: enabled (500 samples, IMDA PART1)"
echo "  Post-training:  COMMENTED OUT (GPUs occupied)"
echo ""
echo "Monitor: tail -f tune_${NAME}.log"
echo "WER:     grep -E 'Post-prune WER|Final Test WER' tune_${NAME}.log"
