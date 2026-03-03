#!/bin/bash
# 环境变量
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"

cd $WORKDIR

# 确保旧进程已清理
pkill -f post_training_meralion.py
sleep 2

# 1. 启动 Attn-Only (GPU 4)
echo "Launching Attention experiment..."
CUDA_VISIBLE_DEVICES=4 nohup $PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-Attn-50 \
    --output_dir meralion_tune_log/MERaLiON-2-3B-Attn-50-tune \
    --lora_r 16 --lora_alpha 16 --learning_rate 5e-5 --num_epochs 3 --batch_size 8 --micro_batch_size 2 > tune_attn_v10.log 2>&1 &

# 2. 启动 MLP-Only (GPU 3)
echo "Launching MLP experiment..."
CUDA_VISIBLE_DEVICES=3 nohup $PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-MLP-50 \
    --output_dir meralion_tune_log/MERaLiON-2-3B-MLP-50-tune \
    --lora_r 16 --lora_alpha 16 --learning_rate 5e-5 --num_epochs 3 --batch_size 8 --micro_batch_size 2 > tune_mlp_v10.log 2>&1 &

# 3. 启动 Whisper-Only (GPU 5)
echo "Launching Whisper experiment..."
CUDA_VISIBLE_DEVICES=5 nohup $PYTHON_PATH -u post_training_meralion.py \
    --base_model meralion_checkpoints/MERaLiON-2-3B-0_5-4-23-whisper-0_5-4-28 \
    --output_dir meralion_tune_log/MERaLiON-2-3B-Whisper-50-tune \
    --lora_r 16 --lora_alpha 16 --learning_rate 5e-5 --num_epochs 3 --batch_size 8 --micro_batch_size 2 > tune_whisper_v11.log 2>&1 &

echo "All tasks submitted to nohup."
