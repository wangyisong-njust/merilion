#!/bin/bash
cd /home/jinchao/runtao/LLM-Pruner

# Task A: MLP (Resume)
CUDA_VISIBLE_DEVICES=3 WANDB_MODE=offline nohup conda run --no-capture-output -n audiobench_quant python post_training_meralion.py \
  --base_model meralion_checkpoints/MERaLiON-2-3B-MLP-50 \
  --output_dir meralion_tune_log/MERaLiON-2-3B-MLP-50-tune \
  --lora_r 16 --lora_alpha 16 --learning_rate 5e-5 --num_epochs 3 --batch_size 8 --micro_batch_size 2 \
  --resume_from_checkpoint True > tune_mlp_v4.log 2>&1 &

# Task B: Attn (Resume)
CUDA_VISIBLE_DEVICES=4 WANDB_MODE=offline nohup conda run --no-capture-output -n audiobench_quant python post_training_meralion.py \
  --base_model meralion_checkpoints/MERaLiON-2-3B-Attn-50 \
  --output_dir meralion_tune_log/MERaLiON-2-3B-Attn-50-tune \
  --lora_r 16 --lora_alpha 16 --learning_rate 5e-5 --num_epochs 3 --batch_size 8 --micro_batch_size 2 \
  --resume_from_checkpoint True > tune_attn_v4.log 2>&1 &

# Task C: Whisper (Still in Pruning or First Epoch)
CUDA_VISIBLE_DEVICES=5 nohup conda run --no-capture-output -n audiobench_quant bash -c "python meralion.py --base_model /home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B --pruning_ratio 0.5 --pruner_type taylor --taylor param_mix --block_wise --whisper_block_layer_start 4 --whisper_block_layer_end 28 --save_model_path meralion_checkpoints/MERaLiON-2-3B-Whisper-50 --num_examples 3 --iterative_steps 1 && python post_training_meralion.py --base_model meralion_checkpoints/MERaLiON-2-3B-Whisper-50 --output_dir meralion_tune_log/MERaLiON-2-3B-Whisper-50-tune --lora_r 16 --lora_alpha 16 --learning_rate 5e-5 --num_epochs 3 --batch_size 8 --micro_batch_size 2" > tune_whisper_v4.log 2>&1 &
