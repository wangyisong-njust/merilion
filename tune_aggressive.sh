# Fine-tuning the aggressively pruned model (50% LLM + 50% Whisper)
PRUNED_MODEL=meralion_checkpoints/MERaLiON-2-3B-0_5-4-23-whisper-0_5-4-28
OUTPUT_DIR=meralion_tune_log/MERaLiON-2-3B-aggressive-50-tune

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python post_training_meralion.py --base_model $PRUNED_MODEL --output_dir $OUTPUT_DIR --lora_r 16 --lora_alpha 16 --learning_rate 5e-5 --num_epochs 3 --batch_size 8 --micro_batch_size 2
