python post_training.py \
    --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin \
    --data_path yahma/alpaca-cleaned \
    --output_dir tune_log/$tune_ckpt_path \
    --wandb_project llama_tune \
    --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64