prune_ckpt_path='SmolVLM-500M-Instruct-bl-0_2-4-28'
echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=2 python smolvlm.py \
      --base_model HuggingFaceTB/SmolVLM-500M-Instruct \
      --pruning_ratio 0.2 \
      --save_ckpt_log_name $prune_ckpt_path \
      --pruner_type taylor --taylor param_mix \
      --save_model \
      --block_wise \
      --block_mlp_layer_start 4 --block_mlp_layer_end 28 \
      --block_attention_layer_start 4 --block_attention_layer_end 28 \
      --max_seq_len 2048 \
      --num_examples 20 \
      --save_model_path vlm_checkpoints/$prune_ckpt_path 
echo "[FINISH] - Finish Pruning Model"




# prune_ckpt_path='SmolVLM-500M-Instruct-bl-0_2-4-28'
# tune_ckpt_path='SmolVLM-500M-Instruct-bl-0_2-4-28-tuned_qlora'
# echo "[START] - Start Tuning"
# CUDA_VISIBLE_DEVICES=2 python post_training_smolvlm.py --base_model ./vlm_checkpoints/$prune_ckpt_path \
#  --output_dir tune_log/$tune_ckpt_path \
#  --lora_r 16 --num_epochs 4 --learning_rate 4e-4 --batch_size 32 \
#  --coco_data_path /path/to/coco \
#  --llava_data_path /path/to/llava-instruct-150K \
#  --cache_dataset
# echo "[FINISH] - Finish Prune and Post-Training."