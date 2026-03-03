prune_ckpt_path='SmolVLM-500M-Instruct-bl-0.4'
tune_ckpt_path='SmolVLM-500M-Instruct-bl-0.4-tuned_qlora'
# prune_ckpt_path='tmp'
# tune_ckpt_path='tmp_finetuned'
#Qwen/Qwen2.5-3B \

# echo "[START] - Start Pruning Model"
# python smolvlm.py \
#       --base_model HuggingFaceTB/SmolVLM-500M-Instruct \
#       --pruning_ratio 0.5 \
#       --save_ckpt_log_name $prune_ckpt_path \
#       --pruner_type taylor --taylor param_mix \
#       --save_model \
#       --block_wise \
#       --block_mlp_layer_start 1 --block_mlp_layer_end 31 \
#       --block_attention_layer_start 1 --block_attention_layer_end 31 \
#       --max_seq_len 2048 \
#       --num_examples 20
# echo "[FINISH] - Finish Pruning Model"

# echo "[START] - Start Tuning"
# python post_training_smolvlm.py --base_model ./vlm_checkpoints/SmolVLM-500M-Instruct-bl \
#  --output_dir tune_log/$tune_ckpt_path \
#  --lora_r 16 --num_epochs 5 --learning_rate 5e-4 --batch_size 32 #--cache_dataset
# echo "[FINISH] - Finish Prune and Post-Training."
# echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

python merge_smolvlm_lora.py --ckpt prune_log/SmolVLM-500M-Instruct-bl-0.5/pytorch_model.bin --lora_ckpt scripts/SmolVLM-500M-Instruct-bl-0.4-lora \
    

# python generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path
# python generate.py --model_type tune_prune_LLM --ckpt scripts/SmolVLM-500M-Instruct-bl-0.4-lora/pytorch_model.bin --lora_ckpt scripts/SmolVLM-500M-Instruct-bl-0.4-lora


# prune_ckpt_path='Llama-3.2-3B-Instruct-bl-0.25'
# tune_ckpt_path='Llama-3.2-3B-Instruct-bl-0.25-finetuned'
# python llama3.py \
#       --base_model /home/kaixin/programs/MiniCPM-checkpoints/Llama-3.2-3B-Instruct \
#       --pruning_ratio 0.5 \
#       --block_wise \
#       --block_mlp_layer_start 4 --block_mlp_layer_end 24 \
#       --block_attention_layer_start 4 --block_attention_layer_end 24 \
#       --save_ckpt_log_name $prune_ckpt_path \
#       --pruner_type taylor --taylor param_first \
#       --max_seq_len 2048 \
#       --save_model


# echo "[START] - Start Pruning Model"
# python qwen.py \
#       --base_model qwen_checkpoints/Qwen2_5_3B_bl_0_4 \
#       --pruning_ratio 0.25 \
#       --save_ckpt_log_name $prune_ckpt_path \
#       --pruner_type taylor --taylor vectorize \
#       --save_model \
#       --channel_wise \
#       --max_seq_len 2048 
# echo "[FINISH] - Finish Pruning Model"
