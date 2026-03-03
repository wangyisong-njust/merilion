prune_ckpt_path='deepseek-7b-bl-0.3'
tune_ckpt_path='deepseek-7b-bl-0.3-finetuned'

echo "[START] - Start Pruning Model"
python deepseek.py \
      --base_model deepseek-ai/deepseek-llm-7b-chat \
      --pruning_ratio 0.5 \
      --save_ckpt_log_name $prune_ckpt_path \
      --pruner_type taylor --taylor vectorize \
      --block_wise \
      --block_mlp_layer_start 5 --block_mlp_layer_end 25 \
      --block_attention_layer_start 5 --block_attention_layer_end 25 \
      --save_model \
      --max_seq_len 2048 
echo "[FINISH] - Finish Pruning Model"

echo "[START] - Start Tuning"
# python post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin \
#  --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path \
#  --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 
python post_training_c4zh.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin \
 --data_path silk-road/alpaca-data-gpt4-chinese --output_dir tune_log/$tune_ckpt_path \
 --lora_r 8 --num_epochs 2 --learning_rate 2e-4 --batch_size 32 #--cache_dataset
echo "[FINISH] - Finish Prune and Post-Training."
echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

echo "You can use the command:"
echo "       python generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path"
echo "to use the pruned model"


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

