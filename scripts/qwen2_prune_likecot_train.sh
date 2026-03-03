prune_ckpt_path='Qwen2.5-3B-Instruct-bl-0.2-c4book'
tune_ckpt_path='Qwen2.5-3B-Instruct-bl-0.2-c4book-finetuned_4datasets_7_newprompt_fullmath'
output_ckpt_path='Qwen2.5-3B-Instruct-bl-0.2-c4book-finetuned_4datasets_7_newprompt_fullmath_likecot_stage2'
# prune_ckpt_path='tmp'
# tune_ckpt_path='tmp_finetuned'
#Qwen/Qwen2.5-3B \

# echo "[START] - Start Pruning Model"
# python qwen.py \
#       --base_model Qwen/Qwen2.5-3B-Instruct \
#       --pruning_ratio 0.5 \
#       --save_ckpt_log_name $prune_ckpt_path \
#       --pruner_type taylor --taylor vectorize \
#       --save_model \
#       --block_wise \
#       --block_mlp_layer_start 6 --block_mlp_layer_end 32 \
#       --block_attention_layer_start 6 --block_attention_layer_end 32 \
#       --max_seq_len 2048 \
#       --num_examples 20
# echo "[FINISH] - Finish Pruning Model"

echo "[START] - Start Tuning"
python liekcot_train.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --tune_model tune_log/$tune_ckpt_path/ \
 --data_path silk-road/alpaca-data-gpt4-chinese --output_dir tune_log/$output_ckpt_path \
 --lora_r 8 --num_epochs 10 --learning_rate 4e-4 --batch_size 4 #--cache_dataset
echo "[FINISH] - Finish Prune and Post-Training."
echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$output_ckpt_path}/"

# echo "You can use the command:"
# echo "       python generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$output_ckpt_path"
# echo "to use the pruned model"


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
