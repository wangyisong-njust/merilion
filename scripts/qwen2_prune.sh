prune_ckpt_path='Qwen2.5-3B-ch-0.4_div16'
tune_ckpt_path='Qwen2.5-3B-ch-0.4_div16-finetuned'
# prune_ckpt_path='tmp'
# tune_ckpt_path='tmp_finetuned'
#Qwen/Qwen2.5-3B \
#--channel_wise \

echo "[START] - Start Pruning Model"
# python qwen.py \
#       --base_model Qwen/Qwen2.5-3B \
#       --pruning_ratio 0.4 \
#       --save_ckpt_log_name $prune_ckpt_path \
#       --pruner_type taylor --taylor param_first \
#       --save_model \
#       --channel_wise \
#       --max_seq_len 2048 
      # --block_wise \
      # --block_mlp_layer_start 0 --block_mlp_layer_end 35 \
      # --block_attention_layer_start 0 --block_attention_layer_end 35 \
      # --test_after_train --test_before_train  
      # --max_seq_len 2048 \
echo "[FINISH] - Finish Pruning Model"

echo "[START] - Start Tuning"
python post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin \
 --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path \
 --lora_r 32 --num_epochs 10 --learning_rate 1e-4 --batch_size 32 #--resume_from_checkpoint tune_log/Qwen2.5-3B-ch-0.3_v3-finetuned/checkpoint-1554/
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

