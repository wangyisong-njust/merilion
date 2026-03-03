# prune_ckpt_path='SmolVLM-500M-Instruct-bl-0.6' # 43.125%, tgs: 72.40 ± 2.94 tok/s
prune_ckpt_path='SmolVLM-500M-Instruct-bl-0_2-8-24' # LM: 10%, ppl: 7.93, tgs: 54.56 ± 2.63
prune_ckpt_path='SmolVLM-500M-Instruct-bl-0_2-4-28' # LM: 15%, ppl: 9.11
# prune_ckpt_path='SmolVLM-500M-Instruct-bl-0_2-1-32' # LM: 19.375%, ppl: 13.33
# prune_ckpt_path='SmolVLM-500M-Instruct-bl-0_2-0-32' # LM: 20%, ppl: 19.97
# prune_ckpt_path='SmolVLM-500M-Instruct-bl-0_4-8-24' # LM: 20%, ppl: 14.41, tgs: 59.05 ± 1.63
# prune_ckpt_path='SmolVLM-500M-Instruct-bl-0_4-5-29' # LM: 30%, ppl: 26.82, tgs: 64.58 ± 2.82
# prune_ckpt_path='SmolVLM-500M-Instruct-bl-0_4-3-30' # LM: 33.75%, ppl: 41.18
# prune_ckpt_path='SmolVLM-500M-Instruct-bl-0_4-1-32' # LM: 38.75%, ppl: 195.09
# prune_ckpt_path='SmolVLM-500M-Instruct-bl-0_6-6-27' # LM: 39.375%, ppl: 112.91, tgs: 69.85 ± 8.08
# tune_ckpt_path='SmolVLM-500M-Instruct-bl-0.4-tuned_qlora'
# tune_ckpt_path='SmolVLM-500M-Instruct-bl-0.5-tuned_qlora-4data'
# tune_ckpt_path='SmolVLM-500M-Instruct-bl-0.5-tuned_qlora-3data-r32' # lora_r 32, learning_rate 5e-4, warm_up 0.1
# tune_ckpt_path='SmolVLM-500M-Instruct-bl-0.5-tuned_qlora-3data-r64' # lora_r 32, learning_rate 5e-4, warm_up 0.1
# tune_ckpt_path='SmolVLM-500M-Instruct-bl-0.6-tuned_qlora-3data-r32' # lora_r 32, learning_rate 5e-4, warm_up 0.1
# tune_ckpt_path='SmolVLM-500M-Instruct-bl-0.6-tuned_qlora-2data-r32' # lora_r 32, learning_rate 5e-5, warm_up 0.1
# tune_ckpt_path='SmolVLM-500M-Instruct-bl-0.6-tuned_qlora-2data-r32-new' # lora_r 32, learning_rate 5e-4, warm_up 0.05, mask input token
# tune_ckpt_path='SmolVLM-500M-Instruct-bl-0.6-tuned_qlora-2data-r32-new-new' # lora_r 32, learning_rate 5e-4, warm_up 0.05
# tune_ckpt_path='SmolVLM-500M-Instruct-bl-0.6-tuned_qlora-1data-r32' # lora_r 32, learning_rate 5e-3, warm_up 0.05
# tune_ckpt_path='SmolVLM-500M-Instruct-bl-0.6-tuned_qlora-1data-r32-new' # lora_r 32, learning_rate 1e-3, warm_up 0.05, 10000 training data
# tune_ckpt_path='SmolVLM-500M-Instruct-bl-0.6-tuned_qlora-1data-r32-wild' # lora_r 32, learning_rate 1e-3, warm_up 0.05, 10000 training data + llava-wild
# tune_ckpt_path='SmolVLM-500M-Instruct-bl-0.6-tuned_qlora-1data-r32-wild-dominate' # lora_r 32, learning_rate 4e-4, warm_up 0.05, 100 training data + llava-wild
# tune_ckpt_path='SmolVLM-500M-Instruct-bl-0.6-tuned_qlora-1data-r32-llava150K' # lora_r 32, learning_rate 8e-4, warm_up 0.05, 10000 llava150K
# tune_ckpt_path='SmolVLM-500M-Instruct-bl-0.6-tuned_qlora-r32-llava150K-wild' # lora_r 32, learning_rate 8e-4, warm_up 0.05, 10000 llava150K + wild*34

tune_ckpt_path='SmolVLM-500M-Instruct-bl-0_2-8-24-tuned_qlora-r16-llava150K+textvqa_freezevision' # lora_r 16, 4 epochs, learning_rate 4e-4, warm_up 0.05, 10000 llava150K
tune_ckpt_path='SmolVLM-500M-Instruct-bl-0_2-4-28-tuned_qlora-r16-llava150K+textvqa_freezevision' # lora_r 16, 4 epochs, learning_rate 4e-4, warm_up 0.05, 10000 llava150K
# tune_ckpt_path='SmolVLM-500M-Instruct-bl-0_2-4-28-tuned_qlora-r16-llava150K' # lora_r 16, 4 epochs, learning_rate 4e-4, warm_up 0.05, 10000 llava150K
# tune_ckpt_path='SmolVLM-500M-Instruct-bl-0_4-8-24-tuned_qlora-r16-llava150K' # lora_r 16, learning_rate 4e-4, warm_up 0.05, 10000 llava150K
# tune_ckpt_path='SmolVLM-500M-Instruct-bl-0_4-5-29-tuned_qlora-r32-llava150K' # lora_r 32, learning_rate 8e-4, warm_up 0.05, 10000 llava150K
# tune_ckpt_path='SmolVLM-500M-Instruct-bl-0_6-6-27-tuned_qlora-r32-llava150K' # lora_r 32, learning_rate 8e-4, warm_up 0.05, 10000 llava150K

# prune_ckpt_path='tmp'
# tune_ckpt_path='tmp_finetuned'
#Qwen/Qwen2.5-3B \

echo "[START] - Start Pruning Model"
# python smolvlm.py \
#       --base_model HuggingFaceTB/SmolVLM-500M-Instruct \
#       --pruning_ratio 0.6 \
#       --save_model_path vlm_checkpoints/$prune_ckpt_path \
#       --pruner_type taylor --taylor param_mix \
#       --save_model \
#       --block_wise \
#       --block_mlp_layer_start 5 --block_mlp_layer_end 28 \
#       --block_attention_layer_start 5 --block_attention_layer_end 28 \
#       --max_seq_len 2048 \
#       --num_examples 20 \
#       --save_ckpt_log_name $prune_ckpt_path
# python smolvlm.py \
#       --base_model HuggingFaceTB/SmolVLM-500M-Instruct \
#       --pruning_ratio 0.2 \
#       --save_ckpt_log_name $prune_ckpt_path \
#       --pruner_type taylor --taylor param_mix \
#       --save_model \
#       --block_wise \
#       --block_mlp_layer_start 4 --block_mlp_layer_end 28 \
#       --block_attention_layer_start 4 --block_attention_layer_end 28 \
#       --max_seq_len 2048 \
#       --num_examples 20 \
#       --save_model_path vlm_checkpoints/$prune_ckpt_path 
echo "[FINISH] - Finish Pruning Model"

echo "[START] - Start Tuning"
python post_training_smolvlm.py --base_model ./vlm_checkpoints/$prune_ckpt_path \
 --output_dir tune_log/$tune_ckpt_path \
 --lora_r 16 --num_epochs 4 --learning_rate 4e-4 --batch_size 32 \
 --cache_dataset
#  --resume_from_checkpoint ./tune_log/SmolVLM-500M-Instruct-bl-0.5-tuned_qlora-3data-r64/checkpoint-500
echo "[FINISH] - Finish Prune and Post-Training."
# echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

# echo "You can use the command:"
# echo "       python generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path"
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
