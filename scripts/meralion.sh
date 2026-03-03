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



prune_ckpt_path='MERaLiON-2-10B-ASR-0_5-5-40'
tune_ckpt_path='MERaLiON-2-10B-ASR-0_5-5-40-tuned-r16'
tune_ckpt_path='MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32'
tune_ckpt_path='MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2' # 300 steps
tune_ckpt_path='MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2-imda2' # 300 steps
tune_ckpt_path='MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2-imda2-5e-5' # 300 steps
# tune_ckpt_path='MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2-mix-5e-5' # 300 steps
tune_ckpt_path='MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2-imda2-5e-5-grad_accu_2' # 300 steps, single dataset, gradient_accumulation_steps=2
tune_ckpt_path='MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2' # 600 steps, 2 dataset, gradient_accumulation_steps=2
# tune_ckpt_path='MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2-imda2-5e-5-SEQ_2_SEQ_LM' # 300 steps
# tune_ckpt_path='MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2-seq2seq'

prune_ckpt_path='MERaLiON-2-10B-ASR-0_25-7-35'
tune_ckpt_path='MERaLiON-2-10B-ASR-0_25-7-35-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2'
tune_ckpt_path='MERaLiON-2-10B-ASR-0_25-7-35-tuned-4bit-r16-full_gemma2-mix-5e-5-grad_accu_2'
tune_ckpt_path='MERaLiON-2-10B-ASR-0_25-7-35-tuned-4bit-r16-full_gemma2-mix-1e-4-grad_accu_2'
tune_ckpt_path='MERaLiON-2-10B-ASR-0_25-7-35-tuned-4bit-r16-full_gemma2-mix-2e-4-grad_accu_2'

prune_ckpt_path='MERaLiON-2-3B-0_25-3-23'
tune_ckpt_path='MERaLiON-2-10B-ASR-0_25-3-23-tuned-r16-full_gemma2-mix-1e-5-grad_accu_2-dropout01' # 350 steps, 'eval_loss': 0.2385
tune_ckpt_path='MERaLiON-2-10B-ASR-0_25-3-23-tuned-r16-full_gemma2-mix-2_5e-5-grad_accu_2-dropout01' # 400 steps
tune_ckpt_path='MERaLiON-2-10B-ASR-0_25-3-23-tuned-r16-full_gemma2-mix-5e-5-grad_accu_2-dropout01'
tune_ckpt_path='MERaLiON-2-10B-ASR-0_25-3-23-tuned-r16-full_gemma2-mix-1e-4-grad_accu_2-dropout01'

tune_ckpt_path='MERaLiON-2-10B-ASR-0_25-3-23-tuned-r32-full_gemma2-mix-1e-5-grad_accu_2-dropout01' # 350 steps, 'eval_loss': 0.2373
tune_ckpt_path='MERaLiON-2-10B-ASR-0_25-3-23-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-dropout01' 

tune_ckpt_path='MERaLiON-2-10B-ASR-0_25-3-23-tuned-r32-full_gemma2-mix-1e-5-grad_accu_4-dropout01'
tune_ckpt_path='MERaLiON-2-10B-ASR-0_25-3-23-tuned-r32-full_gemma2-mix-5e-5-grad_accu_4-dropout01' 

tune_ckpt_path='MERaLiON-2-10B-ASR-0_25-3-23-tuned-r8-a16-full_gemma2-mix-1e-5-grad_accu_2-dropout01'
tune_ckpt_path='MERaLiON-2-10B-ASR-0_25-3-23-tuned-r8-a16-full_gemma2-mix-5e-5-grad_accu_2-dropout01' # 250 steps, 'eval_loss': 0.2461642026901245
tune_ckpt_path='MERaLiON-2-10B-ASR-0_25-3-23-tuned-r8-a16-full_gemma2-mix-1e-4-grad_accu_2-dropout01'
tune_ckpt_path='MERaLiON-2-10B-ASR-0_25-3-23-tuned-r8-a16-full_gemma2-mix-3e-4-grad_accu_2-dropout01'

tune_ckpt_path='MERaLiON-2-3B-0_25-3-23-tuned-r16-a16-full_gemma2-mix-5e-6-bs32-dropout005' # 250 steps, 'eval_loss': 0.2412317991256714
# tune_ckpt_path='MERaLiON-2-3B-0_25-3-23-tuned-r16-a16-full_gemma2-mix-1e-5-bs32-dropout005' # 150 steps, 'eval_loss': 0.2435568869113922
# tune_ckpt_path='MERaLiON-2-3B-0_25-3-23-tuned-r16-a16-full_gemma2-mix-5e-5-bs32-dropout005' 
# tune_ckpt_path='MERaLiON-2-3B-0_25-3-23-tuned-r16-a16-full_gemma2-mix-1e-4-bs32-dropout005'



# tune_ckpt_path='MERaLiON-2-10B-ASR-0_25-3-23-tuned-4bit-r16-full_gemma2-mix-5e-5-grad_accu_2'
# tune_ckpt_path='MERaLiON-2-10B-ASR-0_25-3-23-tuned-4bit-r16-full_gemma2-mix-1e-4-grad_accu_2'
# tune_ckpt_path='MERaLiON-2-10B-ASR-0_25-3-23-tuned-4bit-r16-full_gemma2-mix-2e-4-grad_accu_2'

prune_ckpt_path='MERaLiON-2-3B-0_5-3-23'
# tune_ckpt_path='MERaLiON-2-10B-ASR-0_5-3-23-tuned-r16-full_gemma2-mix-1e-5-grad_accu_2-dropout01' # 350 steps, 'eval_loss': 0.2688
# tune_ckpt_path='MERaLiON-2-10B-ASR-0_5-3-23-tuned-r16-full_gemma2-mix-2_5e-5-grad_accu_2-dropout01' # 400 steps
# tune_ckpt_path='MERaLiON-2-10B-ASR-0_5-3-23-tuned-r16-full_gemma2-mix-5e-5-grad_accu_2-dropout01'
# tune_ckpt_path='MERaLiON-2-10B-ASR-0_5-3-23-tuned-r16-full_gemma2-mix-1e-4-grad_accu_2-dropout01'

# tune_ckpt_path='MERaLiON-2-10B-ASR-0_5-3-23-tuned-r32-full_gemma2-mix-1e-5-grad_accu_2-dropout01' # 350 steps, 'eval_loss': 0.2691
# tune_ckpt_path='MERaLiON-2-10B-ASR-0_5-3-23-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-dropout01' 

# tune_ckpt_path='MERaLiON-2-10B-ASR-0_5-3-23-tuned-r32-full_gemma2-mix-1e-5-grad_accu_4-dropout01'
# tune_ckpt_path='MERaLiON-2-10B-ASR-0_5-3-23-tuned-r32-full_gemma2-mix-5e-5-grad_accu_4-dropout01' 

# tune_ckpt_path='MERaLiON-2-10B-ASR-0_5-3-23-tuned-4bit-r16-full_gemma2-mix-5e-5-grad_accu_2'
# tune_ckpt_path='MERaLiON-2-10B-ASR-0_5-3-23-tuned-4bit-r16-full_gemma2-mix-1e-4-grad_accu_2'
# tune_ckpt_path='MERaLiON-2-10B-ASR-0_5-3-23-tuned-4bit-r16-full_gemma2-mix-2e-4-grad_accu_2'

tune_ckpt_path='MERaLiON-2-3B-0_5-3-23-tuned-r16-a16-full_gemma2-mix-5e-6-bs32-dropout005'
tune_ckpt_path='MERaLiON-2-3B-0_5-3-23-tuned-r16-a16-full_gemma2-mix-1e-5-bs32-dropout005' # 200 steps, 'eval_loss': 0.27764901518821716
tune_ckpt_path='MERaLiON-2-3B-0_5-3-23-tuned-r16-a16-full_gemma2-mix-5e-5-bs32-dropout005' 
tune_ckpt_path='MERaLiON-2-3B-0_5-3-23-tuned-r16-a16-full_gemma2-mix-1e-4-bs32-dropout005'


prune_ckpt_path='MERaLiON-2-3B-0_5-4-23'
tune_ckpt_path='MERaLiON-2-3B-0_25-4-23-tuned-r16-a16-1e-6-bs8-imda1m3c'
tune_ckpt_path='MERaLiON-2-3B-0_25-4-23-tuned-r16-a16-5e-6-bs8-imda1m3c'
tune_ckpt_path='MERaLiON-2-3B-0_25-4-23-tuned-r16-a16-1e-5-bs8-imda1m3c'
tune_ckpt_path='MERaLiON-2-3B-0_25-4-23-tuned-r16-a16-5e-5-bs8-imda1m3c' # 3100 steps, "eval_loss": 1.0146843194961548,

tune_ckpt_path='MERaLiON-2-3B-0_25-4-23-tuned-r16-a16-1e-6-bs1024-imda1m3c' # 1.16 still going down
# tune_ckpt_path='MERaLiON-2-3B-0_25-4-23-tuned-r16-a16-5e-6-bs1024-imda1m3c' # 30 steps, 'eval_loss': 1.11068
# tune_ckpt_path='MERaLiON-2-3B-0_25-4-23-tuned-r16-a16-1e-5-bs1024-imda1m3c' # 20 steps, 'eval_loss': 1.110245
# tune_ckpt_path='MERaLiON-2-3B-0_25-4-23-tuned-r16-a16-5e-5-bs1024-imda1m3c' # 60 steps, 'eval_loss': 1.147992

prune_ckpt_path='MERaLiON-2-3B-0_25-4-23-both' # Param Ratio = 81.7308%
tune_ckpt_path='MERaLiON-2-3B-0_25-4-23-both-tuned-r16-a16-5e-6-bs8-imda1m3c'
tune_ckpt_path='MERaLiON-2-3B-0_25-4-23-both-tuned-r16-a16-1e-5-bs8-imda1m3c'
tune_ckpt_path='MERaLiON-2-3B-0_25-4-23-both-tuned-r16-a16-5e-5-bs8-imda1m3c' # 2600 steps, 'eval_loss': 1.0427570
tune_ckpt_path='MERaLiON-2-3B-0_25-4-23-both-tuned-r16-a16-1e-4-bs8-imda1m3c' # epoch 0.88, 'eval_loss': 1.0491824

tune_ckpt_path='MERaLiON-2-3B-0_25-4-23-both-tuned-r16-a16-5e-6-bs1024-imda1m3c' 
tune_ckpt_path='MERaLiON-2-3B-0_25-4-23-both-tuned-r16-a16-1e-5-bs1024-imda1m3c'

echo "[START] - Start Pruning Model"
# CUDA_VISIBLE_DEVICES=2 python meralion.py \
#       --base_model MERaLiON-2-3B \
#       --pruning_ratio 0.25 \
#       --pruner_type taylor --taylor param_mix \
#       --block_wise \
#       --block_mlp_layer_start 4 --block_mlp_layer_end 23 \
#       --block_attention_layer_start 4 --block_attention_layer_end 23 \
#       --num_examples 20 \
#       --max_seq_len 256 \
#       --save_model \
#       --save_ckpt_log_name $prune_ckpt_path \
#       --save_model_path meralion_checkpoints/$prune_ckpt_path 

# CUDA_VISIBLE_DEVICES=7 python meralion.py \
#       --base_model MERaLiON-2-10B-ASR \
#       --pruning_ratio 0.5 \
#       --pruner_type taylor --taylor param_mix \
#       --block_wise \
#       --block_mlp_layer_start 5 --block_mlp_layer_end 40 \
#       --block_attention_layer_start -1 --block_attention_layer_end -1 \
#       --num_examples 20 \
#       --max_seq_len 256 \
#       --save_model \
#       --save_ckpt_log_name $prune_ckpt_path \
#       --save_model_path meralion_checkpoints/$prune_ckpt_path \
#       --test_before_train 
      # --whisper_block_layer_start 4 --whisper_block_layer_end 28 \
    #    \
    #   
echo "[FINISH] - Finish Pruning Model"

echo "[START] - Start Tuning"
# python post_training_meralion.py --base_model ./meralion_checkpoints/$prune_ckpt_path \
CUDA_VISIBLE_DEVICES=5 WANDB_MODE=offline python post_training_meralion.py --base_model MERaLiON-2-3B \
 --output_dir meralion_tune_log/$tune_ckpt_path \
 --lora_r 16 --lora_alpha 16 --learning_rate 1e-5 --num_epochs 3 --batch_size 1024 --micro_batch_size 4 --lora_dropout 0.05 \
#  --wandb_project $prune_ckpt_path-finetune --wandb_run_name $tune_ckpt_path
#  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"

# python post_training_smolvlm.py --base_model ./vlm_checkpoints/$prune_ckpt_path \
#  --output_dir tune_log/$tune_ckpt_path \
#  --lora_r 16 --num_epochs 4 --learning_rate 4e-4 --batch_size 32 \
#  --cache_dataset
#  --resume_from_checkpoint ./tune_log/SmolVLM-500M-Instruct-bl-0.5-tuned_qlora-3data-r64/checkpoint-500
echo "[FINISH] - Finish Prune and Post-Training."
# echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

# echo "You can use the command:"
# echo "       python generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path"
# echo "to use the pruned model"

# merge and save model
# CUDA_VISIBLE_DEVICES=1 python merge_meralion.py \
#     --ckpt ./meralion_checkpoints/MERaLiON-2-10B-ASR-0_5-5-40 \
#     --lora_ckpt ./meralion_tune_log/MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2/checkpoint-100 \
#     --save_path MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2-merged

  
# CUDA_VISIBLE_DEVICES=3 python merge_meralion.py \
#     --ckpt /home/kaixin/programs/A100_backup/meralion_checkpoints/MERaLiON-2-10B-ASR-0_25-7-35 \
#     --lora_ckpt ./meralion_tune_log/MERaLiON-2-10B-ASR-0_25-7-35-tuned-r16 \
#     --save_path MERaLiON-2-10B-ASR-0_25-7-35-merged


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
