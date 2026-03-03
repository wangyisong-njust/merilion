# prune MiniCPM channel-wise 

# python MiniCPM_prune.py \
#       --base_model /home/kaixin/programs/MiniCPM-checkpoints/MiniCPM-2B-128k \
#       --pruning_ratio 0.11 \
#       --channel_wise \
#       --pruner_type taylor \
#       --device cuda  --eval_device cuda \
#       --data_path yahma/alpaca-cleaned \
#       --output_dir /home/kaixin/programs/MiniCPM-checkpoints/MiniCPM-2B-128k-pruned-ch-0.1-taylor-new2-finetuned \
#       --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 \
#       --save_model_path /home/kaixin/programs/MiniCPM-checkpoints/MiniCPM-2B-128k-pruned-ch-0.1-taylor-new2 \
#       # --base_model /home/kaixin/programs/MiniCPM-checkpoints/MiniCPM-2B-128k \

python MiniCPM_prune.py \
      --base_model /home/kaixin/programs/MiniCPM-checkpoints/MiniCPM-2B-128k-pruned-ch-0.25-taylor-new \
      --pruning_ratio 0 \
      --test_before_train \
      --layer_wise \
      --pruner_type taylor \
      --device cuda  --eval_device cuda \
      --data_path yahma/alpaca-cleaned \
      --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 \
      # --base_model /home/kaixin/programs/MiniCPM-checkpoints/MiniCPM-2B-128k \
# prune llama 2 model
# python hf_prune.py \
#       --base_model /home/kaixin/programs/llama/llama2-7b \
#       --pruning_ratio 0.25 \
#       --block_wise \
#       --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
#       --block_attention_layer_start 4 --block_attention_layer_end 30 \
#       --pruner_type taylor \
#       --device cuda  --eval_device cuda \
#       --data_path yahma/alpaca-cleaned \
#       --output_dir /home/kaixin/programs/MiniCPM-checkpoints/llama2-7b-pruned-bl-0.25-taylor-finetuned \
#       --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64
#       --save_model_path /home/kaixin/programs/MiniCPM-checkpoints/llama2-7b-ch-0.25-cuda


# prune MiniCPM block-wise 
# python MiniCPM_prune.py \
#       --base_model /home/kaixin/programs/MiniCPM-checkpoints/MiniCPM-2B-128k \
#       --pruning_ratio 0.5 \
#       --block_wise \
#       --block_mlp_layer_start 10 --block_mlp_layer_end 30 \
#       --block_attention_layer_start 10 --block_attention_layer_end 30 \
#       --pruner_type taylor \
#       --device cuda  --eval_device cuda \
#       --data_path yahma/alpaca-cleaned \
#       --output_dir /home/kaixin/programs/MiniCPM-checkpoints/MiniCPM-2B-128k-pruned-bl-0.2-taylor-finetuned \
#       --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64

            # --save_model_path /home/kaixin/programs/MiniCPM-checkpoints/MiniCPM-2B-128k-pruned-bl-0.3-taylor  --test_after_train
