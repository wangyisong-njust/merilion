CUDA_VISIBLE_DEVICES=0 python test_pruned_smolvlm.py \
    --base_model HuangRT/SmolVLM-500M-Instruct-bl-15 \
    --quant_8bit \
    --input_file ./questions/001.jsonl \
    --output_dir ./output
