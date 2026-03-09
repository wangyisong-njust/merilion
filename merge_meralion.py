'''
Refer to
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
'''

import os
import sys
import argparse
from typing import List
from pathlib import Path

import torch
import transformers
from datasets import load_dataset
from transformers.models.idefics3 import Idefics3Processor, Idefics3ForConditionalGeneration
from LLMPruner.peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from LLMPruner.datasets.ppl_dataset import get_loaders
from transformers.image_utils import load_image
from LLMPruner.peft import PeftModel
from LLMPruner.evaluator.ppl import PPLMetric

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration

from peft import PeftModel

import shutil
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

def measure_decode_speed(model, prompt, processor, images=None, max_length=30):
    """
    Measure the decode speed for generating the rest of the tokens.
    """
    inputs = processor(prompt, images=images, return_tensors="pt").to(device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_length,
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output

def copy_all_files(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    dst_dir.mkdir(parents=True, exist_ok=True)

    for item in src_dir.iterdir():
        dst_path = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst_path)  # preserves metadata

def copy_files_only(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for file in src_dir.glob("*"):
        if file.is_file():
            shutil.copy2(file, dst_dir / file.name)
            print("Copied", file.name)

def merge_base_and_lora_weight(args):
    # pruned_dict = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    # processor, model = pruned_dict['processor'], pruned_dict['model']
    # # processor = Idefics3Processor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
    # model = model.to(device)
    
    processor = AutoProcessor.from_pretrained(
        args.ckpt, 
        trust_remote_code=True,
    )
    model = MERaLiON2ForConditionalGeneration.from_pretrained(
        args.ckpt,
        use_safetensors=True,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",
        # attn_implementation="sdpa",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    if args.lora_ckpt is not None:
        model = PeftModel.from_pretrained(
            model,
            args.lora_ckpt,
            torch_dtype=torch.bfloat16,
        )

        model = model.merge_and_unload()

    
    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
        print(model)
        # Fix generation_config conflict: use_cache=False + cache_implementation="hybrid"
        if hasattr(model, 'generation_config'):
            model.generation_config.use_cache = True
        model.save_pretrained(args.save_path)
        processor.save_pretrained(args.save_path)
    
    # copy configuration and modeling files
    # copy_files_only("./meralion2_bl_infer", args.save_path)
    copy_files_only("./meralion2_bl", args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    # Model Type&Path
    # parser.add_argument('--base_model', type=str, default="/home/kaixin/programs/MiniCPM-checkpoints/MiniCPM-2B-128k", help='base model name')
    parser.add_argument('--ckpt', type=str, default='./MiniCPM-checkpoints/MiniCPM-2B-128k-pruned-bl-0.3-taylor', help='pruned model path')
    parser.add_argument('--lora_ckpt', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--push-to-hub', type=str, default=None, help='Push the model to HuggingFace Hub')

    args = parser.parse_args()

    merge_base_and_lora_weight(args)

# MERaLiON-2-3B-0_25-3-23
# CUDA_VISIBLE_DEVICES=0 python merge_meralion.py \
#     --ckpt ./meralion_checkpoints/MERaLiON-2-3B-0_25-3-23 \
#     --lora_ckpt meralion_tune_log/MERaLiON-2-10B-ASR-0_25-3-23-tuned-r32-full_gemma2-mix-1e-5-grad_accu_2-dropout01 \
#     --save_path MERaLiON-2-3B-0_25-3-23-tuned-r32-full_gemma2-mix-1e-5-grad_accu_2-dropout01-merged

# CUDA_VISIBLE_DEVICES=0 python merge_meralion.py \
#     --ckpt ./meralion_checkpoints/MERaLiON-2-3B-0_25-4-23-both \
#     --lora_ckpt meralion_tune_log/MERaLiON-2-3B-0_25-4-23-both-tuned-r16-a16-5e-5-bs8-imda1m3c/checkpoint-2600 \
#     --save_path MERaLiON-2-3B-0_25-4-23-both-tuned-r16-a16-5e-6-bs8-imda1m3c-merged

# MERaLiON-2-3B-0_25-3-23
# CUDA_VISIBLE_DEVICES=0 python merge_meralion.py \
#     --ckpt ./meralion_checkpoints/MERaLiON-2-3B-0_25-3-23 \
#     --lora_ckpt meralion_tune_log/MERaLiON-2-10B-ASR-0_25-3-23-tuned-r32-full_gemma2-mix-1e-5-grad_accu_2-dropout01 \
#     --save_path MERaLiON-2-3B-0_25-3-23-tuned-r32-full_gemma2-mix-1e-5-grad_accu_2-dropout01-merged



# MERaLiON-2-3B-0_5-3-23
# CUDA_VISIBLE_DEVICES=1 python merge_meralion.py \
#     --ckpt ./meralion_checkpoints/MERaLiON-2-3B-0_5-3-23 \
#     --lora_ckpt meralion_tune_log/MERaLiON-2-10B-ASR-0_5-3-23-tuned-r16-full_gemma2-mix-1e-5-grad_accu_2-dropout01 \
#     --save_path MERaLiON-2-3B-0_5-3-23-tuned-r16-full_gemma2-mix-1e-5-grad_accu_2-dropout01-merged



# 0_25-7-35
# CUDA_VISIBLE_DEVICES=0 python merge_meralion.py \
#     --ckpt /home/jinchao/runtao/LLM_base_model/MERaLiON-2-10B-ASR-0_25-7-35 \
#     --lora_ckpt meralion_tune_log/MERaLiON-2-10B-ASR-0_25-7-35-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-600_steps \
#     --save_path MERaLiON-2-10B-ASR-0_25-7-35-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-new

# # 0_5-5-40
# CUDA_VISIBLE_DEVICES=6 python merge_meralion.py \
#     --ckpt ./meralion_checkpoints/MERaLiON-2-10B-ASR-0_5-5-40 \
#     --lora_ckpt meralion_tune_log/MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2/checkpoint-300 \
#     --save_path MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged


    
