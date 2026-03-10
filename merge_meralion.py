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
import json
from pathlib import Path
from safetensors import safe_open

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

def fix_config_for_vllm(save_path):
    """Update config.json to match actual pruned weight dimensions for vLLM compatibility.

    The pruned model's config.json stores original dimensions + midblock_ratio.
    Custom HF code uses resize_to_match() to handle this, but vLLM's built-in
    Gemma2/Whisper uses config dimensions directly. This function reads the actual
    weight shapes and updates config.json to match.
    """
    config_path = os.path.join(save_path, "config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path) as f:
        config = json.load(f)

    # Collect weight shapes from safetensors
    st_files = sorted([f for f in os.listdir(save_path) if f.endswith('.safetensors')])
    if not st_files:
        print("[fix_config] No safetensors files found, skipping")
        return

    weight_shapes = {}
    for st_file in st_files:
        with safe_open(os.path.join(save_path, st_file), framework="pt") as f:
            for key in f.keys():
                weight_shapes[key] = f.get_tensor(key).shape

    changed = False

    # --- Fix text_config (Gemma2 text decoder) ---
    # IMPORTANT: key matching must use 'text_decoder' prefix to avoid mixing up
    # with speech_encoder keys (e.g., Whisper's k_proj has d_model=1280 which
    # would be wrongly interpreted as 5 KV heads if matched for text decoder).
    text_config = config.get("text_config", {})

    # MLP: intermediate_size from gate_proj [intermediate_size, hidden_size]
    for key, shape in weight_shapes.items():
        if 'text_decoder' in key and '.layers.0.mlp.gate_proj.weight' in key:
            actual = shape[0]
            orig = text_config.get('intermediate_size')
            if orig and actual != orig:
                print(f"[fix_config] text intermediate_size: {orig} -> {actual}")
                text_config['intermediate_size'] = actual
                changed = True
            break

    # Attention: num_attention_heads from q_proj [num_heads*head_dim, hidden_size]
    head_dim = text_config.get('head_dim', 256)
    for key, shape in weight_shapes.items():
        if 'text_decoder' in key and '.layers.0.self_attn.q_proj.weight' in key:
            actual_q = shape[0]
            if actual_q % head_dim == 0:
                actual_heads = actual_q // head_dim
                orig_heads = text_config.get('num_attention_heads')
                if orig_heads and actual_heads != orig_heads:
                    print(f"[fix_config] num_attention_heads: {orig_heads} -> {actual_heads}")
                    text_config['num_attention_heads'] = actual_heads
                    changed = True
            else:
                print(f"[fix_config] WARNING: q_proj dim {actual_q} not divisible by head_dim {head_dim}")
            break

    # KV heads from k_proj [num_kv_heads*head_dim, hidden_size]
    for key, shape in weight_shapes.items():
        if 'text_decoder' in key and '.layers.0.self_attn.k_proj.weight' in key:
            actual_k = shape[0]
            if actual_k % head_dim == 0:
                actual_kv = actual_k // head_dim
                orig_kv = text_config.get('num_key_value_heads')
                if orig_kv and actual_kv != orig_kv:
                    print(f"[fix_config] num_key_value_heads: {orig_kv} -> {actual_kv}")
                    text_config['num_key_value_heads'] = actual_kv
                    changed = True
            else:
                print(f"[fix_config] WARNING: k_proj dim {actual_k} not divisible by head_dim {head_dim}")
            break

    # Remove midblock fields (all layers are uniform after full-layer pruning)
    for field in ['midblock_ratio', 'midblock_start', 'midblock_end']:
        if field in text_config:
            del text_config[field]
            changed = True

    config['text_config'] = text_config

    # --- Fix speech_config (Whisper encoder) if pruned ---
    speech_config = config.get("speech_config", {})

    # Whisper MLP: encoder_ffn_dim from fc1 [encoder_ffn_dim, d_model]
    for key, shape in weight_shapes.items():
        if 'speech_encoder' in key and '.encoder.layers.0.fc1.weight' in key:
            actual_ffn = shape[0]
            orig_ffn = speech_config.get('encoder_ffn_dim')
            if orig_ffn and actual_ffn != orig_ffn:
                print(f"[fix_config] whisper encoder_ffn_dim: {orig_ffn} -> {actual_ffn}")
                speech_config['encoder_ffn_dim'] = actual_ffn
                changed = True
            break

    # Whisper attention: d_model from q_proj [d_model, d_model]
    for key, shape in weight_shapes.items():
        if 'speech_encoder' in key and '.encoder.layers.0.self_attn.q_proj.weight' in key:
            actual_d = shape[0]
            orig_d = speech_config.get('d_model')
            if orig_d and actual_d != orig_d:
                print(f"[fix_config] whisper d_model: {orig_d} -> {actual_d}")
                speech_config['d_model'] = actual_d
                # Also update encoder_attention_heads if needed
                orig_head_dim = orig_d // speech_config.get('encoder_attention_heads', 20)
                if actual_d % orig_head_dim == 0:
                    actual_enc_heads = actual_d // orig_head_dim
                    print(f"[fix_config] whisper encoder_attention_heads: {speech_config.get('encoder_attention_heads')} -> {actual_enc_heads}")
                    speech_config['encoder_attention_heads'] = actual_enc_heads
                changed = True
            break

    # Remove whisper midblock fields
    for field in ['whisper_midblock_ratio', 'whisper_midblock_start', 'whisper_midblock_end']:
        if field in speech_config:
            del speech_config[field]
            changed = True

    config['speech_config'] = speech_config

    if changed:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[fix_config] Config updated for vLLM compatibility: {config_path}")
    else:
        print(f"[fix_config] Config already correct, no changes needed")


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

    # Fix config.json dimensions for vLLM compatibility
    if args.save_path is not None:
        fix_config_for_vllm(args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    # Model Type&Path
    # parser.add_argument('--base_model', type=str, default="/home/kaixin/programs/MiniCPM-checkpoints/MiniCPM-2B-128k", help='base model name')
    parser.add_argument('--ckpt', type=str, default='./MiniCPM-checkpoints/MiniCPM-2B-128k-pruned-bl-0.3-taylor', help='pruned model path')
    parser.add_argument('--lora_ckpt', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--fix_config', type=str, default=None, help='Fix config.json for an existing merged model (no merge)')
    parser.add_argument('--push-to-hub', type=str, default=None, help='Push the model to HuggingFace Hub')

    args = parser.parse_args()

    if args.fix_config:
        # Standalone mode: just fix config.json for an existing merged model
        fix_config_for_vllm(args.fix_config)
    else:
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


    
