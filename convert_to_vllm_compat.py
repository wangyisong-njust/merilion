"""
Convert pruned MERaLiON-2 merged models to vLLM-compatible format.

Problem: Pruned models have non-uniform layer dimensions (midblock system).
  - Layers 0-3: original size (unpruned)
  - Layers 4-23: pruned size
  vLLM requires ALL layers to have the same dimensions.

Solution: Truncate unpruned layers (0-3, 24-25) to match pruned layers,
  then update config.json with uniform dimensions.

Usage:
  python convert_to_vllm_compat.py --input <merged_model_dir> --output <vllm_model_dir>
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from collections import defaultdict

import torch
from safetensors.torch import load_file, save_file


def analyze_dimensions(state_dict):
    """Analyze actual dimensions per layer for text decoder and whisper encoder."""
    text_mlp_sizes = {}  # layer_idx -> intermediate_size
    text_attn_q_sizes = {}  # layer_idx -> q_proj out_features
    text_attn_kv_sizes = {}  # layer_idx -> kv_proj out_features
    whisper_attn_sizes = {}  # layer_idx -> (q_size, kv_size)
    whisper_mlp_sizes = {}  # layer_idx -> ffn_dim

    for key, tensor in state_dict.items():
        # Text decoder MLP
        if 'text_decoder.model.layers.' in key and '.mlp.gate_proj.weight' in key:
            layer_idx = int(key.split('layers.')[1].split('.')[0])
            text_mlp_sizes[layer_idx] = tensor.shape[0]

        # Text decoder attention
        if 'text_decoder.model.layers.' in key and '.self_attn.q_proj.weight' in key:
            layer_idx = int(key.split('layers.')[1].split('.')[0])
            text_attn_q_sizes[layer_idx] = tensor.shape[0]

        if 'text_decoder.model.layers.' in key and '.self_attn.k_proj.weight' in key:
            layer_idx = int(key.split('layers.')[1].split('.')[0])
            text_attn_kv_sizes[layer_idx] = tensor.shape[0]

        # Whisper encoder attention
        if 'speech_encoder.encoder.layers.' in key and '.self_attn.q_proj.weight' in key:
            layer_idx = int(key.split('layers.')[1].split('.')[0])
            whisper_attn_sizes[layer_idx] = tensor.shape[0]

        # Whisper encoder MLP
        if 'speech_encoder.encoder.layers.' in key and '.fc1.weight' in key:
            layer_idx = int(key.split('layers.')[1].split('.')[0])
            whisper_mlp_sizes[layer_idx] = tensor.shape[0]

    return text_mlp_sizes, text_attn_q_sizes, text_attn_kv_sizes, whisper_attn_sizes, whisper_mlp_sizes


def truncate_linear_weight(tensor, target_out, target_in=None):
    """Truncate a weight matrix to target dimensions (keep first N rows/cols)."""
    if target_in is not None:
        return tensor[:target_out, :target_in].contiguous()
    return tensor[:target_out].contiguous()


def convert_model(input_dir, output_dir):
    print(f"Loading model from: {input_dir}")

    # Load all safetensors
    safetensor_files = sorted(Path(input_dir).glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {input_dir}")

    state_dict = {}
    for f in safetensor_files:
        state_dict.update(load_file(str(f)))

    print(f"Loaded {len(state_dict)} tensors")

    # Analyze dimensions
    text_mlp_sizes, text_attn_q_sizes, text_attn_kv_sizes, whisper_attn_sizes, whisper_mlp_sizes = analyze_dimensions(state_dict)

    # Find target (minimum) dimensions — these are the pruned layer sizes
    if text_mlp_sizes:
        target_mlp = min(text_mlp_sizes.values())
        print(f"\nText MLP sizes: min={target_mlp}, max={max(text_mlp_sizes.values())}")
        for idx in sorted(text_mlp_sizes.keys()):
            if text_mlp_sizes[idx] != target_mlp:
                print(f"  Layer {idx}: {text_mlp_sizes[idx]} -> {target_mlp} (will truncate)")

    if text_attn_q_sizes:
        target_q = min(text_attn_q_sizes.values())
        target_kv = min(text_attn_kv_sizes.values())
        print(f"\nText Attention Q sizes: min={target_q}, max={max(text_attn_q_sizes.values())}")
        print(f"Text Attention KV sizes: min={target_kv}, max={max(text_attn_kv_sizes.values())}")

    if whisper_attn_sizes:
        target_whisper_attn = min(whisper_attn_sizes.values())
        print(f"\nWhisper Attention sizes: min={target_whisper_attn}, max={max(whisper_attn_sizes.values())}")

    if whisper_mlp_sizes:
        target_whisper_mlp = min(whisper_mlp_sizes.values())
        print(f"Whisper MLP sizes: min={target_whisper_mlp}, max={max(whisper_mlp_sizes.values())}")

    # Truncate weights to uniform dimensions
    new_state_dict = {}
    modified_count = 0

    for key, tensor in state_dict.items():
        new_tensor = tensor

        # Text decoder MLP
        if 'text_decoder.model.layers.' in key:
            layer_idx = int(key.split('layers.')[1].split('.')[0])

            if '.mlp.gate_proj.weight' in key and text_mlp_sizes.get(layer_idx, target_mlp) > target_mlp:
                new_tensor = tensor[:target_mlp, :]
                modified_count += 1
            elif '.mlp.up_proj.weight' in key and text_mlp_sizes.get(layer_idx, target_mlp) > target_mlp:
                new_tensor = tensor[:target_mlp, :]
                modified_count += 1
            elif '.mlp.down_proj.weight' in key and text_mlp_sizes.get(layer_idx, target_mlp) > target_mlp:
                new_tensor = tensor[:, :target_mlp]
                modified_count += 1

            # Text decoder Attention
            elif '.self_attn.q_proj.weight' in key and text_attn_q_sizes.get(layer_idx, target_q) > target_q:
                new_tensor = tensor[:target_q, :]
                modified_count += 1
            elif '.self_attn.k_proj.weight' in key and text_attn_kv_sizes.get(layer_idx, target_kv) > target_kv:
                new_tensor = tensor[:target_kv, :]
                modified_count += 1
            elif '.self_attn.v_proj.weight' in key and text_attn_kv_sizes.get(layer_idx, target_kv) > target_kv:
                new_tensor = tensor[:target_kv, :]
                modified_count += 1
            elif '.self_attn.o_proj.weight' in key and text_attn_q_sizes.get(layer_idx, target_q) > target_q:
                new_tensor = tensor[:, :target_q]
                modified_count += 1

        # Whisper encoder
        elif 'speech_encoder.encoder.layers.' in key:
            layer_idx = int(key.split('layers.')[1].split('.')[0])

            if '.self_attn.q_proj' in key and whisper_attn_sizes.get(layer_idx, target_whisper_attn) > target_whisper_attn:
                if '.weight' in key:
                    new_tensor = tensor[:target_whisper_attn, :]
                elif '.bias' in key:
                    new_tensor = tensor[:target_whisper_attn]
                modified_count += 1
            elif '.self_attn.k_proj' in key and whisper_attn_sizes.get(layer_idx, target_whisper_attn) > target_whisper_attn:
                if '.weight' in key:
                    new_tensor = tensor[:target_whisper_attn, :]
                elif '.bias' in key:
                    new_tensor = tensor[:target_whisper_attn]
                modified_count += 1
            elif '.self_attn.v_proj' in key and whisper_attn_sizes.get(layer_idx, target_whisper_attn) > target_whisper_attn:
                if '.weight' in key:
                    new_tensor = tensor[:target_whisper_attn, :]
                elif '.bias' in key:
                    new_tensor = tensor[:target_whisper_attn]
                modified_count += 1
            elif '.self_attn.out_proj' in key and whisper_attn_sizes.get(layer_idx, target_whisper_attn) > target_whisper_attn:
                if '.weight' in key:
                    new_tensor = tensor[:, :target_whisper_attn]
                elif '.bias' in key:
                    pass  # out_proj bias is hidden_size, not pruned
                modified_count += 1

            elif '.fc1' in key and whisper_mlp_sizes.get(layer_idx, target_whisper_mlp) > target_whisper_mlp:
                if '.weight' in key:
                    new_tensor = tensor[:target_whisper_mlp, :]
                elif '.bias' in key:
                    new_tensor = tensor[:target_whisper_mlp]
                modified_count += 1
            elif '.fc2' in key and whisper_mlp_sizes.get(layer_idx, target_whisper_mlp) > target_whisper_mlp:
                if '.weight' in key:
                    new_tensor = tensor[:, :target_whisper_mlp]
                elif '.bias' in key:
                    pass  # fc2 bias is d_model, not pruned
                modified_count += 1

        if new_tensor.shape != tensor.shape:
            print(f"  Truncated {key}: {tuple(tensor.shape)} -> {tuple(new_tensor.shape)}")

        new_state_dict[key] = new_tensor.contiguous()

    print(f"\nModified {modified_count} tensors")

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "model.safetensors")
    print(f"Saving to {output_file}...")
    save_file(new_state_dict, output_file)

    # Copy and update config.json
    config_path = os.path.join(input_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    head_dim = config.get("text_config", config).get("head_dim", 256)

    # Update text_config
    if "text_config" in config:
        tc = config["text_config"]
        if text_mlp_sizes:
            tc["intermediate_size"] = target_mlp
        if text_attn_q_sizes:
            tc["num_attention_heads"] = target_q // head_dim
        if text_attn_kv_sizes:
            tc["num_key_value_heads"] = target_kv // head_dim
        # Remove midblock config (no longer needed, all layers uniform)
        tc["midblock_ratio"] = 1.0
        tc["midblock_start"] = -1
        tc["midblock_end"] = -1
        tc.pop("text_mlp_midblock_ratio", None)

    # Update top-level fields too
    if text_mlp_sizes:
        config["intermediate_size"] = target_mlp
    if text_attn_q_sizes:
        config["num_attention_heads"] = target_q // head_dim
    if text_attn_kv_sizes:
        config["num_key_value_heads"] = target_kv // head_dim
    config["midblock_ratio"] = 1.0
    config["midblock_start"] = -1
    config["midblock_end"] = -1

    # Update speech_config
    if "speech_config" in config:
        sc = config["speech_config"]
        if whisper_attn_sizes:
            whisper_head_dim = sc.get("d_model", 1280) // sc.get("encoder_attention_heads", 20)
            sc["encoder_attention_heads"] = target_whisper_attn // whisper_head_dim
            sc["d_model"] = target_whisper_attn  # d_model = num_heads * head_dim
        if whisper_mlp_sizes:
            sc["encoder_ffn_dim"] = target_whisper_mlp
        sc["whisper_midblock_start"] = -1
        sc["whisper_midblock_end"] = -1
        sc.pop("whisper_attn_midblock_ratio", None)
        sc.pop("whisper_mlp_midblock_ratio", None)

    out_config_path = os.path.join(output_dir, "config.json")
    with open(out_config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Updated config saved to {out_config_path}")

    # Copy all other files (tokenizer, processor, modeling code, etc.)
    for item in Path(input_dir).iterdir():
        if item.name.endswith('.safetensors') or item.name == 'config.json':
            continue
        dst = Path(output_dir) / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)
    print("Copied remaining files")

    print(f"\nDone! vLLM-compatible model saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to merged pruned model")
    parser.add_argument("--output", required=True, help="Output path for vLLM-compatible model")
    args = parser.parse_args()
    convert_model(args.input, args.output)
