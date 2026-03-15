"""Merge a LoRA adapter into its pruned MERaLiON-2 base model and save a full
HuggingFace model that vLLM can load directly.

Usage:
    python merge_lora.py \
        --base    meralion_checkpoints/MERaLiON-2-3B-v3-td50-mid3-22 \
        --adapter meralion_tune_log/MERaLiON-2-3B-v3-td50-mid3-22-tune \
        --output  meralion_tune_log/MERaLiON-2-3B-v3-td50-mid3-22-tune

--output may be the same path as --adapter; the merged model files will
overwrite the adapter-only files in place.
"""
import argparse
import json
import os
import shutil
import sys

import torch
from peft import PeftModel
from transformers import AutoProcessor

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into pruned MERaLiON-2 base model")
    parser.add_argument("--base",    required=True, help="Pruned base model directory (contains config.json)")
    parser.add_argument("--adapter", required=True, help="LoRA adapter directory (post-training output)")
    parser.add_argument("--output",  required=True, help="Destination for the merged full model")
    args = parser.parse_args()

    args.base    = os.path.abspath(args.base)
    args.adapter = os.path.abspath(args.adapter)
    args.output  = os.path.abspath(args.output)

    print(f"Base model  : {args.base}")
    print(f"LoRA adapter: {args.adapter}")
    print(f"Output dir  : {args.output}")

    from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration

    print("\nLoading base model (CPU)...")
    base_model = MERaLiON2ForConditionalGeneration.from_pretrained(
        args.base,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print("Loading LoRA adapter...")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter)

    print("Merging LoRA weights...")
    merged = peft_model.merge_and_unload()

    # Restore use_cache=True (was disabled during training for gradient checkpointing).
    # Must be set on both config and generation_config — the latter is validated on save.
    merged.config.use_cache = True
    if hasattr(merged, "generation_config"):
        merged.generation_config.use_cache = True
    if hasattr(merged, "text_decoder"):
        merged.text_decoder.config.use_cache = True
        if hasattr(merged.text_decoder, "generation_config"):
            merged.text_decoder.generation_config.use_cache = True

    os.makedirs(args.output, exist_ok=True)
    print(f"Saving merged model to {args.output} ...")
    merged.save_pretrained(args.output)

    # save_pretrained rewrites config.json and may corrupt auto_map (e.g. set
    # it to an empty string or a package reference that HuggingFace treats as a
    # remote repo ID).  Restore auto_map from the base config so AutoModel can
    # find the local .py files we copy below.
    base_cfg_path = os.path.join(args.base, "config.json")
    out_cfg_path  = os.path.join(args.output, "config.json")
    if os.path.exists(base_cfg_path):
        with open(base_cfg_path) as f:
            base_cfg = json.load(f)
        with open(out_cfg_path) as f:
            out_cfg = json.load(f)
        if "auto_map" in base_cfg:
            out_cfg["auto_map"] = base_cfg["auto_map"]
            with open(out_cfg_path, "w") as f:
                json.dump(out_cfg, f, indent=2)
            print(f"  restored auto_map from base config: {base_cfg['auto_map']}")

    print("Saving processor/tokenizer...")
    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    processor.save_pretrained(args.output)

    # Copy custom model code files from base (needed for trust_remote_code).
    # save_pretrained writes weights/config but not the *.py files that
    # AutoConfig/AutoModel look up when auto_map points to them.
    print("Copying custom model code files from base...")
    for fname in os.listdir(args.base):
        if fname.endswith(".py"):
            src = os.path.join(args.base, fname)
            dst = os.path.join(args.output, fname)
            shutil.copy2(src, dst)
            print(f"  copied {fname}")

    print("\nDone — directory now contains a full model loadable by vLLM.")


if __name__ == "__main__":
    main()
