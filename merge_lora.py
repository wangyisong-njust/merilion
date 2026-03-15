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


def fix_auto_map(model_dir):
    """Reconstruct auto_map in config.json from local .py files.

    LLM-Pruner and transformers save_pretrained can corrupt auto_map values
    to empty strings.  We scan .py files to find the right module.ClassName:
      - "AutoConfig" keys → Config class (arch_name + "Config")
      - "AutoModel*" keys → model class (architectures[0])
    """
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        return
    with open(cfg_path) as f:
        cfg = json.load(f)

    auto_map = cfg.get("auto_map", {})
    architectures = cfg.get("architectures", [])

    print(f"  [fix_auto_map] {os.path.basename(model_dir)}: current auto_map={auto_map}", flush=True)

    bad = not auto_map or any(not v or "." not in v for v in auto_map.values())
    if not bad:
        print(f"  [fix_auto_map] auto_map looks valid, skipping", flush=True)
        return

    if not architectures:
        print(f"  [fix_auto_map] WARNING: no architectures in config", flush=True)
        return

    arch_class = architectures[0]
    model_stem = arch_class.split("For")[0] if "For" in arch_class else arch_class
    config_class = f"{model_stem}Config"

    def find_module(class_name):
        for fname in sorted(os.listdir(model_dir)):
            if not fname.endswith(".py"):
                continue
            with open(os.path.join(model_dir, fname)) as f:
                if f"class {class_name}" in f.read():
                    return fname[:-3]
        return None

    model_module = find_module(arch_class)
    config_module = find_module(config_class) or model_module

    if model_module is None:
        print(f"  [fix_auto_map] WARNING: no .py defines class {arch_class}", flush=True)
        return

    keys_to_set = list(auto_map.keys()) or ["AutoModelForSpeechSeq2Seq"]
    new_auto_map = {}
    for key in keys_to_set:
        if "Config" in key:
            new_auto_map[key] = f"{config_module}.{config_class}"
        else:
            new_auto_map[key] = f"{model_module}.{arch_class}"
    if "AutoModelForSpeechSeq2Seq" not in new_auto_map:
        new_auto_map["AutoModelForSpeechSeq2Seq"] = f"{model_module}.{arch_class}"

    cfg["auto_map"] = new_auto_map
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  [fix_auto_map] set auto_map -> {new_auto_map}", flush=True)
    print(f"  [fix_auto_map] set auto_map -> {new_auto_map}")


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
        changed = {}
        for key in ("architectures", "auto_map"):
            if key in base_cfg:
                out_cfg[key] = base_cfg[key]
                changed[key] = base_cfg[key]
        if changed:
            with open(out_cfg_path, "w") as f:
                json.dump(out_cfg, f, indent=2)
            for k, v in changed.items():
                print(f"  restored {k} from base config: {v}")

    # Reconstruct auto_map from local .py files in case the base config
    # also had a corrupted auto_map (e.g. LLM-Pruner called save_pretrained).
    fix_auto_map(args.output)

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
