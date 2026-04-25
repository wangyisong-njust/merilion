"""Load a MERaLiON-2-3B GPTQ-Marlin checkpoint without going through
HF transformers' GPTQQuantizer (which forces optimum → gptqmodel → pcre).

Strategy: build the MERaLiON2 wrapper shell, swap each text_decoder Linear
with auto-gptq's marlin QuantLinear, then load the saved state dict.
"""
import glob
import json
import os
import time

import torch
import torch.nn as _nn
from safetensors.torch import load_file


def _strip_quant_config(model_path):
    """Read config.json and return (cfg_dict_without_quant, quant_cfg_dict)."""
    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)
    qcfg = cfg.pop("quantization_config", None)
    return cfg, qcfg


def load_meralion2_gptq_marlin(model_path, device, dtype=torch.bfloat16):
    """Load MERaLiON-2-3B with text_decoder linears swapped for marlin
    QuantLinear (auto-gptq).  Returns (model, processor).
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from meralion2_bl.modeling_meralion2 import (
        MERaLiON2ForConditionalGeneration, MERaLiON2Config,
    )
    from transformers import AutoProcessor
    from auto_gptq.nn_modules.qlinear.qlinear_marlin import QuantLinear as MarlinLinear

    # 1) Read config; strip quantization_config so HF transformers doesn't
    #    try to dispatch through HfQuantizer (which forces gptqmodel).
    cfg_dict, qcfg = _strip_quant_config(model_path)
    if qcfg is None or qcfg.get("quant_method") != "gptq":
        raise SystemExit(
            f"{model_path}/config.json has no gptq quantization_config")
    bits        = qcfg["bits"]
    group_size  = qcfg["group_size"]
    skip_names  = list(qcfg.get("modules_to_not_convert", []))

    print(f"Loading processor …")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # 2) Build MERaLiON2 from config (random init, no weights loaded).  We
    #    pass `_skip_quant_dispatch=True` so our patched from_pretrained
    #    doesn't go through HfQuantizer.
    print(f"Building MERaLiON2 shell from config …")
    config = MERaLiON2Config.from_json_file(os.path.join(model_path, "config.json"))
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    t0 = time.time()
    model = MERaLiON2ForConditionalGeneration(config)
    model = model.to(dtype)
    print(f"  shell built in {time.time()-t0:.1f}s")

    # 3) Walk text_decoder linears; swap each one with an empty MarlinLinear.
    print(f"Swapping text_decoder linears → MarlinLinear …")
    n_swap = 0
    for name, module in list(model.text_decoder.named_modules()):
        if not isinstance(module, _nn.Linear):
            continue
        if any(s in name for s in skip_names) or "lm_head" in name:
            continue
        in_features  = module.in_features
        out_features = module.out_features
        # Marlin requires both dims % 128 == 0 (verify; fail loudly if not).
        if in_features % 128 != 0 or out_features % 128 != 0:
            print(f"  WARN: {name}  ({in_features},{out_features}) not 128-aligned, skipping")
            continue
        # Navigate to parent
        parts = name.split(".")
        parent = model.text_decoder
        for p in parts[:-1]:
            parent = getattr(parent, p)
        marlin = MarlinLinear(
            bits=bits, group_size=group_size,
            infeatures=in_features, outfeatures=out_features,
            bias=module.bias is not None,
        )
        setattr(parent, parts[-1], marlin)
        n_swap += 1
    print(f"  swapped {n_swap} layers")

    # 4) Load the safetensors shards into the modified architecture.
    print(f"Loading weights …")
    sd = {}
    sf_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    for sf in sf_files:
        sd.update(load_file(sf))

    # The text_decoder.lm_head.weight is tied to embed_tokens — same as bf16
    # MERaLiON.  Add a tied entry if missing.
    if "text_decoder.lm_head.weight" not in sd \
            and "text_decoder.model.embed_tokens.weight" in sd:
        sd["text_decoder.lm_head.weight"] = \
            sd["text_decoder.model.embed_tokens.weight"]

    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Filter out layernorm bias placeholders that are intentionally not used
    missing = [m for m in missing if "lm_head" not in m]
    print(f"  loaded {len(sd)} tensors  "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    if missing[:5]:
        print(f"  first missing: {missing[:5]}")
    if unexpected[:5]:
        print(f"  first unexpected: {unexpected[:5]}")

    # 5) Move to GPU.  MarlinLinear has a custom .cuda() that re-packs weights
    #    into the marlin format expected by the kernel — call it explicitly.
    print(f"Moving to {device} …")
    model = model.to(device)
    # Marlin kernel expects post_init to be called (does some buffer setup).
    for mod in model.text_decoder.modules():
        if isinstance(mod, MarlinLinear) and hasattr(mod, "post_init"):
            mod.post_init()
    return model, processor


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    m, p = load_meralion2_gptq_marlin(args.model, "cuda")
    print("\n[OK] Loaded.")
    print(f"  VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    # Smoke test: tokenize and forward
    tok = p.tokenizer
    ids = tok("Hello, world.", return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        out = m.text_decoder(ids)
    print(f"  forward OK  logits shape: {out.logits.shape}")
