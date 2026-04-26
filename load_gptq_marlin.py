"""Load a MERaLiON-2-3B GPTQ-Marlin checkpoint without going through
HF transformers' GPTQQuantizer (which routes via optimum → gptqmodel →
pcre, blocked in this env).

Strategy:
  1. The saved checkpoint has speech_encoder + audio_adapter (bf16) plus
     text_decoder (gptq qweight/qzeros/scales/g_idx).  We extract just the
     text_decoder portion into a standalone Gemma2 dir (cached on disk).
  2. AutoGPTQForCausalLM.from_quantized(gemma2_dir, use_marlin=True) loads
     it as a Gemma2 model and auto-repacks qweight → marlin B/s format,
     dispatching forward to autogptq_marlin_cuda.mul.
  3. We graft the marlin Gemma2 module into a freshly-loaded bf16 MERaLiON
     wrapper (replacing its text_decoder).
"""
import glob
import json
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _extract_gemma2_dir(model_path, dst_dir):
    """Strip everything except text_decoder.* from `model_path` and write
    a standalone Gemma2 dir at `dst_dir`.  Idempotent: re-uses cache if
    dst_dir already has a model.safetensors."""
    if os.path.exists(os.path.join(dst_dir, "model.safetensors")):
        print(f"  cached: {dst_dir}")
        return
    os.makedirs(dst_dir, exist_ok=True)

    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)
    text_cfg = dict(cfg["text_config"])
    text_cfg["architectures"] = ["Gemma2ForCausalLM"]
    text_cfg.setdefault("model_type", "gemma2")
    text_cfg["torch_dtype"] = "bfloat16"

    # Carry the gptq quantization_config across.  AutoGPTQForCausalLM
    # actually reads quantize_config.json (not config.json's quantization_config)
    # so we also write that.
    qc = cfg.get("quantization_config")
    if qc is None:
        raise SystemExit(f"{model_path}/config.json has no quantization_config")
    text_cfg["quantization_config"] = qc
    with open(os.path.join(dst_dir, "config.json"), "w") as f:
        json.dump(text_cfg, f, indent=2)

    qcfg_for_autogptq = {
        "bits":           qc["bits"],
        "group_size":     qc["group_size"],
        "damp_percent":   0.01,
        "desc_act":       qc.get("desc_act", False),
        "sym":            qc.get("sym", True),
        "true_sequential": True,
        "model_name_or_path": None,
        "model_file_base_name": "model",
    }
    with open(os.path.join(dst_dir, "quantize_config.json"), "w") as f:
        json.dump(qcfg_for_autogptq, f, indent=2)

    # Strip "text_decoder." prefix from each key, drop everything else.
    sd_out = {}
    for sf in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(sf, framework="pt") as f:
            for k in f.keys():
                if k.startswith("text_decoder."):
                    new_k = k[len("text_decoder."):]
                    sd_out[new_k] = f.get_tensor(k)
    save_file(sd_out, os.path.join(dst_dir, "model.safetensors"),
              metadata={"format": "pt"})

    # Tokenizer / generation_config (auto-gptq doesn't strictly need them,
    # but having them around helps).
    for fname in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                  "special_tokens_map.json", "generation_config.json"):
        sp = os.path.join(model_path, fname)
        if os.path.exists(sp):
            shutil.copy2(sp, os.path.join(dst_dir, fname))


def _patch_autogptq_for_gemma2():
    """Register gemma2 in auto-gptq's SUPPORTED_MODELS (same as quantize)."""
    from auto_gptq.modeling._const import SUPPORTED_MODELS
    if "gemma2" not in SUPPORTED_MODELS:
        SUPPORTED_MODELS.append("gemma2")
    from auto_gptq.modeling.auto import GPTQ_CAUSAL_LM_MODEL_MAP
    from auto_gptq.modeling.gemma import GemmaGPTQForCausalLM
    GPTQ_CAUSAL_LM_MODEL_MAP["gemma2"] = GemmaGPTQForCausalLM


def load_meralion2_gptq_marlin(model_path, bf16_path, device,
                                dtype=torch.float16, cache_dir=None,
                                kernel="exllamav2"):
    """Load a MERaLiON-2-3B with marlin-quantized text_decoder.

    Args:
        model_path:  dir produced by quantize_gptq_marlin.py
        bf16_path:   original bf16 MERaLiON-2-3B (source of speech_encoder /
                     audio_adapter / processor weights).  These come back as
                     bf16 in the final model.
        device:      target device
        cache_dir:   where to write the extracted standalone Gemma2 dir.
                     Default = model_path + "_gemma2_only".
        dtype:       must be torch.float16 — all auto-gptq W4A16 kernels
                     (marlin / exllama / exllamav2) require fp16 I/O.
                     The rest of MERaLiON loads in fp16 too.
        kernel:      "marlin" | "exllama" | "exllamav2"  (default exllamav2,
                     fastest at batch=1 decode which is what EAGLE's draft
                     and most-of-the-time verifier do).
    Returns: (model, processor)
    """
    if dtype != torch.float16:
        raise ValueError(
            "auto-gptq W4A16 kernels require dtype=torch.float16 "
            f"(got {dtype})")
    if kernel not in ("marlin", "exllama", "exllamav2"):
        raise ValueError(f"unknown kernel: {kernel}")
    from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration
    from transformers import AutoProcessor

    _patch_autogptq_for_gemma2()
    from auto_gptq import AutoGPTQForCausalLM

    # 1) Extract text_decoder slice for auto-gptq
    gemma2_dir = cache_dir or (model_path.rstrip("/") + "_gemma2_only")
    print(f"[1/3] Extracting text_decoder → {gemma2_dir}")
    _extract_gemma2_dir(model_path, gemma2_dir)

    # 2) Load with auto-gptq, dispatching to the chosen kernel.
    if kernel == "marlin":
        from_q_kw = dict(use_marlin=True)
    elif kernel == "exllamav2":
        # auto-gptq picks exllamav2 by default unless disabled
        from_q_kw = dict(disable_exllama=True, disable_exllamav2=False)
    elif kernel == "exllama":
        from_q_kw = dict(disable_exllama=False, disable_exllamav2=True)
    else:
        from_q_kw = {}
    print(f"[2/3] Loading Gemma2 W4A16 via auto-gptq (kernel={kernel}) …")
    t0 = time.time()
    qmodel = AutoGPTQForCausalLM.from_quantized(
        gemma2_dir,
        torch_dtype=dtype,
        trust_remote_code=False,
        **from_q_kw,
    )
    qmodel = qmodel.to(device)
    print(f"  loaded in {time.time()-t0:.1f}s "
          f"(VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB)")

    # 3) Load bf16 MERaLiON shell, swap in marlin text_decoder
    print(f"[3/3] Loading bf16 MERaLiON wrapper for speech_encoder/adapter …")
    processor = AutoProcessor.from_pretrained(bf16_path, trust_remote_code=True)
    model = MERaLiON2ForConditionalGeneration.from_pretrained(
        bf16_path, torch_dtype=dtype, use_safetensors=True)
    # AutoGPTQForCausalLM wraps the actual nn.Module in qmodel.model
    marlin_text_decoder = qmodel.model
    marlin_text_decoder = marlin_text_decoder.to(device)
    model.text_decoder = marlin_text_decoder
    model = model.to(device)

    print(f"  done. final VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    return model, processor


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",     required=True, help="GPTQ-Marlin MERaLiON dir")
    ap.add_argument("--bf16_path", required=True, help="Original bf16 MERaLiON dir")
    ap.add_argument("--kernel", default="exllamav2",
                    choices=["marlin", "exllama", "exllamav2"])
    args = ap.parse_args()

    m, p = load_meralion2_gptq_marlin(args.model, args.bf16_path, "cuda",
                                       kernel=args.kernel)
    print("\n[OK] Loaded.")
    print(f"  text_decoder type: {type(m.text_decoder).__name__}")
    # Quick smoke forward on text-only input
    tok = p.tokenizer
    ids = tok("Hello, world.", return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        out = m.text_decoder(ids)
    print(f"  forward OK  logits shape: {out.logits.shape}")
