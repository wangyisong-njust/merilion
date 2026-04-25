"""Quantize MERaLiON-2-3B's text_decoder (Gemma2) with GPTQModel into a
W4A16 checkpoint that HF transformers loads with the Marlin kernel.

Output layout (vLLM-detached, HF-transformers-friendly):
    <out>/
      config.json                    # MERaLiON2Config + quantization_config (gptq + use_marlin)
      tokenizer.* / processor.*
      *.safetensors                  # speech_encoder + audio_adapter (bf16) +
                                     # text_decoder (qweight/qzeros/scales)

The final dir can be loaded by HF transformers with our existing
`MERaLiON2ForConditionalGeneration.from_pretrained()` patched dispatcher
(quantization_config detected → super().from_pretrained handles
GPTQConfig(use_marlin=True)).

Calibration data: short text-only prompts.  RTN-equivalent (bits=4,
desc_act=False) doesn't depend strongly on calibration distribution,
so text calibration is fine — that's why our existing RTN W4A16 works.

Usage:
  python quantize_gptq_marlin.py \\
      --src     /path/to/MERaLiON-2-3B \\
      --out     quant_checkpoints/MERaLiON-2-3B-W4A16-GPTQ-Marlin \\
      --bits    4 --group_size 128 --n_calib 128
"""
import argparse
import json
import os
import shutil
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--n_calib", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print("[1/5] Loading MERaLiON-2-3B (bf16) …")
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.src, trust_remote_code=True)
    full_model = MERaLiON2ForConditionalGeneration.from_pretrained(
        args.src, torch_dtype=torch.bfloat16, use_safetensors=True)
    full_model.eval()

    # We only quantize text_decoder (the Gemma2 stack).  Pull it out as a
    # standalone Gemma2ForCausalLM that GPTQModel can ingest, run the quantize,
    # then reassemble with the original speech_encoder/adapter weights.
    print("[2/5] Building standalone Gemma2 text_decoder for quantization …")
    text_decoder = full_model.text_decoder
    print(f"  text_decoder type: {type(text_decoder).__name__}")
    print(f"  text_decoder params: {sum(p.numel() for p in text_decoder.parameters())/1e9:.2f} B")

    # Move text_decoder to GPU for fast quantization
    text_decoder = text_decoder.to(args.device)

    # Build calibration data — short generic text.  GPTQ at low rank with
    # group_size=128 is robust to the calibration distribution; we mostly
    # just need real activation magnitudes to anchor scales.
    print(f"[3/5] Building {args.n_calib} text calibration samples …")
    tokenizer = processor.tokenizer
    calib_texts = [
        "Transcribe the speech: hello world, this is a test of the audio model.",
        "The quick brown fox jumps over the lazy dog.",
        "ASR systems need to handle a wide variety of accents and speaking styles.",
        "Singapore English includes vocabulary borrowed from Malay and Hokkien.",
        "Recurrent neural networks learn to model temporal dependencies in speech.",
    ] * (args.n_calib // 5 + 1)
    calib_texts = calib_texts[:args.n_calib]

    # auto-gptq API
    print("[4/5] Running auto-gptq quantization (marlin-compatible format) …")
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    qcfg = BaseQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=False,         # required for Marlin
        sym=True,               # required for Marlin
    )

    # auto-gptq loads via from_pretrained on a model dir.  We first save the
    # text_decoder as a standalone Gemma2 dir, then load + quantize there.
    print("  saving text_decoder to a temp Gemma2 dir for auto-gptq …")
    tmp_gemma2 = os.path.join(args.out + "_tmp_gemma2")
    os.makedirs(tmp_gemma2, exist_ok=True)

    with open(os.path.join(args.src, "config.json")) as f:
        meralion_cfg = json.load(f)
    gemma2_cfg = dict(meralion_cfg["text_config"])
    gemma2_cfg["architectures"] = ["Gemma2ForCausalLM"]
    gemma2_cfg.setdefault("model_type", "gemma2")
    gemma2_cfg["torch_dtype"] = "bfloat16"
    with open(os.path.join(tmp_gemma2, "config.json"), "w") as f:
        json.dump(gemma2_cfg, f, indent=2)

    from safetensors.torch import save_file
    sd = {k: v.detach().cpu().contiguous() for k, v in text_decoder.state_dict().items()}
    save_file(sd, os.path.join(tmp_gemma2, "model.safetensors"),
              metadata={"format": "pt"})

    for f in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
              "special_tokens_map.json", "generation_config.json"):
        sp = os.path.join(args.src, f)
        if os.path.exists(sp):
            shutil.copy2(sp, os.path.join(tmp_gemma2, f))

    # Free GPU mem before auto-gptq re-loads
    del text_decoder
    torch.cuda.empty_cache()

    # Build calibration examples (auto-gptq wants list of dicts with
    # input_ids + attention_mask, or list of token-id lists).
    calib_examples = []
    for txt in calib_texts:
        enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=256)
        calib_examples.append({
            "input_ids":      enc.input_ids,
            "attention_mask": enc.attention_mask,
        })

    t0 = time.time()
    qmodel = AutoGPTQForCausalLM.from_pretrained(
        tmp_gemma2, quantize_config=qcfg, torch_dtype=torch.bfloat16,
        trust_remote_code=False)
    qmodel.quantize(calib_examples, batch_size=1, use_triton=True)
    print(f"  quantize took {time.time()-t0:.1f}s")

    # Save in safetensors format.  This writes a quantize_config.json + the
    # quantized .safetensors with qweight/qzeros/scales — the layout HF
    # transformers' GPTQConfig + auto-gptq's marlin kernel both understand.
    qmodel.save_quantized(args.out, use_safetensors=True)
    del qmodel
    torch.cuda.empty_cache()

    # ---- Reassemble: copy speech_encoder/adapter (still bf16) into args.out ----
    print("[5/5] Reassembling MERaLiON wrapper …")
    # The quantized save in args.out has Gemma2 keys (no "text_decoder." prefix);
    # we need to add the prefix back AND merge in the bf16 non-text-decoder
    # weights from the original MERaLiON checkpoint.
    from safetensors import safe_open

    # 1) Re-load the just-saved quantized weights and prepend "text_decoder."
    import glob
    quant_sf = sorted(glob.glob(os.path.join(args.out, "*.safetensors")))
    merged = {}
    for sf in quant_sf:
        with safe_open(sf, framework="pt") as f:
            for k in f.keys():
                merged[f"text_decoder.{k}"] = f.get_tensor(k)
    print(f"  reprefixed {len(merged)} text_decoder keys")

    # 2) Merge in the bf16 non-text-decoder weights from full_model (already in memory)
    n_added = 0
    for k, v in full_model.state_dict().items():
        if not k.startswith("text_decoder."):
            merged[k] = v.detach().cpu().contiguous()
            n_added += 1
    print(f"  added {n_added} non-text-decoder bf16 keys (speech_encoder/adapter/etc.)")

    # Write merged shard, replacing the gemma2-only files
    for sf in quant_sf:
        os.remove(sf)
    save_file(merged, os.path.join(args.out, "model.safetensors"),
              metadata={"format": "pt"})

    # 3) Patch config.json in args.out.  auto-gptq writes a separate
    #    quantize_config.json (not embedded in config.json), so we build
    #    quantization_config from that + the saved gptq metadata.
    final_cfg = dict(meralion_cfg)
    qc_path = os.path.join(args.out, "quantize_config.json")
    if os.path.exists(qc_path):
        with open(qc_path) as f:
            qcfg_dict = json.load(f)
    else:
        qcfg_dict = {"bits": args.bits, "group_size": args.group_size,
                     "desc_act": False, "sym": True}
    qc = {
        "quant_method":              "gptq",
        "bits":                      qcfg_dict["bits"],
        "group_size":                qcfg_dict["group_size"],
        "desc_act":                  qcfg_dict["desc_act"],
        "sym":                       qcfg_dict["sym"],
        "use_marlin":                True,
        "modules_to_not_convert":    ["speech_encoder", "speech_audio_adapter", "lm_head"],
    }
    final_cfg["quantization_config"] = qc
    final_cfg["torch_dtype"] = "bfloat16"
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(final_cfg, f, indent=2)

    # 4) Copy MERaLiON's processor + custom modeling files
    for f in ("preprocessor_config.json", "processing_meralion2.py",
              "configuration_meralion2.py", "modeling_meralion2.py",
              "chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
              "tokenizer.model", "special_tokens_map.json", "generation_config.json"):
        sp = os.path.join(args.src, f)
        if os.path.exists(sp):
            shutil.copy2(sp, os.path.join(args.out, f))

    # Cleanup
    shutil.rmtree(tmp_gemma2, ignore_errors=True)

    print(f"\nSaved → {args.out}")
    print("Test loading:")
    print(f"  python -c \"from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration; "
          f"m=MERaLiON2ForConditionalGeneration.from_pretrained('{args.out}', torch_dtype='bfloat16'); print(type(m).__name__)\"")


if __name__ == "__main__":
    main()
