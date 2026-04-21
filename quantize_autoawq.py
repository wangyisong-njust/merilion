#!/usr/bin/env python3
"""
AutoAWQ W4A16 quantization for MERaLiON-2-3B text decoder.

AutoAWQ processes layer-by-layer so GPU memory stays manageable.
Speech encoder and audio adapter remain in FP16.

Saved artifacts (--save dir):
    non_td_weights.pt      speech encoder + adapter weights (FP16)
    text_decoder_awq/      AutoAWQ-quantized Gemma2 (safetensors + config)
    awq_config.json        quantization metadata
    *                      config/tokenizer files copied from original

Usage:
    python quantize_autoawq.py \\
        --model   /path/to/MERaLiON-2-3B \\
        --dataset /path/to/IMDA_PART1_mono_en_30_ASR \\
        --save    /path/to/MERaLiON-2-3B-AutoAWQ4 \\
        --num_calib 64 --q_group_size 128
"""

import argparse, json, os, shutil, sys, time
import torch


def get_calib_texts(dataset_path, num_calib):
    from datasets import load_from_disk
    data = load_from_disk(os.path.abspath(dataset_path))
    data = data.shuffle(seed=42)
    start = min(10500, len(data))
    pool  = data.select(range(start, min(start + num_calib * 4, len(data))))

    texts = []
    for sample in pool:
        if len(texts) >= num_calib:
            break
        try:
            # prefer ground-truth transcription; fall back to instruction text
            for key in ("answer", "response", "text"):
                val = sample.get(key, "")
                if isinstance(val, dict):
                    val = val.get("text", "")
                if val and len(str(val).strip()) > 15:
                    texts.append(str(val).strip())
                    break
            else:
                instr = sample.get("instruction", "")
                if isinstance(instr, dict):
                    instr = instr.get("text", "")
                if instr and len(str(instr).strip()) > 15:
                    texts.append(str(instr).strip())
        except Exception:
            continue

    return texts


def _patch_gemma2_forward():
    """Patch Gemma2Model.forward to propagate attention_type to AutoAWQ Catcher."""
    from transformers.models.gemma2 import modeling_gemma2 as _g2_mod
    _orig = _g2_mod.Gemma2Model.forward

    def _patched(self, *args, **kwargs):
        for layer in self.layers:
            if not hasattr(layer, "attention_type"):
                wrapped = getattr(layer, "module", None)
                if wrapped is not None and hasattr(wrapped, "attention_type"):
                    layer.attention_type = wrapped.attention_type
        return _orig(self, *args, **kwargs)

    _g2_mod.Gemma2Model.forward = _patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",        required=True, help="Path to MERaLiON-2-3B (original or pruned)")
    ap.add_argument("--dataset",      required=True, help="IMDA_PART1 dataset path")
    ap.add_argument("--save",         required=True, help="Output directory")
    ap.add_argument("--num_calib",    type=int, default=64,  help="Calibration samples (default 64)")
    ap.add_argument("--q_group_size", type=int, default=128, help="AWQ group size (default 128)")
    ap.add_argument("--pruned",       action="store_true",
                    help="Model has non-uniform pruned architecture (bypasses AutoAWQ from_pretrained)")
    ap.add_argument("--device",       default="cuda")
    args = ap.parse_args()

    os.makedirs(args.save, exist_ok=True)

    # ── Load full model ───────────────────────────────────────────────────────
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.model)))
    from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration
    from transformers import AutoProcessor

    print(f"Loading {os.path.basename(args.model)} in FP16 …")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    t0 = time.time()
    model = MERaLiON2ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, use_safetensors=True)
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ── Save non-text-decoder weights (speech encoder + adapter) ─────────────
    print("Saving non-text-decoder weights …")
    non_td_sd = {k: v.cpu() for k, v in model.state_dict().items()
                 if not k.startswith("text_decoder.")}
    torch.save(non_td_sd, os.path.join(args.save, "non_td_weights.pt"))
    print(f"  {len(non_td_sd)} tensors saved")

    # ── Calibration texts ─────────────────────────────────────────────────────
    print(f"Collecting {args.num_calib} calibration texts …")
    calib_texts = get_calib_texts(args.dataset, args.num_calib)
    print(f"  Got {len(calib_texts)} texts")
    if len(calib_texts) < 8:
        raise RuntimeError("Too few calibration texts — check dataset field names.")

    # ── AutoAWQ quantization ──────────────────────────────────────────────────
    quant_config = {
        "zero_point":   True,
        "q_group_size": args.q_group_size,
        "w_bit":        4,
        "version":      "GEMM",
    }

    _patch_gemma2_forward()

    if args.pruned:
        # Pruned models have non-uniform intermediate_size per layer.
        # AutoAWQ's from_pretrained reconstructs from config (uniform) → shape
        # mismatch.  Instead, monkeypatch AutoModelForCausalLM.from_pretrained
        # to inject our already-loaded text_decoder so AutoAWQ's full init
        # path runs (model_type detection, class selection) without loading
        # from disk.
        # Keep on CPU — AutoAWQ manages device placement during quantize()
        text_decoder = model.text_decoder.half()
        del model
        torch.cuda.empty_cache()

        # Save minimal config dir so AutoAWQ can detect model_type from config.json
        td_cfg_dir = os.path.join(args.save, "_td_cfg_tmp")
        os.makedirs(td_cfg_dir, exist_ok=True)
        text_decoder.config.save_pretrained(td_cfg_dir)
        processor.tokenizer.save_pretrained(td_cfg_dir)

        # Force model_type = "gemma2" so AutoAWQ can look it up in its mapping dict
        _cfg_path = os.path.join(td_cfg_dir, "config.json")
        with open(_cfg_path) as _f:
            _cfg_data = json.load(_f)
        if _cfg_data.get("model_type") != "gemma2":
            print(f"  Overriding model_type '{_cfg_data.get('model_type')}' → 'gemma2'")
            _cfg_data["model_type"] = "gemma2"
            with open(_cfg_path, "w") as _f:
                json.dump(_cfg_data, _f, indent=2)

        from awq import AutoAWQForCausalLM
        from transformers import AutoModelForCausalLM as _AutoMCLM
        from unittest.mock import patch

        print(f"Pruned model: injecting text_decoder into AutoAWQ via monkeypatch …")
        with patch.object(_AutoMCLM, "from_pretrained", return_value=text_decoder):
            awq_model = AutoAWQForCausalLM.from_pretrained(
                td_cfg_dir, trust_remote_code=True)

        # Verify injection: awq_model.model should be our text_decoder
        if getattr(awq_model, "model", None) is not text_decoder:
            raise RuntimeError(
                "Monkeypatch injection did not work — awq_model.model is not text_decoder. "
                f"Got type: {type(getattr(awq_model, 'model', None))}. "
                "Check AutoAWQ version or update the patch target.")
        print("  Injection verified: awq_model.model is text_decoder ✓")

        shutil.rmtree(td_cfg_dir, ignore_errors=True)
    else:
        # Non-pruned: save as standalone HF model, load via from_pretrained.
        td_fp16_dir = os.path.join(args.save, "_td_fp16_tmp")
        print(f"Saving text_decoder to {td_fp16_dir} …")
        model.text_decoder.save_pretrained(td_fp16_dir)
        processor.tokenizer.save_pretrained(td_fp16_dir)
        del model
        torch.cuda.empty_cache()

        from awq import AutoAWQForCausalLM
        print(f"Loading text_decoder for AutoAWQ (group={args.q_group_size}) …")
        awq_model = AutoAWQForCausalLM.from_pretrained(
            td_fp16_dir, device_map="auto", safetensors=True)

    print("Running AutoAWQ calibration + quantization …")
    t0 = time.time()
    awq_model.quantize(
        processor.tokenizer,
        quant_config=quant_config,
        calib_data=calib_texts,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Save quantized text decoder ───────────────────────────────────────────
    td_awq_dir = os.path.join(args.save, "text_decoder_awq")
    print(f"Saving quantized text_decoder to {td_awq_dir} …")
    awq_model.save_quantized(td_awq_dir)
    processor.tokenizer.save_pretrained(td_awq_dir)

    # ── Copy config / tokenizer files for full-model loading ─────────────────
    COPY_EXTS = {".json", ".txt", ".py", ".model", ".tiktoken"}
    for fn in os.listdir(args.model):
        if any(fn.endswith(ext) for ext in COPY_EXTS) and fn != "awq_config.json":
            shutil.copy(os.path.join(args.model, fn), os.path.join(args.save, fn))

    cfg = {
        "quant_type":   "autoawq4",
        "q_group_size": args.q_group_size,
        "w_bit":        4,
        "num_calib":    len(calib_texts),
        "version":      "GEMM",
        "pruned":       args.pruned,
        "source_model": os.path.abspath(args.model),
    }
    with open(os.path.join(args.save, "awq_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    if not args.pruned:
        shutil.rmtree(os.path.join(args.save, "_td_fp16_tmp"), ignore_errors=True)

    print(f"\nDone.  Model saved to {args.save}")
    print(f"  non_td_weights.pt : speech encoder + adapter (FP16)")
    print(f"  text_decoder_awq/ : quantized text decoder (AutoAWQ GEMM)")
    print(f"Load with: python infer_gpu.py --model {args.save} --quant autoawq4")


if __name__ == "__main__":
    main()
