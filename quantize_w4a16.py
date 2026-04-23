"""Produce a W4A16 (weight-INT4, activation-BF16) compressed-tensors
checkpoint for MERaLiON-2-3B that the Marlin kernel path in transformers
can load directly on sm_80+ GPUs.

Three methods selectable via --method:

    RTN   (default) — Round-to-nearest, data-free, ~1 minute.  Matches
                      how `MERaLiON-2-3B-W4A16-RTN-textonly` was made.
    GPTQ           — Calibration-based error-minimisation, 30-60 min
                      depending on calib size.  Best quality for INT4.
    AWQ            — Activation-aware, similar quality to GPTQ.

Quantizes ONLY `text_decoder.*` Linears (82 % of params).  Leaves
speech_encoder / ln / adapter / lm_head in their original dtype to
avoid quality loss on the audio path.

Usage:
    python quantize_w4a16.py \\
        --model        /path/to/MERaLiON-2-3B \\
        --calib_ds     /path/to/IMDA_PART1_train   (needed for GPTQ/AWQ) \\
        --num_calib    512 \\
        --output_dir   /path/to/MERaLiON-2-3B-W4A16-GPTQ \\
        --method       GPTQ
"""
import argparse
import json
import os
import sys

import torch


def build_calib_examples(ds_path, tokenizer, n=512, seq_len=512, start_idx=30):
    """Build a HF `datasets.Dataset` with one `input_ids` column, sized for
    AWQ/GPTQ calibration on IMDA reference transcripts.

    Calibration runs the model on TEXT only — not audio — so we tokenize
    the reference transcripts directly with BOS.  For audio-aware
    calibration you would need to also feed input_features through the
    speech encoder, but llm-compressor's oneshot doesn't support that
    shape naturally, and text-only calibration has worked fine for the
    W4A16-RTN-textonly checkpoint we already have.
    """
    from datasets import load_from_disk, Dataset
    data = load_from_disk(os.path.abspath(ds_path))
    rows = []
    # Pad to seq_len so every row has the same length — llm-compressor
    # often expects fixed-shape tensor columns.
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0
    for i in range(start_idx, len(data)):
        if len(rows) >= n:
            break
        s = data[i]
        oa = s.get("other_attributes") or {}
        ref = oa.get("Transcription") or oa.get("transcription")
        if ref is None:
            ans = s.get("answer")
            ref = ans.get("text") if isinstance(ans, dict) else ans
        if not ref or not str(ref).strip():
            continue
        ids = tokenizer.encode(str(ref), add_special_tokens=True)
        if len(ids) < 16:
            continue
        ids = ids[:seq_len]
        attn = [1] * len(ids)
        # Pad to seq_len
        pad_len = seq_len - len(ids)
        ids  = ids  + [pad_id] * pad_len
        attn = attn + [0]     * pad_len
        rows.append({"input_ids": ids, "attention_mask": attn})
    ds = Dataset.from_list(rows)
    print(f"  built {len(ds)} calibration examples (seq_len={seq_len})")
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      required=True, help="Base MERaLiON-2-3B path")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--method",     default="RTN", choices=["RTN", "GPTQ", "AWQ"])
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--calib_ds",   default=None,
                    help="Train-split IMDA dataset path (required for GPTQ/AWQ)")
    ap.add_argument("--num_calib",  type=int, default=512)
    ap.add_argument("--calib_seq_len", type=int, default=512)
    ap.add_argument("--device",     default="cuda")
    args = ap.parse_args()

    if args.method in ("GPTQ", "AWQ") and not args.calib_ds:
        ap.error(f"--method {args.method} needs --calib_ds")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load MERaLiON-2-3B through our bundled patched code.
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path: sys.path.insert(0, here)
    from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration
    from transformers import AutoProcessor

    print(f"Loading base model from {args.model} …")
    model = MERaLiON2ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device).eval()
    # llm-compressor's save hook calls hf_hub_download(config._name_or_path,…)
    # to copy modeling files — fails if empty.  Set to the local dir so it
    # falls through to the local-path branch.
    model.config._name_or_path = os.path.abspath(args.model)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = processor.tokenizer
    print(f"  base loaded, VRAM={torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # Build recipe.  `targets=["Linear"]` + ignore regex means quantise all
    # Linears EXCEPT those matching the ignore list (speech encoder, adapter,
    # layernorms, lm_head).  text_decoder's ~182 Linears get quantised.
    ignore = [
        "re:.*speech_encoder.*",
        "re:.*speech_audio_adapter.*",
        "re:.*ln_speech.*",
        "re:.*lm_head.*",
    ]

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier, GPTQModifier

    if args.method == "RTN":
        # Data-free W4A16 RTN.  Matches the existing *-W4A16-RTN-textonly format.
        recipe = QuantizationModifier(
            targets=["Linear"],
            scheme="W4A16",
            ignore=ignore,
        )
        calib = None
    elif args.method == "GPTQ":
        # Calibration-driven W4A16.  group_size=128 is the sweet spot for
        # Marlin kernel performance.
        recipe = GPTQModifier(
            targets=["Linear"],
            scheme="W4A16",
            ignore=ignore,
            dampening_frac=0.01,
        )
        calib = build_calib_examples(
            args.calib_ds, tokenizer, n=args.num_calib, seq_len=args.calib_seq_len)
    elif args.method == "AWQ":
        # AWQ recipe via llmcompressor.  The default Gemma2 mapping set uses
        # plain regexes like `re:.*q_proj$` that ALSO match speech_encoder's
        # projections — and AWQ modifies the `smooth_layer` weights in place,
        # which corrupts the frozen speech encoder and produces a model whose
        # prefill argmax is EOS.  Constrain every regex to `text_decoder.`.
        from llmcompressor.modifiers.awq import AWQModifier
        from llmcompressor.modifiers.awq.mappings import AWQMapping
        td = r"text_decoder\."
        mappings = [
            # Pre-attention LN → q/k/v
            AWQMapping(
                smooth_layer=f"re:{td}.*input_layernorm$",
                balance_layers=[f"re:{td}.*q_proj$",
                                f"re:{td}.*k_proj$",
                                f"re:{td}.*v_proj$"],
            ),
            # v_proj → o_proj
            AWQMapping(
                smooth_layer=f"re:{td}.*v_proj$",
                balance_layers=[f"re:{td}.*o_proj$"],
            ),
            # Pre-FFN LN → gate/up
            AWQMapping(
                smooth_layer=f"re:{td}.*pre_feedforward_layernorm$",
                balance_layers=[f"re:{td}.*gate_proj$",
                                f"re:{td}.*up_proj$"],
            ),
            # up_proj → down_proj
            AWQMapping(
                smooth_layer=f"re:{td}.*up_proj$",
                balance_layers=[f"re:{td}.*down_proj$"],
            ),
        ]
        recipe = AWQModifier(
            targets=["Linear"],
            scheme="W4A16",
            ignore=ignore,
            mappings=mappings,
        )
        calib = build_calib_examples(
            args.calib_ds, tokenizer, n=args.num_calib, seq_len=args.calib_seq_len)

    print(f"\nQuantizing with {args.method} (W4A16, group={args.group_size}) …")
    print(f"  ignore: {ignore}")

    oneshot(
        model=model,
        processor=processor,             # includes tokenizer
        recipe=recipe,
        dataset=calib,
        output_dir=args.output_dir,
        save_compressed=True,            # compressed-tensors format
    )

    # Copy processor config so the output dir is self-contained.
    # Normally oneshot saves tokenizer too, but we ensure everything is there.
    import shutil
    for aux in ["preprocessor_config.json", "processor_config.json",
                "processing_meralion2.py", "chat_template.jinja"]:
        src = os.path.join(args.model, aux)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.output_dir, aux))

    print(f"\nSaved to {args.output_dir}")
    print("Verify with:")
    print(f"  ls -la {args.output_dir}/*.safetensors")
    print("  look for *.weight_packed / *.weight_scale keys inside")


if __name__ == "__main__":
    main()
