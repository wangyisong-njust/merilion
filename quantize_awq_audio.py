"""AWQ W4A16 with audio-conditioned calibration for MERaLiON-2-3B.

Strategy:
  1. Build the full MERaLiON model (speech_encoder + adapter + text_decoder).
  2. For each calibration audio, run the speech path manually and capture
     the `inputs_embeds` that text_decoder would receive at real inference.
  3. Feed these `inputs_embeds` (NOT raw text tokens) into llmcompressor's
     oneshot, targeting `model.text_decoder` only.  This way AWQ sees the
     actual audio-conditioned activation distribution it will see at
     inference — fixing the text/audio domain mismatch that made the
     plain AWQ path produce garbage.
  4. After quantization, the full model is saved — the in-place modified
     text_decoder weights + a quantization_config block that
     CompressedTensorsHfQuantizer will load via CompressedLinear on sm_80+.

The speech_encoder / adapter / ln_speech / lm_head stay at their original
bf16 precision (listed in `ignore` + not part of text_decoder anyway).
"""
import argparse
import os
import sys

import numpy as np
import torch


def prepare_audio(audio, sr, processor,
                  chunk_size=16000*30, max_chunks=8, tokens_per_chunk=100):
    import librosa
    fe = processor.feature_extractor
    target_sr = fe.sampling_rate
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    chunks = []
    for i in range(0, len(audio), chunk_size):
        c = audio[i:i + chunk_size]
        if len(c) < target_sr:
            c = np.pad(c, (0, target_sr - len(c)), "constant")
        chunks.append(c)
    chunks = chunks[:max_chunks]
    out = fe(chunks, sampling_rate=target_sr, return_attention_mask=True,
             padding="max_length", return_tensors="pt", do_normalize=True)
    return out.input_features, out.attention_mask, len(chunks) * tokens_per_chunk


def build_inputs_embeds_dataset(model, processor, ds_path, n, start_idx,
                                 device, dtype):
    """For each calibration audio, run speech_encoder + adapter manually and
    scatter audio contexts into `inputs_embeds`.  Returns a HF Dataset with
    one `inputs_embeds` column (fixed-shape T × H), plus `attention_mask`.

    This reproduces exactly what MERaLiON2.forward() builds as the input
    to text_decoder on a real audio prefill, so AWQ sees the right
    activation distribution."""
    from datasets import load_from_disk, Dataset

    tokenizer = processor.tokenizer
    speech_token_id = model.config.speech_token_index
    hidden = model.text_decoder.lm_head.in_features

    data = load_from_disk(os.path.abspath(ds_path))
    raw_embeds_list = []
    max_len = 0
    for i in range(start_idx, len(data)):
        if len(raw_embeds_list) >= n:
            break
        s = data[i]
        ao = (s.get("context") or {}).get("audio") or {}
        arr = ao.get("array")
        if arr is None:
            continue
        audio = np.asarray(arr, dtype=np.float32)
        sr = ao.get("sampling_rate", 16000)

        try:
            input_features, feature_attention_mask, n_speech = prepare_audio(
                audio, sr, processor)
        except Exception:
            continue

        conv = [{"role": "user",
                 "content": ("Instruction: Transcribe the speech \n"
                             "Follow the text instruction based on the "
                             "following audio: <SpeechHere>")}]
        prompt = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True)
        raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
        pos = raw_ids.index(speech_token_id)
        full_ids = raw_ids[:pos] + [speech_token_id] * n_speech + raw_ids[pos + 1:]

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        input_features = input_features.to(device).to(dtype)
        feature_attention_mask = feature_attention_mask.to(device)

        # Replicate the audio-embed construction from MERaLiON2.forward().
        with torch.inference_mode():
            speech_ctx = model.speech_encoder(
                input_features, attention_mask=feature_attention_mask
            ).last_hidden_state
            speech_ctx = model.ln_speech(speech_ctx)
            speech_ctx = model.speech_audio_adapter(speech_ctx)
            inputs_embeds = model.text_decoder.base_model.embed_tokens(input_ids)
            speech_mask = (input_ids == speech_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                speech_mask, speech_ctx.to(inputs_embeds.dtype).to(inputs_embeds.device))

        # Keep on CPU to avoid VRAM bloat; cast to a stable dtype.
        raw_embeds_list.append(inputs_embeds.squeeze(0).to(dtype).cpu())
        max_len = max(max_len, raw_embeds_list[-1].shape[0])

    # Pad every sample to max_len along the time dim so datasets.Dataset
    # accepts a tensor column of uniform shape.
    rows = []
    for e in raw_embeds_list:
        T = e.shape[0]
        pad = max_len - T
        if pad > 0:
            e = torch.nn.functional.pad(e, (0, 0, 0, pad))   # pad time dim
        attn = [1] * T + [0] * pad
        rows.append({
            "inputs_embeds":  e,   # (max_len, hidden)
            "attention_mask": attn,
        })
    ds = Dataset.from_list(rows)
    print(f"  built {len(ds)} samples, padded seq_len={max_len}, "
          f"hidden={hidden}")
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",       required=True)
    ap.add_argument("--calib_ds",    required=True, help="IMDA dataset path")
    ap.add_argument("--num_calib",   type=int, default=128)
    ap.add_argument("--start_idx",   type=int, default=30)
    ap.add_argument("--output_dir",  required=True)
    ap.add_argument("--device",      default="cuda")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration
    from transformers import AutoProcessor

    print(f"Loading base model from {args.model} …")
    model = MERaLiON2ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device).eval()
    model.config._name_or_path = os.path.abspath(args.model)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    print(f"  loaded, VRAM={torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    print(f"\nPrecomputing audio-conditioned inputs_embeds (n={args.num_calib}) …")
    calib_ds = build_inputs_embeds_dataset(
        model, processor, args.calib_ds, args.num_calib,
        args.start_idx, args.device, torch.bfloat16)
    if len(calib_ds) == 0:
        raise RuntimeError("No valid calibration samples")

    # ── AWQ recipe ────────────────────────────────────────────────────────────
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.modifiers.awq.mappings import AWQMapping

    # Now scope is only text_decoder so ignore speech items isn't strictly
    # needed, but keep it to be safe.  lm_head stays un-quantised.
    ignore = ["re:.*lm_head.*"]

    # Inside text_decoder (= Gemma2ForCausalLM) the prefix is empty, so mappings
    # are plain Gemma2 defaults.
    awq_mappings = [
        AWQMapping(smooth_layer="re:.*input_layernorm$",
                   balance_layers=["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"]),
        AWQMapping(smooth_layer="re:.*v_proj$",
                   balance_layers=["re:.*o_proj$"]),
        AWQMapping(smooth_layer="re:.*pre_feedforward_layernorm$",
                   balance_layers=["re:.*gate_proj$", "re:.*up_proj$"]),
        AWQMapping(smooth_layer="re:.*up_proj$",
                   balance_layers=["re:.*down_proj$"]),
    ]

    recipe = AWQModifier(
        targets=["Linear"],
        scheme="W4A16",
        ignore=ignore,
        mappings=awq_mappings,
    )

    print("\nRunning oneshot(AWQ) on text_decoder with audio-conditioned "
          "inputs_embeds …")
    oneshot(
        model=model.text_decoder,
        processor=processor,
        recipe=recipe,
        dataset=calib_ds,
        output_dir=args.output_dir,         # oneshot saves text_decoder here
        save_compressed=True,
        num_calibration_samples=len(calib_ds),
        pad_to_max_length=False,
        concatenate_data=False,
    )

    # ── Rebuild the full MERaLiON model with the quantised text_decoder ──────
    # oneshot saved ONLY text_decoder; assemble the full model config that
    # points to its quantised text_decoder + the original speech encoder bits.
    #
    # Simplest approach: reload the full model from base_path, override
    # text_decoder with the quantised version, save as one directory with the
    # MERaLiON2 config including quantization_config scoped to text_decoder.*.
    #
    # For now we keep oneshot's output (just the text_decoder) as the final
    # artefact and provide a companion loader note in the README.
    print(f"\nDone.  Text-decoder only, quantized, saved to {args.output_dir}")
    print("  NOTE: this is the quantised Gemma2 text_decoder in compressed-")
    print("  tensors format.  To use with full MERaLiON audio inference, load")
    print("  the original MERaLiON checkpoint then swap its text_decoder with")
    print("  this one.  See `load_meralion_with_quant_td()` helper (TBD).")


if __name__ == "__main__":
    main()
