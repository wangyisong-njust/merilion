#!/usr/bin/env python3
"""
AWQ (Activation-aware Weight Quantization) for MERaLiON-2-3B.

Calibrates per-column scales using real audio→text activation statistics,
then applies W4A16 INT4 group quantization to the text_decoder (Gemma2).
Speech encoder and audio adapter remain in FP16.

Saved artifacts (--save dir):
    model_awq4.pt      full model state dict with _AWQ4Linear buffers
    awq_config.json    quantization metadata
    *                  config/tokenizer files copied from original model

Usage:
    python quantize_awq.py \\
        --model   /path/to/MERaLiON-2-3B \\
        --dataset /path/to/IMDA_PART1_mono_en_30_ASR \\
        --save    /path/to/MERaLiON-2-3B-AWQ4 \\
        --num_calib 64 --group_size 64 --alpha 0.5
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE             = 16_000
CHUNK_SIZE              = SAMPLE_RATE * 30
MAX_CHUNKS              = 8
SPEECH_TOKENS_PER_CHUNK = 100
SKIP_MODULES            = {"speech_encoder", "speech_audio_adapter", "lm_head"}


# ── AWQ INT4 linear layer ─────────────────────────────────────────────────────

class _AWQ4Linear(nn.Module):
    """
    W4A16 group-quantized linear layer with AWQ-calibrated per-column scales.

    Layout:
        weight_q   [O, I_pad//2]  uint8  — nibble-packed INT4 (W / col_scale)
        scales     [O, n_groups]  fp16   — per-group scale for quantised W'
        zeros      [O, n_groups]  fp16   — per-group zero  for quantised W'
        col_scale  [I]            fp16   — per-input-channel AWQ scale

    Forward:
        out = dequant(weight_q) * col_scale @ x
            ≈ W @ x
    """

    def __init__(self, in_features, out_features, has_bias, group_size=64):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.group_size   = group_size
        self._pad         = (-in_features) % group_size
        I_pad    = in_features + self._pad
        n_groups = I_pad // group_size
        self.register_buffer("weight_q",   torch.zeros(out_features, I_pad // 2, dtype=torch.uint8))
        self.register_buffer("scales",     torch.zeros(out_features, n_groups,   dtype=torch.float16))
        self.register_buffer("zeros",      torch.zeros(out_features, n_groups,   dtype=torch.float16))
        self.register_buffer("col_scale",  torch.ones(in_features,               dtype=torch.float16))
        if has_bias:
            self.register_buffer("linear_bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.linear_bias = None

    @classmethod
    def from_linear(cls, module: nn.Linear, x_stats: torch.Tensor,
                    group_size: int = 64, alpha: float = 0.5) -> "_AWQ4Linear":
        """
        Quantize a Linear layer using AWQ-calibrated column scales.

        x_stats  [in_features] — mean |activation| per input channel over calibration set.
        """
        O, I = module.weight.shape
        new  = cls(I, O, module.bias is not None, group_size)
        W    = module.weight.detach().float()

        # ── AWQ scale: balance activation and weight magnitude ────────────────
        x_s = x_stats.float().clamp(min=1e-8).pow(alpha)
        w_s = W.abs().max(0).values.clamp(min=1e-8).pow(1 - alpha)
        col_scale = (x_s * w_s)
        col_scale = (col_scale / col_scale.mean()).clamp(min=1e-4)  # unit-mean normalised

        # W' = W / col_scale  →  at inference: out = W' @ (x * col_scale)
        W_sc = W / col_scale.unsqueeze(0)

        # ── INT4 group quantisation of W' ─────────────────────────────────────
        I_pad = I + new._pad
        n_g   = I_pad // group_size
        if new._pad:
            W_sc = F.pad(W_sc, (0, new._pad))
        Wg     = W_sc.view(O, n_g, group_size)
        w_min  = Wg.min(-1).values
        w_max  = Wg.max(-1).values
        q_sc   = (w_max - w_min).div(15.0).clamp(min=1e-8)
        q_zero = w_min
        q = ((Wg - q_zero.unsqueeze(-1)) / q_sc.unsqueeze(-1)).round().clamp(0, 15).to(torch.uint8)
        q = q.view(O, I_pad)
        packed = (q[:, 1::2] << 4) | q[:, 0::2]

        new.weight_q.copy_(packed)
        new.scales.copy_(q_sc.to(torch.float16))
        new.zeros.copy_(q_zero.to(torch.float16))
        new.col_scale.copy_(col_scale.to(torch.float16))
        if module.bias is not None:
            new.linear_bias.copy_(module.bias.detach().to(torch.float16))
        return new

    def _dequantize(self) -> torch.Tensor:
        O     = self.out_features
        I_pad = self.weight_q.shape[1] * 2
        n_g   = self.scales.shape[1]
        lo = (self.weight_q & 0x0F).to(torch.float16)
        hi = (self.weight_q >>  4).to(torch.float16)
        q  = torch.stack([lo, hi], dim=-1).view(O, I_pad)
        q  = q.view(O, n_g, self.group_size)
        w  = self.scales.unsqueeze(-1) * q + self.zeros.unsqueeze(-1)
        w  = w.view(O, I_pad)[:, :self.in_features]
        return w * self.col_scale.unsqueeze(0)  # undo column scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w    = self._dequantize().to(x.dtype)
        bias = self.linear_bias.to(x.dtype) if self.linear_bias is not None else None
        return F.linear(x, w, bias)


# ── calibration ───────────────────────────────────────────────────────────────

def collect_activation_stats(model, processor, dataset_iter, num_calib, device):
    """Hook into every target Linear layer and accumulate |input|.mean(0)."""
    stats  = {}   # name -> list of [in_features] fp32 tensors
    hooks  = []

    def make_hook(name):
        def _h(module, inp, _out):
            x = inp[0].detach().float()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            elif x.dim() != 2:
                return
            stats.setdefault(name, []).append(x.abs().mean(0).cpu())
        return _h

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(s in name for s in SKIP_MODULES) or "text_decoder" not in name:
            continue
        hooks.append(mod.register_forward_hook(make_hook(name)))

    import librosa
    tokenizer       = processor.tokenizer
    fe              = processor.feature_extractor
    speech_token_id = model.config.speech_token_index
    target_sr       = fe.sampling_rate

    n_done = 0
    print(f"Collecting activations from {num_calib} calibration samples …")
    for sample in dataset_iter:
        if n_done >= num_calib:
            break
        try:
            audio = np.asarray(sample["context"]["audio"]["array"], dtype=np.float32)
            if audio.ndim == 2:
                audio = audio.mean(-1)
            sr = sample["context"]["audio"]["sampling_rate"]
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            instr = (sample["instruction"]["text"]
                     if isinstance(sample["instruction"], dict)
                     else sample["instruction"])

            chunks = [audio[i:i+CHUNK_SIZE] for i in range(0, len(audio), CHUNK_SIZE)]
            chunks = [c for c in chunks if len(c) >= target_sr][:MAX_CHUNKS]
            if not chunks:
                continue
            chunks = [np.pad(c, (0, max(0, CHUNK_SIZE-len(c))), "constant") for c in chunks]

            out_fe  = fe(chunks, sampling_rate=target_sr, return_attention_mask=True,
                         padding="max_length", return_tensors="pt", do_normalize=True)
            inp_feat  = out_fe.input_features.to(device, dtype=torch.float16)
            feat_mask = out_fe.attention_mask.to(device)
            n_speech  = len(chunks) * SPEECH_TOKENS_PER_CHUNK

            conv = [{"role": "user",
                     "content": f"Instruction: {instr} \nFollow the text instruction based on the following audio: <SpeechHere>"}]
            prompt  = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
            pos     = raw_ids.index(speech_token_id)
            input_ids  = torch.tensor(
                [raw_ids[:pos] + [speech_token_id]*n_speech + raw_ids[pos+1:]], dtype=torch.long, device=device)
            attn_mask  = torch.ones_like(input_ids)

            with torch.inference_mode():
                model(input_ids=input_ids, attention_mask=attn_mask,
                      input_features=inp_feat, feature_attention_mask=feat_mask,
                      use_cache=False, return_dict=True)
            n_done += 1
            if n_done % 16 == 0:
                print(f"  [{n_done}/{num_calib}]")
        except Exception as e:
            print(f"  [warn] sample skipped: {e}")

    for h in hooks:
        h.remove()

    avg = {name: torch.stack(lst).mean(0) for name, lst in stats.items()}
    print(f"  Collected stats for {len(avg)} layers ({n_done} samples used)")
    return avg


# ── quantisation ──────────────────────────────────────────────────────────────

def apply_awq4(model, act_stats, group_size=64, alpha=0.5):
    n = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        if any(s in name for s in SKIP_MODULES) or "text_decoder" not in name:
            continue
        x_stats = act_stats.get(name, torch.ones(mod.in_features))
        parts  = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1],
                _AWQ4Linear.from_linear(mod, x_stats, group_size, alpha))
        n += 1
    print(f"  Replaced {n} Linear layers (group_size={group_size}, alpha={alpha})")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",       required=True, help="Path to original MERaLiON-2-3B")
    ap.add_argument("--dataset",     required=True, help="IMDA_PART1 dataset path")
    ap.add_argument("--save",        required=True, help="Output directory for AWQ model")
    ap.add_argument("--num_calib",   type=int,   default=64,  help="Calibration samples")
    ap.add_argument("--group_size",  type=int,   default=64,  help="INT4 group size")
    ap.add_argument("--alpha",       type=float, default=0.5, help="AWQ balance exponent")
    ap.add_argument("--device",      default="cuda")
    args = ap.parse_args()

    os.makedirs(args.save, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────────────
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.model)))
    from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration
    from transformers import AutoProcessor

    print(f"Loading {os.path.basename(args.model)} in FP16 …")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    t0    = time.time()
    model = MERaLiON2ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, use_safetensors=True)
    model = model.to(args.device).eval()
    print(f"  Loaded in {time.time()-t0:.1f}s  "
          f"({torch.cuda.max_memory_allocated(args.device)/1e9:.1f} GB)")

    # ── calibration ───────────────────────────────────────────────────────────
    from datasets import load_from_disk
    data    = load_from_disk(os.path.abspath(args.dataset))
    data    = data.shuffle(seed=42)
    start   = min(10500, len(data))
    pool    = data.select(range(start, min(start + args.num_calib * 4, len(data))))
    act_stats = collect_activation_stats(
        model, processor, pool, args.num_calib, args.device)

    # ── quantise on CPU ───────────────────────────────────────────────────────
    print("Moving model to CPU for quantisation …")
    model = model.cpu()
    apply_awq4(model, act_stats, args.group_size, args.alpha)

    # ── save ──────────────────────────────────────────────────────────────────
    print(f"Saving to {args.save} …")
    torch.save(model.state_dict(), os.path.join(args.save, "model_awq4.pt"))

    cfg = {
        "quant_type":       "awq4",
        "group_size":       args.group_size,
        "alpha":            args.alpha,
        "num_calib":        args.num_calib,
        "skip_modules":     list(SKIP_MODULES),
        "bits":             4,
    }
    with open(os.path.join(args.save, "awq_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # copy tokenizer / config files
    COPY_EXTS = {".json", ".txt", ".py", ".model", ".tiktoken"}
    for fn in os.listdir(args.model):
        if any(fn.endswith(ext) for ext in COPY_EXTS) and fn != "awq_config.json":
            shutil.copy(os.path.join(args.model, fn), os.path.join(args.save, fn))

    size_gb = os.path.getsize(os.path.join(args.save, "model_awq4.pt")) / 1e9
    print(f"Done.  model_awq4.pt = {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
