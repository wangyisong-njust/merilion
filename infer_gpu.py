"""GPU inference benchmark for MERaLiON-2 (original + pruned models).

Mirrors infer_cpu.py in structure and JSON output schema so the two can be
compared directly in a summary table or make_demo_html.py.

Quantization backends:
  bf16  (default) — BF16 + FlashAttention-2 / SDPA    (fastest; full quality)
  fp16            — FP16 + FlashAttention-2 / SDPA
  int8            — BitsAndBytes LLM.int8() — speech modules kept in FP16
  int4            — BitsAndBytes NF4 4-bit  — bfloat16 compute dtype

Timing uses torch.cuda.synchronize() for wall-clock accuracy.
GPU VRAM is measured via torch.cuda.max_memory_allocated().

Output JSON adds gpu_mem_gb and device fields alongside the same
wer / avg_latency_s / avg_decode_tps / quant_method keys as the CPU version.

Usage:
    # BF16 baseline on single GPU (default):
    python infer_gpu.py \\
        --model /path/to/MERaLiON-2-3B \\
        --dataset /path/to/IMDA_PART1_mono_en_30_ASR \\
        --num_samples 20 --output gpu_bf16.json

    # BitsAndBytes INT8:
    python infer_gpu.py --model ... --dataset ... --quant int8 --output gpu_int8.json

    # BitsAndBytes NF4 INT4:
    python infer_gpu.py --model ... --dataset ... --quant int4 --output gpu_int4.json

    # Single audio file:
    python infer_gpu.py --model ... --audio sample.wav
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


# ── shared helpers (same as infer_cpu.py) ────────────────────────────────────

class NGramDraft:
    """N-gram draft predictor with optional pre-built corpus index.

    If a corpus index is provided (dict mapping prefix tuple → next token),
    lookup is O(1).  Falls back to scanning the current generated context
    (original behaviour) when the prefix is absent from the index.
    """

    def __init__(self, ngram_sizes: tuple = (3, 4), index: dict = None):
        self.ngram_sizes = sorted(ngram_sizes, reverse=True)
        self.index = index or {}   # prefix_tuple → next_token_id

    @classmethod
    def from_corpus_file(cls, path: str) -> "NGramDraft":
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(ngram_sizes=tuple(data["ngram_sizes"]), index=data["index"])
        print(f"  NGramDraft: loaded {len(obj.index):,} prefixes from {path}")
        return obj

    def propose(self, ctx: list, gamma: int) -> list:
        draft: list = []
        cur = list(ctx)
        for _ in range(gamma):
            tok = self._next(cur)
            if tok is None:
                break
            draft.append(tok)
            cur.append(tok)
        return draft

    def _next(self, ctx: list):
        for ng in self.ngram_sizes:
            plen = ng - 1
            if len(ctx) < plen:
                continue
            prefix = tuple(ctx[-plen:])
            # 1) corpus index lookup (O(1))
            if prefix in self.index:
                return self.index[prefix]
            # 2) fallback: scan current generated context
            n = len(ctx)
            if n >= ng:
                for i in range(n - ng):
                    if ctx[i: i + plen] == list(prefix):
                        return ctx[i + plen]
        return None


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def _normalize_text_audiobench(text: str) -> str:
    import jiwer
    _pipeline = jiwer.Compose([
        jiwer.RemoveMultipleSpaces(),
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemoveKaldiNonWords(),
        jiwer.RemovePunctuation(),
    ])
    text = text.lower()
    for digit, word in [("0","zero"),("1","one"),("2","two"),("3","three"),
                        ("4","four"),("5","five"),("6","six"),("7","seven"),
                        ("8","eight"),("9","nine")]:
        text = re.sub(r'\b' + digit + r'\b', word, text)
    text = re.sub(r'[\(\[\{\<][^\n\(\)\[\]\{\}\<\>]*[\)\]\}\>]', "", text)
    text = _pipeline(text)
    text = re.sub(r'\b(uh|umm|um|er|ah)\b', '', text)
    return text.strip()


def _model_is_pruned(model) -> bool:
    try:
        cfg = model.text_decoder.model.config
        return (getattr(cfg, "midblock_start", -1) >= 0
                and getattr(cfg, "midblock_ratio", 1.0) < 1.0)
    except Exception:
        return False


def _trim_kv_cache(past_kv, keep_len: int) -> None:
    """Discard KV cache entries beyond keep_len to fix stale state after partial spec acceptance."""
    from transformers.cache_utils import DynamicCache
    if isinstance(past_kv, DynamicCache):
        for i in range(len(past_kv.key_cache)):
            past_kv.key_cache[i] = past_kv.key_cache[i][:, :, :keep_len, :]
            past_kv.value_cache[i] = past_kv.value_cache[i][:, :, :keep_len, :]
    else:  # HybridCache — pre-allocated; zero out stale slots so causal mask excludes them
        for i in range(len(past_kv.key_cache)):
            kc = past_kv.key_cache[i]
            if kc is not None and kc.shape[2] > keep_len:
                kc[:, :, keep_len:, :].zero_()
        for i in range(len(past_kv.value_cache)):
            vc = past_kv.value_cache[i]
            if vc is not None and vc.shape[2] > keep_len:
                vc[:, :, keep_len:, :].zero_()
    past_kv._seen_tokens = keep_len


SAMPLE_RATE = 16000
CHUNK_SIZE = SAMPLE_RATE * 30
SPEECH_TOKENS_PER_CHUNK = 100
MAX_CHUNKS = 8


def prepare_audio(audio_array: np.ndarray, sample_rate: int, processor):
    """Resample + mel features. Same as infer_cpu.py."""
    import librosa
    fe = processor.feature_extractor
    target_sr = fe.sampling_rate
    if sample_rate != target_sr:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate,
                                       target_sr=target_sr)
    chunks = []
    for i in range(0, len(audio_array), CHUNK_SIZE):
        chunk = audio_array[i:i + CHUNK_SIZE]
        if len(chunk) < target_sr:
            chunk = np.pad(chunk, (0, target_sr - len(chunk)), "constant")
        chunks.append(chunk)
    chunks = chunks[:MAX_CHUNKS]
    out = fe(chunks, sampling_rate=target_sr, return_attention_mask=True,
             padding="max_length", return_tensors="pt", do_normalize=True)
    return out.input_features, out.attention_mask, len(chunks) * SPEECH_TOKENS_PER_CHUNK


# ── MLX-style int4 quantization ──────────────────────────────────────────────

class _MLX4Linear(torch.nn.Module):
    """Per-group asymmetric int4 linear layer matching MLX affine quantization.

    Mimics mlx.nn.QuantizedLinear:
        scale = (w_max - w_min) / 15
        bias  = w_min
        q     = round((w - bias) / scale)  ∈ [0, 15]
        w_rec = scale * q + bias

    Weights packed as uint8 (2×int4 per byte). Dequantized on every forward().
    """
    def __init__(self, in_features: int, out_features: int,
                 has_bias: bool, group_size: int = 64):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.group_size   = group_size
        # Padding: in_features rounded up to nearest multiple of group_size
        self._pad = (-in_features) % group_size
        I_pad     = in_features + self._pad
        n_groups  = I_pad // group_size
        # Packed 4-bit: 2 values per byte (low nibble first)
        self.register_buffer("weight_q",
            torch.zeros(out_features, I_pad // 2, dtype=torch.uint8))
        self.register_buffer("scales",
            torch.zeros(out_features, n_groups, dtype=torch.float16))
        self.register_buffer("zeros",
            torch.zeros(out_features, n_groups, dtype=torch.float16))
        if has_bias:
            self.register_buffer("linear_bias",
                torch.zeros(out_features, dtype=torch.float16))
        else:
            self.linear_bias = None

    @staticmethod
    def from_linear(module: torch.nn.Linear, group_size: int = 64) -> "_MLX4Linear":
        O, I     = module.weight.shape
        new      = _MLX4Linear(I, O, module.bias is not None, group_size)
        w        = module.weight.detach().float()       # (O, I)
        I_pad    = I + new._pad
        n_groups = I_pad // group_size

        # Pad if needed
        if new._pad:
            w = torch.nn.functional.pad(w, (0, new._pad))

        # Per-group min/max → scale + zero
        wg     = w.view(O, n_groups, group_size)       # (O, G_n, 64)
        w_min  = wg.min(dim=-1).values                  # (O, G_n)
        w_max  = wg.max(dim=-1).values
        scale  = (w_max - w_min) / 15.0
        scale  = scale.clamp(min=1e-8)                  # avoid /0
        zero   = w_min                                  # = bias in MLX notation

        # Quantize
        q = ((wg - zero.unsqueeze(-1)) / scale.unsqueeze(-1))
        q = q.round().clamp(0, 15).to(torch.uint8)     # (O, G_n, 64)
        q = q.view(O, I_pad)

        # Pack: even indices → low nibble, odd → high nibble
        q_low  = q[:, 0::2]                            # (O, I_pad//2)
        q_high = q[:, 1::2]
        packed = (q_high << 4) | q_low                 # uint8

        new.weight_q.copy_(packed)
        new.scales.copy_(scale.to(torch.float16))
        new.zeros.copy_(zero.to(torch.float16))
        if module.bias is not None:
            new.linear_bias.copy_(module.bias.detach().to(torch.float16))
        return new

    def _dequantize(self) -> torch.Tensor:
        O        = self.out_features
        I_pad    = self.weight_q.shape[1] * 2
        n_groups = self.scales.shape[1]

        # Unpack nibbles
        lo = (self.weight_q & 0x0F).to(torch.float16)  # (O, I_pad//2)
        hi = (self.weight_q >> 4).to(torch.float16)
        q  = torch.stack([lo, hi], dim=-1).view(O, I_pad)  # (O, I_pad)

        # Dequantize per group
        q  = q.view(O, n_groups, self.group_size)
        s  = self.scales.unsqueeze(-1)   # (O, G_n, 1)
        z  = self.zeros.unsqueeze(-1)    # (O, G_n, 1)
        w  = s * q + z                   # (O, G_n, 64)
        w  = w.view(O, I_pad)[:, :self.in_features]  # trim pad
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w    = self._dequantize().to(x.dtype)
        bias = self.linear_bias.to(x.dtype) if self.linear_bias is not None else None
        return torch.nn.functional.linear(x, w, bias)


class _AWQ4Linear(torch.nn.Module):
    """W4A16 INT4 linear layer with AWQ-calibrated per-column scales (load-only)."""
    def __init__(self, in_features: int, out_features: int,
                 has_bias: bool, group_size: int = 64):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.group_size   = group_size
        self._pad = (-in_features) % group_size
        I_pad     = in_features + self._pad
        n_groups  = I_pad // group_size
        self.register_buffer("weight_q",  torch.zeros(out_features, I_pad // 2, dtype=torch.uint8))
        self.register_buffer("scales",    torch.zeros(out_features, n_groups,   dtype=torch.float16))
        self.register_buffer("zeros",     torch.zeros(out_features, n_groups,   dtype=torch.float16))
        self.register_buffer("col_scale", torch.ones(in_features,               dtype=torch.float16))
        if has_bias:
            self.register_buffer("linear_bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.linear_bias = None

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
        return w * self.col_scale.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w    = self._dequantize().to(x.dtype)
        bias = self.linear_bias.to(x.dtype) if self.linear_bias is not None else None
        return torch.nn.functional.linear(x, w, bias)


def _apply_mlx4_quant(model, group_size: int = 64) -> None:
    """Replace nn.Linear layers in text_decoder (excluding lm_head) with
    _MLX4Linear.  speech_encoder and speech_audio_adapter are left in FP16.

    Matches the quantization spec from majentik/MERaLiON-2-3B-MLX-4bit:
        Components quantized : Decoder (Gemma2-2B) only
        Components kept full : Whisper encoder, multi-modal adaptor
        Bits: 4, Group size: 64
    """
    SKIP = {"speech_encoder", "speech_audio_adapter", "lm_head"}
    n_replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(s in name for s in SKIP):
            continue
        if "text_decoder" not in name:
            continue

        # Navigate to parent
        parts  = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child  = parts[-1]

        setattr(parent, child,
                _MLX4Linear.from_linear(module, group_size=group_size))
        n_replaced += 1

    print(f"  MLX4 quant: replaced {n_replaced} Linear layers "
          f"(group_size={group_size})")


# ── model loading ─────────────────────────────────────────────────────────────

def load_model_gpu(model_path: str,
                   quant: str = "bf16",
                   flash_attn: bool = True,
                   device: str = "cuda"):
    """Load MERaLiON-2 (original or pruned) on GPU.

    Args:
        quant:      'bf16' | 'fp16' | 'int8' | 'int4'
        flash_attn: use FlashAttention-2 when quant in {bf16, fp16}
                    (requires flash-attn package; falls back to SDPA on failure)
        device:     e.g. 'cuda', 'cuda:0', 'cuda:1'
    """
    from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration
    from transformers import AutoProcessor

    print(f"Loading processor …")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    common_kwargs = dict(use_safetensors=True)

    # meralion2_bl's from_pretrained() ignores device_map and quantization_config;
    # it always loads on CPU.  For BF16/FP16 we just move to GPU afterwards.
    # For BnB INT8/INT4: load weights in FP16 on CPU, then manually swap each
    # nn.Linear into a BnB-typed layer (copying the FP16 data into Int8Params /
    # Params4bit), then .to(device) triggers BnB's cuda() hook which does the
    # actual quantization.  This avoids the meta-tensor issue that arises when
    # replace_with_bnb_linear() is called after weights are already loaded.

    # Modules to leave in FP16 — Whisper encoder + audio adapter + tied lm_head.
    BNB_SKIP = ["speech_encoder", "speech_audio_adapter", "lm_head"]

    if quant == "autoawq4":
        import json as _json
        from awq import AutoAWQForCausalLM
        from transformers import AutoConfig as _AutoConfig

        cfg_path = os.path.join(model_path, "awq_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"awq_config.json not found in {model_path}. "
                "Run quantize_autoawq.py first.")
        with open(cfg_path) as _f:
            _awq_cfg = _json.load(_f)

        print(f"Loading AutoAWQ4 model from {os.path.basename(model_path)} …")
        t0 = time.time()
        td_awq_dir = os.path.join(model_path, "text_decoder_awq")

        if _awq_cfg.get("pruned"):
            import glob as _glob
            import torch.nn as _nn
            from awq.modules.linear import WQLinear_GEMM
            from safetensors.torch import load_file as _lsf

            _src   = _awq_cfg["source_model"]
            _w_bit = _awq_cfg.get("w_bit", 4)
            _grp   = _awq_cfg.get("q_group_size", 128)

            print(f"  Loading pruned source model …")
            _pruned_full = MERaLiON2ForConditionalGeneration.from_pretrained(
                _src, torch_dtype=torch.float16, use_safetensors=True)
            _pruned_td = _pruned_full.text_decoder

            _non_td_sd = {k: v.cpu() for k, v in _pruned_full.state_dict().items()
                          if not k.startswith("text_decoder.")}

            # Load quantized weights from safetensors
            print(f"  Loading quantized weights …")
            _sf_files = sorted(_glob.glob(os.path.join(td_awq_dir, "*.safetensors")))
            if not _sf_files:
                raise FileNotFoundError(f"No safetensors in {td_awq_dir}")
            _qsd = {}
            for _sf in _sf_files:
                _qsd.update(_lsf(_sf))
            _quant_pfx = {k.rsplit(".", 1)[0] for k in _qsd if k.endswith(".qweight")}
            print(f"  {len(_sf_files)} shard(s), {len(_quant_pfx)} quantized layers")

            # Replace Linear → WQLinear_GEMM using ACTUAL module sizes (not config)
            def _replace_wq(mod, pfx):
                for _n, _c in list(mod.named_children()):
                    _p = f"{pfx}.{_n}" if pfx else _n
                    if isinstance(_c, _nn.Linear) and _p in _quant_pfx:
                        setattr(mod, _n, WQLinear_GEMM(
                            _w_bit, _grp, _c.in_features, _c.out_features,
                            _c.bias is not None, "cpu"))
                    else:
                        _replace_wq(_c, _p)
            _replace_wq(_pruned_td, "")

            # Load qweight/qzeros/scales + FP16 embeddings/norms
            _miss, _unex = _pruned_td.load_state_dict(_qsd, strict=False)
            _tied = {"lm_head.weight"}
            _bad  = [k for k in _miss if k not in _tied]
            if _bad:
                print(f"  WARNING: {len(_bad)} missing keys: {_bad[:4]}")

            _fwq = next((m for m in _pruned_td.modules() if hasattr(m, "qweight")), None)
            if _fwq is not None:
                _nz = (_fwq.qweight != 0).sum().item()
                print(f"  WQLinear qweight nonzero: {_nz}/{_fwq.qweight.numel()}")
            del _qsd

            _hf_cfg = _AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model   = MERaLiON2ForConditionalGeneration(_hf_cfg)
            model   = model.to(torch.float16)
            model.load_state_dict(_non_td_sd, strict=False)
            del _pruned_full, _non_td_sd
            torch.cuda.empty_cache()
            _pruned_td = _pruned_td.to(device)
            model      = model.to(device)
            model.text_decoder = _pruned_td
            print(f"  AWQ quantized text decoder loaded ✓")
        else:
            awq_td = AutoAWQForCausalLM.from_quantized(
                td_awq_dir, fuse_layers=True, device_map={"": device})

            _hf_cfg = _AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model   = MERaLiON2ForConditionalGeneration(_hf_cfg)
            model   = model.to(torch.float16)
            non_td_sd = torch.load(
                os.path.join(model_path, "non_td_weights.pt"), map_location="cpu")
            model.load_state_dict(non_td_sd, strict=False)
            model = model.to(device)
            model.text_decoder = awq_td.model

        print(f"  AutoAWQ4 model loaded in {time.time()-t0:.1f}s")

    elif quant == "awq4":
        import json as _json
        cfg_path = os.path.join(model_path, "awq_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"awq_config.json not found in {model_path}. "
                "Run quantize_awq.py first.")
        with open(cfg_path) as f:
            awq_cfg = _json.load(f)
        group_size = awq_cfg.get("group_size", 64)

        print(f"Loading pre-quantized AWQ4 model (group={group_size}) …")
        t0 = time.time()
        # AWQ4 dir has no safetensors — init from config, then load state dict.
        from transformers import AutoConfig as _AutoConfig
        _hf_cfg = _AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = MERaLiON2ForConditionalGeneration(_hf_cfg)
        model = model.to(torch.float16)

        # Replace target Linear layers with empty _AWQ4Linear shells
        SKIP = {"speech_encoder", "speech_audio_adapter", "lm_head"}
        for name, mod in list(model.named_modules()):
            if not isinstance(mod, torch.nn.Linear):
                continue
            if any(s in name for s in SKIP) or "text_decoder" not in name:
                continue
            O, I = mod.weight.shape
            parts  = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], _AWQ4Linear(I, O, mod.bias is not None, group_size))

        # Load AWQ weights into the shells
        sd = torch.load(os.path.join(model_path, "model_awq4.pt"), map_location="cpu")
        model.load_state_dict(sd)
        model = model.to(device)
        print(f"  AWQ4 model loaded in {time.time()-t0:.1f}s")

    elif quant == "mlx4":
        print("Loading model → CPU FP16, will apply MLX-style int4 (group=64) post-hoc …")
        t0 = time.time()
        model = MERaLiON2ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            **common_kwargs,
        )
        _apply_mlx4_quant(model, group_size=64)
        model = model.to(device)

    elif quant in ("int8", "int4"):
        import bitsandbytes as bnb
        from torch import nn as _nn

        print(f"Loading model → CPU FP16, will apply BnB {quant.upper()} post-hoc …")
        t0 = time.time()
        model = MERaLiON2ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            **common_kwargs,
        )

        for mod_name, module in list(model.named_modules()):
            if not isinstance(module, _nn.Linear):
                continue
            if any(skip in mod_name for skip in BNB_SKIP):
                continue

            # Navigate to parent
            parts = mod_name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            child = parts[-1]

            w = module.weight.data.cpu()   # FP16 on CPU
            has_bias = module.bias is not None

            if quant == "int8":
                new_layer = bnb.nn.Linear8bitLt(
                    module.in_features, module.out_features,
                    bias=has_bias, has_fp16_weights=False, threshold=6.0,
                )
                new_layer.weight = bnb.nn.Int8Params(
                    w, requires_grad=False, has_fp16_weights=False)
            else:  # int4 NF4
                new_layer = bnb.nn.Linear4bit(
                    module.in_features, module.out_features,
                    bias=has_bias, quant_type="nf4",
                    compute_dtype=torch.bfloat16,
                )
                new_layer.weight = bnb.nn.Params4bit(
                    w, requires_grad=False, quant_type="nf4")

            if has_bias:
                new_layer.bias = _nn.Parameter(module.bias.data)
            setattr(parent, child, new_layer)

        # .to(device) calls Int8Params.cuda() / Params4bit.cuda() which
        # performs the actual weight quantization on GPU.
        model = model.to(device)

    else:
        dtype = torch.bfloat16 if quant == "bf16" else torch.float16
        attn_impl = "flash_attention_2" if flash_attn else "sdpa"
        print(f"Loading model {quant.upper()} (attn={attn_impl}) on GPU …")
        t0 = time.time()
        # Load to CPU first, then move to target device.
        # device_map=<device-string> is unreliable in older transformers +
        # meralion2_bl — model silently stays on CPU → 0.00 GB VRAM reported.
        try:
            model = MERaLiON2ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
                **common_kwargs,
            )
        except Exception as e:
            if flash_attn and "flash" in str(e).lower():
                print(f"  FlashAttn2 unavailable ({e}), falling back to sdpa …")
                model = MERaLiON2ForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    attn_implementation="sdpa",
                    **common_kwargs,
                )
            else:
                raise
        model = model.to(device)

    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model, processor


# ── inference ─────────────────────────────────────────────────────────────────

def transcribe_gpu(model, processor, audio_array: np.ndarray, sample_rate: int,
                   instruction: str = "Transcribe the speech",
                   max_new_tokens: int = 128,
                   device: str = "cuda",
                   speculative: bool = False,
                   gamma: int = 5,
                   ngram: "NGramDraft | None" = None) -> tuple:
    """Run ASR inference on GPU for a single audio sample.

    Uses the same audio preprocessing and input-building logic as
    infer_cpu.py:transcribe(), but moves tensors to CUDA and wraps
    the generate() call with CUDA sync for accurate wall-clock timing.

    Returns (text, stats) where stats mirrors the CPU version:
        n_tokens, decode_tps, [prefill_s, decode_s for BF16/FP16]
    """
    input_features, feature_attention_mask, n_speech = prepare_audio(
        audio_array, sample_rate, processor)

    tokenizer = processor.tokenizer
    speech_token_id = model.config.speech_token_index

    conversation = [{"role": "user",
                     "content": (f"Instruction: {instruction} \n"
                                 "Follow the text instruction based on the "
                                 "following audio: <SpeechHere>")}]
    prompt = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True)
    raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        pos = raw_ids.index(speech_token_id)
    except ValueError:
        raise RuntimeError(
            f"speech_token_id={speech_token_id} not in tokenized prompt.")

    input_ids = torch.tensor(
        [raw_ids[:pos] + [speech_token_id] * n_speech + raw_ids[pos + 1:]],
        dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    # Detect actual device from model parameters (BnB device_map="auto" may
    # place the model on cuda:0 regardless of the `device` argument).
    try:
        _actual_device = next(p.device for p in model.parameters()
                              if p.device.type != "cpu")
    except StopIteration:
        _actual_device = torch.device(device)

    # Detect compute dtype (BnB quantized params report int8/uint8;
    # fall back to bfloat16 as the safe compute dtype).
    try:
        _dtype = next(p.dtype for p in model.parameters()
                      if p.dtype in (torch.float16, torch.bfloat16))
    except StopIteration:
        _dtype = torch.bfloat16

    # Move inputs to the model's actual device
    input_ids              = input_ids.to(_actual_device)
    attention_mask         = attention_mask.to(_actual_device)
    input_features         = input_features.to(_actual_device).to(_dtype)
    feature_attention_mask = feature_attention_mask.to(_actual_device)

    # Pre-create cache to avoid the Gemma2 HybridCache overflow issue
    # (same workaround as infer_cpu.py:transcribe()).
    _gen_cfg = getattr(model, "generation_config", None)
    if _gen_cfg is not None:
        _gen_cfg.cache_implementation = None

    max_cache = input_ids.shape[1] + max_new_tokens
    if _model_is_pruned(model):
        from transformers import DynamicCache
        past_kv = DynamicCache()
    else:
        from transformers.cache_utils import HybridCache
        past_kv = HybridCache(
            model.text_decoder.model.config,
            max_batch_size=1,
            max_cache_len=max_cache,
            dtype=_dtype,
            device=_actual_device,
        )

    eos_ids = {
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<end_of_turn>"),
    }
    seq_len = input_ids.shape[1]

    if speculative:
        # Manual decode loop — n-gram context built from generated tokens ONLY.
        # prompt_lookup_num_tokens (HF built-in) searches the full input_ids which
        # contains ~100 repeated speech_token_id (255999) placeholders; those match
        # everywhere and always propose speech tokens → acceptance = 0.
        # This loop mirrors transcribe_native() in infer_cpu.py, ported to CUDA.
        ngram = ngram or NGramDraft()
        generated_ids = []
        n_spec_acc = n_spec_tot = 0

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            # Prefill
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                past_key_values=past_kv,
                use_cache=True,
                cache_position=torch.arange(0, seq_len, device=_actual_device),
                return_dict=True,
            )
            next_tok = int(out.logits[0, -1].argmax())
            generated_ids.append(next_tok)
            torch.cuda.synchronize()
            t1 = time.time()

            all_ctx = list(generated_ids)
            cur_pos = seq_len

            while len(generated_ids) < max_new_tokens:
                if next_tok in eos_ids:
                    break

                draft = ngram.propose(all_ctx, gamma)

                if draft:
                    K = min(len(draft), seq_len + max_new_tokens - cur_pos - 1)
                    if K <= 0:
                        draft = []
                if draft:
                    draft    = draft[:K]
                    spec_ids = torch.tensor([[next_tok] + draft],
                                            dtype=torch.long, device=_actual_device)
                    spec_attn = torch.ones(1, cur_pos + K + 1,
                                           dtype=torch.long, device=_actual_device)
                    spec_cpos = torch.arange(cur_pos, cur_pos + K + 1,
                                             device=_actual_device)

                    out = model(
                        input_ids=spec_ids,
                        attention_mask=spec_attn,
                        past_key_values=past_kv,
                        use_cache=True,
                        cache_position=spec_cpos,
                        return_dict=True,
                    )
                    n_spec_tot += K

                    n_acc = 0
                    stopped = False
                    for i in range(K):
                        if len(generated_ids) >= max_new_tokens:
                            stopped = True
                            break
                        pred = int(out.logits[0, i].argmax())
                        if pred == draft[i]:
                            generated_ids.append(draft[i])
                            all_ctx.append(draft[i])
                            n_acc += 1
                            n_spec_acc += 1
                            if draft[i] in eos_ids:
                                next_tok = draft[i]
                                stopped = True
                                break
                        else:
                            generated_ids.append(pred)
                            all_ctx.append(pred)
                            next_tok = pred
                            n_acc += 1
                            stopped = True
                            break

                    if not stopped and len(generated_ids) < max_new_tokens:
                        bonus = int(out.logits[0, K].argmax())
                        generated_ids.append(bonus)
                        all_ctx.append(bonus)
                        next_tok = bonus
                        n_acc += 1

                    valid_end = cur_pos + n_acc
                    if valid_end < cur_pos + K + 1:
                        _trim_kv_cache(past_kv, valid_end)
                    cur_pos = valid_end

                else:
                    cur_attn = torch.ones(1, cur_pos + 1,
                                          dtype=torch.long, device=_actual_device)
                    out = model(
                        input_ids=torch.tensor([[next_tok]],
                                               dtype=torch.long, device=_actual_device),
                        attention_mask=cur_attn,
                        past_key_values=past_kv,
                        use_cache=True,
                        cache_position=torch.tensor([cur_pos], device=_actual_device),
                        return_dict=True,
                    )
                    next_tok = int(out.logits[0, -1].argmax())
                    generated_ids.append(next_tok)
                    all_ctx.append(next_tok)
                    cur_pos += 1

            torch.cuda.synchronize()
            t2 = time.time()

        total_s    = t2 - t0
        generated  = generated_ids
        n_tokens   = max(len(generated), 1)
        decode_tps = max(len(generated) - 1, 1) / (t2 - t1) if t2 > t1 else 0.0
        stats = {"n_tokens": n_tokens, "decode_tps": decode_tps}
        if n_spec_tot > 0:
            stats["spec_accept_rate"] = n_spec_acc / n_spec_tot

        text = tokenizer.decode(generated, skip_special_tokens=True)
        return text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip(), stats

    # ── Non-speculative: use model.generate() ────────────────────────────────
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            past_key_values=past_kv,
            eos_token_id=list(eos_ids),
        )
    torch.cuda.synchronize()
    total_s = time.time() - t0

    generated  = output_ids[0][seq_len:]
    n_tokens   = max(len(generated), 1)
    decode_tps = n_tokens / total_s if total_s > 0 else 0.0
    stats = {"n_tokens": n_tokens, "decode_tps": decode_tps}

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip(), stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GPU ASR inference benchmark — MERaLiON-2")
    parser.add_argument("--model", required=True,
                        help="Model directory (original or pruned+tuned)")
    parser.add_argument("--audio", default=None,
                        help="Single audio file (.wav/.flac/.mp3)")
    parser.add_argument("--instruction", default="Transcribe the speech")
    parser.add_argument("--dataset", default=None,
                        help="IMDA_PART1_mono_en_30_ASR dataset path")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--quant", default="bf16",
                        choices=["bf16", "fp16", "int8", "int4", "mlx4", "awq4", "autoawq4"],
                        help="Quantization: bf16|fp16|int8|int4|mlx4|awq4|autoawq4")
    parser.add_argument("--no_flash_attn", action="store_true",
                        help="Use SDPA instead of FlashAttention-2")
    parser.add_argument("--device", default="cuda",
                        help="CUDA device, e.g. cuda / cuda:0 / cuda:1 (default: cuda)")
    parser.add_argument("--output", default="gpu_results.json")
    parser.add_argument("--save_samples", action="store_true",
                        help="Include per-sample predictions + references in JSON output")
    parser.add_argument("--speculative", action="store_true",
                        help="Enable n-gram prompt-lookup speculative decoding on GPU")
    parser.add_argument("--gamma", type=int, default=5,
                        help="Spec decoding lookahead window (default: 5)")
    parser.add_argument("--corpus", default=None,
                        help="Path to n-gram corpus .pkl built by build_ngram_corpus.py")
    parser.add_argument("--audiobench_norm", action="store_true",
                        help="Use AudioBench normalization for WER (digit→word, contractions, fillers)")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the text decoder (best with int8/int4, non-spec path)")
    args = parser.parse_args()
    args.model = os.path.abspath(args.model)

    if not torch.cuda.is_available():
        print("ERROR: no CUDA device found. Use infer_cpu.py for CPU inference.")
        sys.exit(1)

    torch.cuda.reset_peak_memory_stats(args.device)

    model, processor = load_model_gpu(
        args.model,
        quant=args.quant,
        flash_attn=not args.no_flash_attn,
        device=args.device,
    )

    gpu_mem_load_gb = torch.cuda.max_memory_allocated(args.device) / 1e9
    print(f"  GPU VRAM after load: {gpu_mem_load_gb:.2f} GB")

    if args.compile:
        print("torch.compile: compiling text_decoder …")
        model.text_decoder = torch.compile(
            model.text_decoder,
            mode="default",
            dynamic=True,
            fullgraph=False,
        )
        print("  done (first inference will trigger JIT compilation)")

    _ngram = None
    if args.speculative:
        if args.corpus:
            _ngram = NGramDraft.from_corpus_file(args.corpus)
        else:
            _ngram = NGramDraft()

    def _infer(audio, sr, instruction):
        return transcribe_gpu(model, processor, audio, sr,
                              instruction=instruction,
                              max_new_tokens=args.max_new_tokens,
                              device=args.device,
                              speculative=args.speculative,
                              gamma=args.gamma,
                              ngram=_ngram)

    # ── single audio file ──────────────────────────────────────────────────
    if args.audio:
        import soundfile as sf
        audio, sr = sf.read(args.audio)
        if audio.ndim == 2:
            audio = audio.mean(axis=-1)
        audio = audio.astype(np.float32)
        t0 = time.time()
        text, stats = _infer(audio, sr, args.instruction)
        print(f"\nTranscription ({time.time()-t0:.2f}s, {stats['decode_tps']:.1f} tok/s):\n  {text}")
        return

    # ── dataset benchmark + WER ────────────────────────────────────────────
    if args.dataset:
        from datasets import load_from_disk
        import evaluate

        data = load_from_disk(os.path.abspath(args.dataset))
        shuffled = data.shuffle(seed=42)
        # Clamp start so start + num_samples fits.  Small datasets
        # (< 10500) fall back to index 0.
        start = max(0, min(10500, len(shuffled) - args.num_samples))
        end   = min(start + args.num_samples, len(shuffled))
        subset = shuffled.select(range(start, end))

        def _extract_audio(sample):
            """Support two schemas for the `context` field:
              (a) nested:  sample['context']['audio'] = {array, sampling_rate}
              (b) flat:    sample['context'] = {array, sampling_rate}  (AudioBench)
            """
            ctx = sample.get("context") or {}
            if isinstance(ctx, dict) and "audio" in ctx and isinstance(ctx["audio"], dict):
                return ctx["audio"]
            return ctx

        def _extract_ref(sample):
            """Several (reference-transcript) fields seen across datasets."""
            oa = sample.get("other_attributes") or {}
            ref = oa.get("Transcription") or oa.get("transcription")
            if ref is None:
                ans = sample.get("answer")
                if isinstance(ans, dict):
                    ref = ans.get("text")
                else:
                    ref = ans
            return ref or ""

        # Warm up (first GPU call is slower due to kernel JIT)
        print("Warming up GPU …")
        _sample0 = subset[0]
        _ao = _extract_audio(_sample0)
        _a = np.asarray(_ao["array"], dtype=np.float32)
        _sr = _ao.get("sampling_rate", 16000)
        _instr = (_sample0["instruction"]["text"]
                  if isinstance(_sample0["instruction"], dict)
                  else _sample0["instruction"])
        _infer(_a, _sr, _instr)
        torch.cuda.reset_peak_memory_stats(args.device)

        predictions, references, latencies = [], [], []
        samples_out = []
        n_actual = len(subset)
        for i in range(n_actual):
            sample = subset[i]
            ao = _extract_audio(sample)
            audio = np.asarray(ao["array"], dtype=np.float32)
            sr    = ao.get("sampling_rate", 16000)
            if audio.ndim == 2:
                audio = audio.mean(axis=-1)
            instr = (sample["instruction"]["text"]
                     if isinstance(sample["instruction"], dict)
                     else sample["instruction"])
            ref = _extract_ref(sample)

            t0 = time.time()
            pred, stats = _infer(audio, sr, instr)
            elapsed = time.time() - t0
            predictions.append(pred)
            references.append(ref)
            latencies.append(elapsed)
            print(f"  [{i+1:3d}/{n_actual}] {elapsed:5.2f}s  "
                  f"{stats['decode_tps']:6.1f} tok/s | {pred[:60]}")
            entry = {"idx": i, "reference": ref, "prediction": pred,
                     "latency_s": elapsed, **stats}
            samples_out.append(entry)

        wer_metric = evaluate.load("wer")
        _norm = _normalize_text_audiobench if args.audiobench_norm else _normalize_text
        norm_preds = [_norm(p) for p in predictions]
        norm_refs  = [_norm(r) for r in references]
        wer        = wer_metric.compute(predictions=norm_preds, references=norm_refs)
        avg_lat    = float(np.mean(latencies))
        avg_tps    = float(np.mean([s["decode_tps"] for s in samples_out]))
        gpu_peak_gb = torch.cuda.max_memory_allocated(args.device) / 1e9
        acc_rates  = [s["spec_accept_rate"] for s in samples_out
                      if "spec_accept_rate" in s]
        avg_acc    = float(np.mean(acc_rates)) if acc_rates else None

        print(f"\n{'='*60}")
        print(f"WER:           {wer:.4f}  ({wer*100:.2f}%)  [normalized]")
        print(f"Avg latency:   {avg_lat:.2f} s/sample")
        print(f"Avg decode:    {avg_tps:.2f} tok/s")
        if avg_acc is not None:
            print(f"Spec acc rate: {avg_acc:.1%}")
        print(f"GPU VRAM peak: {gpu_peak_gb:.2f} GB")
        print(f"quant:         {args.quant}")
        print(f"device:        {args.device}")
        print(f"{'='*60}")

        result = {
            "model":              args.model,
            "quant_method":       args.quant,
            "device":             args.device,
            "num_samples":        args.num_samples,
            "speculative":        args.speculative,
            "gamma":              args.gamma if args.speculative else None,
            "compiled":           args.compile,
            "wer":                wer,
            "avg_latency_s":      avg_lat,
            "avg_decode_tps":     avg_tps,
            "avg_spec_accept_rate": avg_acc,
            "gpu_mem_load_gb":    gpu_mem_load_gb,
            "gpu_mem_peak_gb":    gpu_peak_gb,
            "latencies":          latencies,
        }
        if args.save_samples:
            result["samples"] = samples_out
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.output}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
