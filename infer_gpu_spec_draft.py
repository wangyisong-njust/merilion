#!/usr/bin/env python3
"""
Model-based speculative decoding for MERaLiON ASR.

Uses a pruned MERaLiON model (draft) to propose tokens; full model (verifier)
verifies them in one forward pass.  Both models share the same tokenizer.

Example:
    python infer_gpu_spec_draft.py \\
        --verifier /path/to/MERaLiON-2-3B \\
        --draft    meralion_tune_log/MERaLiON-2-3B-v3-td50-mid3-23-tune \\
        --verifier_quant bf16 --draft_quant int4 \\
        --dataset /path/to/IMDA_PART1_mono_en_30_ASR \\
        --num_samples 50 --gamma 5 --output draft_spec.json
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

# ── constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE             = 16000
CHUNK_SIZE              = SAMPLE_RATE * 30
MAX_CHUNKS              = 8
SPEECH_TOKENS_PER_CHUNK = 100


# ── MLX-style int4 quantization ──────────────────────────────────────────────

class _MLX4Linear(torch.nn.Module):
    """Per-group asymmetric int4 linear layer matching MLX affine quantization."""
    def __init__(self, in_features: int, out_features: int,
                 has_bias: bool, group_size: int = 64):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.group_size   = group_size
        self._pad = (-in_features) % group_size
        I_pad     = in_features + self._pad
        n_groups  = I_pad // group_size
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
        w        = module.weight.detach().float()
        I_pad    = I + new._pad
        n_groups = I_pad // group_size
        if new._pad:
            w = torch.nn.functional.pad(w, (0, new._pad))
        wg     = w.view(O, n_groups, group_size)
        w_min  = wg.min(dim=-1).values
        w_max  = wg.max(dim=-1).values
        scale  = (w_max - w_min) / 15.0
        scale  = scale.clamp(min=1e-8)
        zero   = w_min
        q = ((wg - zero.unsqueeze(-1)) / scale.unsqueeze(-1))
        q = q.round().clamp(0, 15).to(torch.uint8)
        q = q.view(O, I_pad)
        q_low  = q[:, 0::2]
        q_high = q[:, 1::2]
        packed = (q_high << 4) | q_low
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
        lo = (self.weight_q & 0x0F).to(torch.float16)
        hi = (self.weight_q >> 4).to(torch.float16)
        q  = torch.stack([lo, hi], dim=-1).view(O, I_pad)
        q  = q.view(O, n_groups, self.group_size)
        s  = self.scales.unsqueeze(-1)
        z  = self.zeros.unsqueeze(-1)
        w  = s * q + z
        w  = w.view(O, I_pad)[:, :self.in_features]
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w    = self._dequantize().to(x.dtype)
        bias = self.linear_bias.to(x.dtype) if self.linear_bias is not None else None
        return torch.nn.functional.linear(x, w, bias)


def _apply_mlx4_quant(model, group_size: int = 64) -> None:
    SKIP = {"speech_encoder", "speech_audio_adapter", "lm_head"}
    n_replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(s in name for s in SKIP):
            continue
        if "text_decoder" not in name:
            continue
        parts  = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1],
                _MLX4Linear.from_linear(module, group_size=group_size))
        n_replaced += 1
    print(f"  MLX4 quant: replaced {n_replaced} Linear layers (group_size={group_size})")


# ── AWQ W4A16 quantization (load-only) ───────────────────────────────────────

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


# ── helpers ───────────────────────────────────────────────────────────────────

def _model_is_pruned(model) -> bool:
    try:
        cfg = model.text_decoder.model.config
        return (getattr(cfg, "midblock_start", -1) >= 0
                and getattr(cfg, "midblock_ratio", 1.0) < 1.0)
    except Exception:
        return False


def _trim_kv_cache(past_kv, keep_len: int) -> None:
    """Discard KV cache entries beyond keep_len positions."""
    from transformers.cache_utils import DynamicCache
    if isinstance(past_kv, DynamicCache):
        for i in range(len(past_kv.key_cache)):
            past_kv.key_cache[i] = past_kv.key_cache[i][:, :, :keep_len, :]
            past_kv.value_cache[i] = past_kv.value_cache[i][:, :, :keep_len, :]
    else:  # HybridCache — zero out stale slots
        for i in range(len(past_kv.key_cache)):
            kc = past_kv.key_cache[i]
            if kc is not None and kc.shape[2] > keep_len:
                kc[:, :, keep_len:, :].zero_()
        for i in range(len(past_kv.value_cache)):
            vc = past_kv.value_cache[i]
            if vc is not None and vc.shape[2] > keep_len:
                vc[:, :, keep_len:, :].zero_()
    past_kv._seen_tokens = keep_len


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
    for digit, word in [("0", "zero"), ("1", "one"), ("2", "two"), ("3", "three"),
                        ("4", "four"), ("5", "five"), ("6", "six"), ("7", "seven"),
                        ("8", "eight"), ("9", "nine")]:
        text = re.sub(r'\b' + digit + r'\b', word, text)
    text = re.sub(r'[\(\[\{\<][^\n\(\)\[\]\{\}\<\>]*[\)\]\}\>]', "", text)
    text = _pipeline(text)
    text = re.sub(r'\b(uh|umm|um|er|ah)\b', '', text)
    return text.strip()


def prepare_audio(audio_array: np.ndarray, sample_rate: int, processor):
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


# ── model loading ─────────────────────────────────────────────────────────────

def load_model_gpu(model_path: str, quant: str = "bf16",
                   flash_attn: bool = True, device: str = "cuda"):
    """Load a MERaLiON model (bf16 / int8 / int4) on GPU."""
    from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration
    from transformers import AutoProcessor

    print(f"Loading processor from {os.path.basename(model_path)} …")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    BNB_SKIP = ["speech_encoder", "speech_audio_adapter", "lm_head"]
    common_kwargs = dict(use_safetensors=True)
    t0 = time.time()

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
        td_awq_dir = os.path.join(model_path, "text_decoder_awq")

        if _awq_cfg.get("pruned"):
            import glob as _glob
            import torch.nn as _nn
            from awq.modules.linear import WQLinear_GEMM
            from safetensors.torch import load_file as _lsf
            from transformers import AutoConfig as _AutoConfig2

            _src   = _awq_cfg["source_model"]
            _w_bit = _awq_cfg.get("w_bit", 4)
            _grp   = _awq_cfg.get("q_group_size", 128)

            # Load in BF16: speech encoder + adaptor use FP16 arithmetic which
            # overflows (max≈65504) and produces NaN that propagates into
            # key_cache[0] before any attention runs.  BF16 (max≈3.4e38)
            # is immune.  The WQLinear GEMM kernel is patched to receive FP16
            # (converted from BF16 on entry) and cast its FP16 output back to
            # BF16, so hidden states stay BF16 throughout the text decoder.
            print(f"  Loading pruned source model (BF16) …")
            _pruned_full = MERaLiON2ForConditionalGeneration.from_pretrained(
                _src, torch_dtype=torch.bfloat16, use_safetensors=True)
            _pruned_td = _pruned_full.text_decoder

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

            # Load qweight/qzeros/scales from checkpoint into WQLinear buffers
            _miss, _unex = _pruned_td.load_state_dict(_qsd, strict=False)
            _tied = {"lm_head.weight"}
            _bad  = [k for k in _miss if k not in _tied]
            if _bad:
                print(f"  WARNING: {len(_bad)} missing keys: {_bad[:4]}")
            _n_wq = sum(1 for m in _pruned_td.modules() if hasattr(m, "qweight"))
            print(f"  {_n_wq} WQLinear layers loaded")
            del _qsd

            # Patch every WQLinear_GEMM to keep hidden-state dtype (BF16).
            # AWQ kernels require FP16 tensors; we cast in→FP16, call the
            # original forward (handles GEMM/GEMV/marlin automatically), then
            # cast output back to the caller's dtype (BF16).
            import types as _types
            _wq_orig_fwd = WQLinear_GEMM.forward
            def _wq_fwd_bf16(self, x):
                _odtype = x.dtype
                if _odtype == torch.float16:
                    return _wq_orig_fwd(self, x)
                _out = _wq_orig_fwd(self, x.to(torch.float16))
                return _out.to(_odtype)

            for _m in _pruned_td.modules():
                if isinstance(_m, WQLinear_GEMM):
                    _m.forward = _types.MethodType(_wq_fwd_bf16, _m)

            # Force eager attention so tanh softcap is applied before softmax.
            # SDPA skips softcap; without it large FP16 scores → inf → NaN.
            _pruned_td.config._attn_implementation = "eager"
            _pruned_td.model.config._attn_implementation = "eager"

            # Reuse the already-loaded _pruned_full as the model shell: its
            # speech_encoder / speech_audio_adapter / ln_speech are BF16 with
            # correct weights.  Just swap in the quantized text_decoder.
            # (Rebuilding a fresh shell from model_path's config and reloading
            #  via strict=False silently dropped speech encoder weights whose
            #  names differed → random init → NaN.)
            model = _pruned_full
            model.text_decoder = _pruned_td
            model = model.to(device)
            del _pruned_full
            torch.cuda.empty_cache()
            print(f"  AWQ quantized text decoder loaded ✓ (BF16 shell + eager attn)")
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

    elif quant == "awq4":
        import json as _json
        from transformers import AutoConfig as _AutoConfig
        cfg_path = os.path.join(model_path, "awq_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"awq_config.json not found in {model_path}. Run quantize_awq.py first.")
        with open(cfg_path) as f:
            awq_cfg = _json.load(f)
        group_size = awq_cfg.get("group_size", 64)
        print(f"Loading pre-quantized AWQ4 model from {os.path.basename(model_path)} (group={group_size}) …")
        # AWQ4 dir has no safetensors — init model from config, then load state dict.
        _hf_cfg = _AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = MERaLiON2ForConditionalGeneration(_hf_cfg)
        model = model.to(torch.float16)
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
        sd = torch.load(os.path.join(model_path, "model_awq4.pt"), map_location="cpu")
        model.load_state_dict(sd)
        model = model.to(device)

    elif quant == "mlx4":
        print(f"Loading {os.path.basename(model_path)} → CPU FP16, applying MLX4 (group=64) …")
        model = MERaLiON2ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, **common_kwargs)
        _apply_mlx4_quant(model, group_size=64)
        model = model.to(device)

    elif quant in ("int8", "int4"):
        import bitsandbytes as bnb
        from torch import nn as _nn

        print(f"Loading {os.path.basename(model_path)} → CPU FP16 then BnB {quant.upper()} …")
        model = MERaLiON2ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, **common_kwargs)

        for mod_name, module in list(model.named_modules()):
            if not isinstance(module, _nn.Linear):
                continue
            if any(skip in mod_name for skip in BNB_SKIP):
                continue
            parts = mod_name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            child = parts[-1]
            w = module.weight.data.cpu()
            has_bias = module.bias is not None
            if quant == "int8":
                new_layer = bnb.nn.Linear8bitLt(
                    module.in_features, module.out_features,
                    bias=has_bias, has_fp16_weights=False, threshold=6.0)
                new_layer.weight = bnb.nn.Int8Params(
                    w, requires_grad=False, has_fp16_weights=False)
            else:
                new_layer = bnb.nn.Linear4bit(
                    module.in_features, module.out_features,
                    bias=has_bias, quant_type="nf4",
                    compute_dtype=torch.bfloat16)
                new_layer.weight = bnb.nn.Params4bit(
                    w, requires_grad=False, quant_type="nf4")
            if has_bias:
                new_layer.bias = _nn.Parameter(module.bias.data)
            setattr(parent, child, new_layer)
        model = model.to(device)

    else:
        dtype = torch.bfloat16 if quant == "bf16" else torch.float16
        attn_impl = "flash_attention_2" if flash_attn else "sdpa"
        print(f"Loading {os.path.basename(model_path)} {quant.upper()} (attn={attn_impl}) …")
        try:
            model = MERaLiON2ForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=dtype,
                attn_implementation=attn_impl, **common_kwargs)
        except Exception as e:
            if flash_attn and "flash" in str(e).lower():
                model = MERaLiON2ForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=dtype, **common_kwargs)
            else:
                raise
        model = model.to(device)

    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return model, processor


# ── core inference ────────────────────────────────────────────────────────────

def transcribe_gpu_draft_spec(
        verifier, draft_model, processor,
        audio_array: np.ndarray, sample_rate: int,
        instruction: str = "Transcribe the speech",
        max_new_tokens: int = 128,
        device: str = "cuda",
        gamma: int = 5,
        debug: bool = False) -> tuple:
    """Speculative decoding: draft proposes, verifier accepts/rejects.

    Both models share the same tokenizer and see the same audio prefill.
    Draft proposes up to gamma tokens greedily; verifier verifies K+1
    in one forward pass.  No bonus token — avoids draft KV sync issues.

    Returns (text, stats) where stats = {n_tokens, decode_tps, spec_accept_rate}.
    """
    input_features, feature_attention_mask, n_speech = prepare_audio(
        audio_array, sample_rate, processor)

    tokenizer = processor.tokenizer
    speech_token_id = verifier.config.speech_token_index

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
        raise RuntimeError(f"speech_token_id={speech_token_id} not found in prompt.")

    input_ids = torch.tensor(
        [raw_ids[:pos] + [speech_token_id] * n_speech + raw_ids[pos + 1:]],
        dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    # Detect actual device from verifier weights
    try:
        _dev = next(p.device for p in verifier.parameters() if p.device.type != "cpu")
    except StopIteration:
        _dev = torch.device(device)

    try:
        _dtype_v = next(p.dtype for p in verifier.parameters()
                        if p.dtype in (torch.float16, torch.bfloat16))
    except StopIteration:
        _dtype_v = torch.bfloat16
    try:
        _dtype_d = next(p.dtype for p in draft_model.parameters()
                        if p.dtype in (torch.float16, torch.bfloat16))
    except StopIteration:
        _dtype_d = torch.bfloat16

    input_ids              = input_ids.to(_dev)
    attention_mask         = attention_mask.to(_dev)
    input_features_v       = input_features.to(_dev).to(_dtype_v)
    # draft speech encoder weights are BF16 (loaded via torch_dtype=bfloat16);
    # use BF16 regardless of _dtype_d to avoid FP16 overflow in the encoder.
    _d_feat_dtype = next(
        (p.dtype for p in draft_model.speech_encoder.parameters()), _dtype_d)
    input_features_d       = input_features.to(_dev).to(_d_feat_dtype)
    feature_attention_mask = feature_attention_mask.to(_dev)

    # Disable HybridCache auto-selection in generation_config
    for mdl in (verifier, draft_model):
        _gen_cfg = getattr(mdl, "generation_config", None)
        if _gen_cfg is not None and hasattr(_gen_cfg, "cache_implementation"):
            _gen_cfg.cache_implementation = None

    seq_len   = input_ids.shape[1]
    max_cache = seq_len + max_new_tokens

    def _make_cache(mdl, dtype):
        # Pruned models have non-uniform num_key_value_heads (mid-block layers
        # are halved), so HybridCache (which pre-allocates per config's uniform
        # kv_heads) gives a shape mismatch.  DynamicCache is used instead;
        # meralion2_bl already selects DynamicCache for pruned models internally.
        if _model_is_pruned(mdl):
            from transformers import DynamicCache
            return DynamicCache()
        from transformers.cache_utils import HybridCache
        return HybridCache(
            mdl.text_decoder.model.config,
            max_batch_size=1,
            max_cache_len=max_cache,
            dtype=dtype,
            device=_dev,
        )

    verifier_kv = _make_cache(verifier, _dtype_v)
    draft_kv    = _make_cache(draft_model, _dtype_d)

    eos_ids = {tokenizer.eos_token_id,
               tokenizer.convert_tokens_to_ids("<end_of_turn>")}
    eos_ids.discard(None)

    generated_ids = []
    n_spec_acc = n_spec_tot = 0

    torch.cuda.synchronize()
    t0 = time.time()

    with torch.inference_mode():
        # ── Prefill both models ────────────────────────────────────────────
        _common = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            feature_attention_mask=feature_attention_mask,
            use_cache=True,
            cache_position=torch.arange(0, seq_len, device=_dev),
            return_dict=True,
        )
        v_out = verifier(**_common,
                         input_features=input_features_v,
                         past_key_values=verifier_kv)

        if debug:
            _probe = {}
            def _mk_hook(tag):
                def _h(mod, inp, out):
                    x = inp[0] if isinstance(inp, tuple) else inp
                    y = out[0] if isinstance(out, tuple) else out
                    _probe[tag] = {
                        "in_nan": bool(torch.isnan(x).any().item()) if torch.is_tensor(x) else None,
                        "in_max": float(x.abs().max().item()) if torch.is_tensor(x) else None,
                        "in_dtype": str(x.dtype) if torch.is_tensor(x) else None,
                        "out_nan": bool(torch.isnan(y).any().item()) if torch.is_tensor(y) else None,
                        "out_max": float(y.float().abs().max().item()) if torch.is_tensor(y) and not torch.isnan(y).any() else float("nan"),
                        "out_dtype": str(y.dtype) if torch.is_tensor(y) else None,
                    }
                return _h
            _dtd = draft_model.text_decoder
            _hooks = []
            _hooks.append(_dtd.model.embed_tokens.register_forward_hook(_mk_hook("embed")))
            _l0 = _dtd.model.layers[0]
            _hooks.append(_l0.input_layernorm.register_forward_hook(_mk_hook("L0.pre_ln")))
            _hooks.append(_l0.self_attn.k_proj.register_forward_hook(_mk_hook("L0.k_proj")))
            _hooks.append(_l0.self_attn.q_proj.register_forward_hook(_mk_hook("L0.q_proj")))

        draft_model(**_common,
                    input_features=input_features_d,
                    past_key_values=draft_kv)

        if debug:
            for _h in _hooks: _h.remove()
            for _tag, _v in _probe.items():
                print(f"  [PROBE] {_tag}: {_v}")

        # next_tok = role-prefix token (e.g. first token of <Speaker1>:).
        # Append it so the full "<Speaker1>:" string is in generated_ids;
        # the explicit replace() below strips it, matching infer_gpu.py behaviour.
        next_tok = int(v_out.logits[0, -1].argmax())
        generated_ids.append(next_tok)

        torch.cuda.synchronize()
        t1 = time.time()

        if debug:
            _kc = getattr(draft_kv, "key_cache", [])
            _any_nan = any(
                torch.isnan(k).any().item()
                for k in _kc if k is not None and k.numel() > 0
            )
            _shapes = [tuple(k.shape) for k in _kc if k is not None and k.numel() > 0]
            print(f"  [DIAG] draft KV cache after prefill: layers={len(_kc)} "
                  f"any_nan={_any_nan} shapes[0]={_shapes[0] if _shapes else 'N/A'}")

        cur_pos = seq_len

        # ── Speculative decode loop ────────────────────────────────────────
        while len(generated_ids) < max_new_tokens:
            if next_tok in eos_ids:
                break

            # Step 1: draft proposes up to gamma tokens
            draft_tokens = []
            d_tok, d_pos = next_tok, cur_pos

            _diag_first_draft = debug and len(generated_ids) == 1
            for _di in range(gamma):
                if d_tok in eos_ids:
                    break
                d_out = draft_model(
                    input_ids=torch.tensor([[d_tok]], dtype=torch.long, device=_dev),
                    attention_mask=torch.ones(1, d_pos + 1, dtype=torch.long, device=_dev),
                    past_key_values=draft_kv,
                    use_cache=True,
                    cache_position=torch.tensor([d_pos], device=_dev),
                    return_dict=True,
                )
                _dl = d_out.logits[0, -1]
                if _diag_first_draft and _di == 0:
                    print(f"  [DIAG] draft logits step0: "
                          f"nan={torch.isnan(_dl).any().item()} "
                          f"inf={torch.isinf(_dl).any().item()} "
                          f"max={_dl.max().item():.4f} "
                          f"min={_dl.min().item():.4f} "
                          f"std={_dl.std().item():.4f} "
                          f"argmax={int(_dl.argmax())}")
                    _etok = draft_model.text_decoder.model.embed_tokens
                    print(f"  [DIAG] embed_tokens wt max="
                          f"{_etok.weight.abs().max().item():.4f}")
                d_tok = int(_dl.argmax())
                draft_tokens.append(d_tok)
                d_pos += 1

            K = min(len(draft_tokens), max_cache - cur_pos - 1)
            draft_tokens = draft_tokens[:K]

            if K == 0:
                # Draft immediately hit EOS — fall back to one greedy verifier step
                v_out = verifier(
                    input_ids=torch.tensor([[next_tok]], dtype=torch.long, device=_dev),
                    attention_mask=torch.ones(1, cur_pos + 1, dtype=torch.long, device=_dev),
                    past_key_values=verifier_kv,
                    use_cache=True,
                    cache_position=torch.tensor([cur_pos], device=_dev),
                    return_dict=True,
                )
                next_tok = int(v_out.logits[0, -1].argmax())
                generated_ids.append(next_tok)
                _trim_kv_cache(draft_kv, cur_pos + 1)
                cur_pos += 1
                continue

            # Step 2: verifier verifies [next_tok, d0..dK-1] in one pass
            spec_ids  = torch.tensor([[next_tok] + draft_tokens],
                                     dtype=torch.long, device=_dev)
            spec_attn = torch.ones(1, cur_pos + K + 1, dtype=torch.long, device=_dev)
            spec_cpos = torch.arange(cur_pos, cur_pos + K + 1, device=_dev)

            v_out = verifier(
                input_ids=spec_ids,
                attention_mask=spec_attn,
                past_key_values=verifier_kv,
                use_cache=True,
                cache_position=spec_cpos,
                return_dict=True,
            )
            n_spec_tot += K

            # Debug: print first round's draft vs verifier predictions
            if debug and len(generated_ids) == 1:
                v_preds = [int(v_out.logits[0, i].argmax()) for i in range(K)]
                print(f"\n[DEBUG round 1]")
                # Draft model structure
                _fwq = next((m for m in draft_model.modules()
                             if hasattr(m, "qweight")), None)
                if _fwq is None:
                    print(f"  [DIAG] draft has NO WQLinear modules (all FP16 Linear)")
                else:
                    _nz = (_fwq.qweight != 0).sum().item()
                    print(f"  [DIAG] draft WQLinear qweight: "
                          f"nonzero={_nz}/{_fwq.qweight.numel()} "
                          f"dev={_fwq.qweight.device}")
                    _snz = (_fwq.scales != 0).sum().item()
                    print(f"  [DIAG] draft WQLinear scales: "
                          f"nonzero={_snz}/{_fwq.scales.numel()} "
                          f"dtype={_fwq.scales.dtype}")
                    _znz = (_fwq.qzeros != 0).sum().item()
                    print(f"  [DIAG] draft WQLinear qzeros: "
                          f"nonzero={_znz}/{_fwq.qzeros.numel()}")
                    with torch.no_grad():
                        _xi = torch.randn(1, 1, _fwq.in_features,
                                          device=_fwq.qweight.device, dtype=torch.float16)
                        _yo = _fwq(_xi)
                        print(f"  [DIAG] WQLinear fwd test: "
                              f"out_nonzero={(_yo != 0).sum().item()}/{_yo.numel()} "
                              f"max={_yo.abs().max().item():.4f}")
                print(f"  next_tok={next_tok!r}  -> {tokenizer.decode([next_tok])!r}")
                print(f"  draft  : {draft_tokens} -> {tokenizer.decode(draft_tokens)!r}")
                print(f"  vpreds : {v_preds}  -> {tokenizer.decode(v_preds)!r}")
                d_logits_sample = v_out.logits[0, 0]
                print(f"  v_logits[0,0] nan={torch.isnan(d_logits_sample).any().item()} "
                      f"inf={torch.isinf(d_logits_sample).any().item()} "
                      f"max={d_logits_sample.max().item():.2f}")

            # Step 3: greedy acceptance (no bonus — keeps draft/verifier KV in sync)
            n_acc   = 0
            stopped = False
            for i in range(K):
                if len(generated_ids) >= max_new_tokens:
                    stopped = True
                    break
                pred = int(v_out.logits[0, i].argmax())
                if pred == draft_tokens[i]:
                    generated_ids.append(draft_tokens[i])
                    n_acc += 1
                    n_spec_acc += 1
                    next_tok = draft_tokens[i]
                    if draft_tokens[i] in eos_ids:
                        stopped = True
                        break
                else:
                    generated_ids.append(pred)
                    n_acc += 1
                    next_tok = pred
                    stopped = True
                    break

            # Step 4: trim BOTH caches to cur_pos + n_acc
            valid_end = cur_pos + n_acc
            _trim_kv_cache(verifier_kv, valid_end)
            _trim_kv_cache(draft_kv,    valid_end)
            cur_pos = valid_end

    torch.cuda.synchronize()
    t2 = time.time()

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    text = text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()
    n_tokens   = max(len(generated_ids), 1)
    decode_tps = max(len(generated_ids) - 1, 1) / (t2 - t1) if t2 > t1 else 0.0
    stats = {
        "n_tokens":          n_tokens,
        "decode_tps":        decode_tps,
        "spec_accept_rate":  n_spec_acc / n_spec_tot if n_spec_tot > 0 else 0.0,
    }
    return text, stats


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Model-based speculative decoding benchmark for MERaLiON ASR")
    parser.add_argument("--verifier", required=True,
                        help="Path to verifier (full) MERaLiON model")
    parser.add_argument("--draft", required=True,
                        help="Path to draft (pruned) MERaLiON model")
    parser.add_argument("--verifier_quant", default="bf16",
                        choices=["bf16", "fp16", "int8", "int4", "mlx4", "awq4", "autoawq4"],
                        help="Verifier quantization (default: bf16)")
    parser.add_argument("--draft_quant", default="autoawq4",
                        choices=["bf16", "fp16", "int8", "int4", "mlx4", "awq4", "autoawq4"],
                        help="Draft quantization (default: autoawq4)")
    parser.add_argument("--gamma", type=int, default=5,
                        help="Max draft tokens per step (default: 5)")
    parser.add_argument("--dataset", default=None,
                        help="HuggingFace dataset path (load_from_disk)")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--audiobench_norm", action="store_true",
                        help="Use AudioBench WER normalization")
    parser.add_argument("--output", default="draft_spec_results.json")
    parser.add_argument("--save_samples", action="store_true")
    parser.add_argument("--debug", action="store_true",
                        help="Print first-round draft vs verifier tokens for diagnosis")
    args = parser.parse_args()
    args.verifier = os.path.abspath(args.verifier)
    args.draft    = os.path.abspath(args.draft)

    if not torch.cuda.is_available():
        print("ERROR: no CUDA device found.")
        sys.exit(1)

    torch.cuda.reset_peak_memory_stats(args.device)

    verifier, processor = load_model_gpu(
        args.verifier, quant=args.verifier_quant, device=args.device)
    gpu_mem_verifier_gb = torch.cuda.max_memory_allocated(args.device) / 1e9
    print(f"  VRAM after verifier load: {gpu_mem_verifier_gb:.2f} GB")

    draft_model, _ = load_model_gpu(
        args.draft, quant=args.draft_quant, flash_attn=False, device=args.device)
    gpu_mem_load_gb = torch.cuda.max_memory_allocated(args.device) / 1e9
    print(f"  VRAM after both loads:    {gpu_mem_load_gb:.2f} GB")

    def _infer(audio, sr, instr):
        return transcribe_gpu_draft_spec(
            verifier, draft_model, processor, audio, sr,
            instruction=instr,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            gamma=args.gamma,
            debug=args.debug,
        )

    if not args.dataset:
        parser.print_help()
        return

    from datasets import load_from_disk
    import evaluate

    data     = load_from_disk(os.path.abspath(args.dataset))
    shuffled = data.shuffle(seed=42)
    start    = min(10500, len(shuffled))
    end      = min(start + args.num_samples, len(shuffled))
    subset   = shuffled.select(range(start, end))

    print("Warming up GPU …")
    s0    = subset[0]
    _a    = np.asarray(s0["context"]["audio"]["array"], dtype=np.float32)
    _sr   = s0["context"]["audio"]["sampling_rate"]
    _ins  = (s0["instruction"]["text"]
             if isinstance(s0["instruction"], dict) else s0["instruction"])
    _infer(_a, _sr, _ins)
    torch.cuda.reset_peak_memory_stats(args.device)

    predictions, references, latencies = [], [], []
    samples_out = []
    n_actual = len(subset)

    for i in range(n_actual):
        sample = subset[i]
        audio  = np.asarray(sample["context"]["audio"]["array"], dtype=np.float32)
        sr     = sample["context"]["audio"]["sampling_rate"]
        if audio.ndim == 2:
            audio = audio.mean(axis=-1)
        instr = (sample["instruction"]["text"]
                 if isinstance(sample["instruction"], dict)
                 else sample["instruction"])
        ref = sample["other_attributes"]["Transcription"]

        t0 = time.time()
        pred, stats = _infer(audio, sr, instr)
        elapsed = time.time() - t0
        predictions.append(pred)
        references.append(ref)
        latencies.append(elapsed)
        acc = stats.get("spec_accept_rate", 0)
        print(f"  [{i+1:3d}/{n_actual}] {elapsed:5.2f}s  "
              f"{stats['decode_tps']:6.1f} tok/s  acc={acc:.0%} | {pred[:55]}")
        samples_out.append({"idx": i, "reference": ref, "prediction": pred,
                            "latency_s": elapsed, **stats})

    wer_metric = evaluate.load("wer")
    _norm      = _normalize_text_audiobench if args.audiobench_norm else _normalize_text
    norm_preds = [_norm(p) for p in predictions]
    norm_refs  = [_norm(r) for r in references]
    wer        = wer_metric.compute(predictions=norm_preds, references=norm_refs)
    avg_lat    = float(np.mean(latencies))
    avg_tps    = float(np.mean([s["decode_tps"] for s in samples_out]))
    gpu_peak_gb = torch.cuda.max_memory_allocated(args.device) / 1e9
    acc_rates  = [s["spec_accept_rate"] for s in samples_out]
    avg_acc    = float(np.mean(acc_rates))

    print(f"\n{'='*60}")
    print(f"WER:             {wer:.4f}  ({wer*100:.2f}%)  [normalized]")
    print(f"Avg latency:     {avg_lat:.2f} s/sample")
    print(f"Avg decode:      {avg_tps:.2f} tok/s")
    print(f"Spec acc rate:   {avg_acc:.1%}")
    print(f"GPU VRAM peak:   {gpu_peak_gb:.2f} GB")
    print(f"verifier_quant:  {args.verifier_quant}")
    print(f"draft_quant:     {args.draft_quant}")
    print(f"gamma:           {args.gamma}")
    print(f"{'='*60}")

    result = {
        "verifier":           args.verifier,
        "draft":              args.draft,
        "verifier_quant":     args.verifier_quant,
        "draft_quant":        args.draft_quant,
        "device":             args.device,
        "gamma":              args.gamma,
        "num_samples":        n_actual,
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


main()
