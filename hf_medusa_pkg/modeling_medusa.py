"""Medusa speculative-decoding adapter for MERaLiON-2-3B.

Loads a Medusa head checkpoint trained to predict future tokens from the
last-layer hidden state of MERaLiON-2-3B's text decoder.  Attaches the heads
to a frozen base model and exposes `generate_medusa()` for accelerated
greedy decoding.

Usage:
    from transformers import AutoProcessor, AutoModelForCausalLM
    from modeling_medusa import MERaLiON2MedusaForASR

    model = MERaLiON2MedusaForASR.from_pretrained(
        "YOUR_HF_USERNAME/MERaLiON-2-3B-Medusa",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    processor = AutoProcessor.from_pretrained("MERaLiON/MERaLiON-2-3B",
                                              trust_remote_code=True)
    # ... prepare audio / prompt like the base model ...
    out_ids = model.generate_medusa(input_ids=..., attention_mask=...,
                                    input_features=..., feature_attention_mask=...,
                                    max_new_tokens=128)
"""
import json
import os
from typing import Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file as _load_safetensors


class _ResBlock(nn.Module):
    """y = x + SiLU(W x).  Must match the module used during training."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class _MedusaHead(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.mlp = nn.Sequential(
            *[_ResBlock(hidden_size) for _ in range(num_layers)]
        )

    def forward(self, h):
        return self.mlp(h)


class _MedusaHeads(nn.Module):
    """Bundle of K heads plus a (non-parameter) reference to the base model's
    lm_head, which the heads share.  ``forward(h)`` returns logits of shape
    ``(K, B, T, V)``."""
    def __init__(self, num_heads: int, hidden_size: int, num_layers: int,
                 lm_head: nn.Linear):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.heads = nn.ModuleList([
            _MedusaHead(hidden_size, num_layers=num_layers)
            for _ in range(num_heads)
        ])
        self._lm_head = lm_head

    def forward(self, h):
        w = self._lm_head.weight
        outs = []
        for head in self.heads:
            refined = head(h)
            logits = refined @ w.t()
            outs.append(logits)
        return torch.stack(outs, dim=0)


class MERaLiON2MedusaForASR(nn.Module):
    """Wrapper that loads MERaLiON-2-3B as base and attaches Medusa heads.

    Prefer `from_pretrained(<medusa_repo>)` — it reads ``adapter_config.json``,
    downloads the base model referenced there, then the Medusa head weights.
    """
    def __init__(self, base_model, medusa_heads: _MedusaHeads,
                 adapter_config: dict):
        super().__init__()
        self.base = base_model
        self.medusa = medusa_heads
        self.adapter_config = adapter_config
        # Convenience passthroughs
        self.config = base_model.config
        self.generation_config = getattr(base_model, "generation_config", None)

    # ── loading ────────────────────────────────────────────────────────────
    @classmethod
    def from_pretrained(cls, repo_or_path: str, torch_dtype=None,
                        trust_remote_code: bool = True, device_map=None, **kw):
        # Resolve adapter_config and heads weight file.
        # Accept both a local directory and an HF Hub repo id.
        if os.path.isdir(repo_or_path):
            cfg_path   = os.path.join(repo_or_path, "adapter_config.json")
            heads_path = os.path.join(repo_or_path, "medusa_heads.safetensors")
        else:
            from huggingface_hub import hf_hub_download
            cfg_path   = hf_hub_download(repo_or_path, "adapter_config.json")
            heads_path = hf_hub_download(repo_or_path, "medusa_heads.safetensors")
        with open(cfg_path) as f:
            acfg = json.load(f)

        base_id = acfg["base_model_name_or_path"]
        print(f"[Medusa] loading base model: {base_id}")
        # MERaLiON-2's upstream HF code is out-of-sync with recent transformers
        # API (missing `_supports_sdpa` attribute).  We bundle a patched copy
        # of the MERaLiON modeling code in `meralion2_bl/` next to this file,
        # and use it directly.  Users never need to set trust_remote_code.
        _pkg_dir = os.path.dirname(os.path.abspath(__file__))
        import sys as _sys
        if _pkg_dir not in _sys.path:
            _sys.path.insert(0, _pkg_dir)
        from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration
        from meralion2_bl.configuration_meralion2 import MERaLiON2Config

        # The base_id may be an HF Hub repo or a local directory.  In both
        # cases we load the config / weights but construct the model class
        # from our bundled code so transformers version mismatches don't bite.
        if os.path.isdir(base_id):
            base_local_dir = base_id
        else:
            from huggingface_hub import snapshot_download
            base_local_dir = snapshot_download(
                base_id,
                allow_patterns=[
                    "config.json", "generation_config.json",
                    "*.safetensors", "*.safetensors.index.json",
                    "tokenizer*", "special_tokens_map.json",
                    "preprocessor_config.json", "processor_config.json",
                    "chat_template.jinja",
                ],
            )
        base = MERaLiON2ForConditionalGeneration.from_pretrained(
            base_local_dir,
            config=MERaLiON2Config.from_pretrained(base_local_dir),
            torch_dtype=torch_dtype, device_map=device_map,
        )
        base.eval()
        for p in base.parameters():
            p.requires_grad_(False)

        # Instantiate heads and load weights.
        hidden_size = acfg["hidden_size"]
        num_heads   = acfg["num_heads"]
        num_layers  = acfg["num_layers"]
        lm_head = base.text_decoder.lm_head
        heads = _MedusaHeads(num_heads=num_heads, hidden_size=hidden_size,
                             num_layers=num_layers, lm_head=lm_head)
        sd = _load_safetensors(heads_path)
        heads.load_state_dict(sd, strict=False)
        heads = heads.to(lm_head.weight.device).to(lm_head.weight.dtype)
        heads._lm_head = lm_head
        heads.eval()

        print(f"[Medusa] loaded K={num_heads} heads, "
              f"{sum(p.numel() for p in heads.heads.parameters())/1e6:.1f} M params")

        return cls(base, heads, acfg)

    # ── inference ──────────────────────────────────────────────────────────
    @torch.inference_mode()
    def generate_medusa(self, input_ids, attention_mask,
                        input_features=None, feature_attention_mask=None,
                        max_new_tokens: int = 128,
                        eos_token_ids=None):
        """Greedy Medusa-speculative generation for ASR.

        Accepts the same audio/text inputs as the base model's forward; returns
        a LongTensor of shape (1, seq_len + generated_len)."""
        from transformers.cache_utils import HybridCache

        base = self.base
        td   = base.text_decoder
        K    = self.medusa.num_heads
        device = input_ids.device
        dtype  = td.lm_head.weight.dtype

        if eos_token_ids is None:
            # Sensible defaults for Gemma2-tokenizer based MERaLiON-2-3B.
            tok = getattr(self, "tokenizer", None)
            eos_token_ids = []
            if tok is not None:
                if tok.eos_token_id is not None:
                    eos_token_ids.append(tok.eos_token_id)
                x = tok.convert_tokens_to_ids("<end_of_turn>")
                if x is not None:
                    eos_token_ids.append(x)
            else:
                eos_token_ids = [1, 107]   # fallback ids for MERaLiON-2-3B
        eos_ids = set(eos_token_ids)

        seq_len   = input_ids.shape[1]
        max_cache = seq_len + max_new_tokens
        past_kv   = HybridCache(td.model.config, max_batch_size=1,
                                max_cache_len=max_cache, dtype=dtype, device=device)

        # Prefill (base model path handles speech encoder + adapter + decoder).
        out = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            past_key_values=past_kv,
            use_cache=True,
            cache_position=torch.arange(0, seq_len, device=device),
            output_hidden_states=True,
            return_dict=True,
        )
        h_last   = out.hidden_states[-1][0, -1:, :]
        next_tok = int(out.logits[0, -1].argmax())
        generated = [next_tok]
        cur_pos = seq_len

        while len(generated) < max_new_tokens and next_tok not in eos_ids:
            # Heads propose K draft tokens in parallel.
            draft_logits = self.medusa(h_last.unsqueeze(0))   # (K, 1, 1, V)
            draft_tokens = []
            for k in range(K):
                tk = int(draft_logits[k, 0, 0].argmax())
                draft_tokens.append(tk)
            # Cap at remaining cache.
            K_draft = min(len(draft_tokens), max_cache - cur_pos - 1)
            draft_tokens = draft_tokens[:K_draft]
            # Truncate at first EOS.
            for i, t in enumerate(draft_tokens):
                if t in eos_ids:
                    draft_tokens = draft_tokens[: i + 1]
                    break
            K_eff = len(draft_tokens)

            if K_eff == 0:
                v_out = td(
                    input_ids=torch.tensor([[next_tok]], dtype=torch.long, device=device),
                    attention_mask=torch.ones(1, cur_pos + 1, dtype=torch.long, device=device),
                    past_key_values=past_kv,
                    use_cache=True,
                    cache_position=torch.tensor([cur_pos], device=device),
                    output_hidden_states=True, return_dict=True,
                )
                h_last = v_out.hidden_states[-1][0, -1:, :]
                next_tok = int(v_out.logits[0, -1].argmax())
                generated.append(next_tok)
                cur_pos += 1
                continue

            # Batched verify of [next_tok, d0..d_{K_eff-1}].
            spec_ids  = torch.tensor([[next_tok] + draft_tokens],
                                     dtype=torch.long, device=device)
            spec_attn = torch.ones(1, cur_pos + K_eff + 1,
                                   dtype=torch.long, device=device)
            spec_cpos = torch.arange(cur_pos, cur_pos + K_eff + 1, device=device)
            v_out = td(
                input_ids=spec_ids, attention_mask=spec_attn,
                past_key_values=past_kv, use_cache=True, cache_position=spec_cpos,
                output_hidden_states=True, return_dict=True,
            )

            n_acc = 0
            stopped = False
            for i in range(K_eff):
                if len(generated) >= max_new_tokens:
                    stopped = True; break
                pred = int(v_out.logits[0, i].argmax())
                if pred == draft_tokens[i]:
                    generated.append(draft_tokens[i]); n_acc += 1
                    next_tok = draft_tokens[i]
                    if draft_tokens[i] in eos_ids:
                        stopped = True; break
                else:
                    generated.append(pred); n_acc += 1
                    next_tok = pred
                    stopped = True; break

            if n_acc == K_eff and not stopped and len(generated) < max_new_tokens:
                next_tok = int(v_out.logits[0, K_eff].argmax())
                generated.append(next_tok)
                n_acc += 1

            # Trim stale KV past last accepted position.
            valid_end = cur_pos + n_acc
            for i_ in range(len(past_kv.key_cache)):
                kc = past_kv.key_cache[i_]
                if kc is not None and kc.shape[2] > valid_end:
                    kc[:, :, valid_end:, :].zero_()
            for i_ in range(len(past_kv.value_cache)):
                vc = past_kv.value_cache[i_]
                if vc is not None and vc.shape[2] > valid_end:
                    vc[:, :, valid_end:, :].zero_()
            past_kv._seen_tokens = valid_end

            idx_h = min(n_acc - 1, v_out.hidden_states[-1].size(1) - 1)
            h_last = v_out.hidden_states[-1][0, idx_h:idx_h + 1, :]
            cur_pos = valid_end

        return torch.cat([input_ids, torch.tensor([generated], device=device)], dim=1)
