"""MERaLiON-2-3B + EAGLE + W4A16 wrapper for HF distribution.

Loads three pieces:
  1. The bf16 base model (speech_encoder + audio_adapter + tokenizer)
     pulled from `MERaLiON/MERaLiON-2-3B` on HF Hub.
  2. The W4A16 GPTQ Gemma2 text_decoder bundled in `text_decoder_w4a16/`,
     dispatched to ExllamaV1 / ExllamaV2 / Marlin via auto-gptq.
  3. The trained EAGLE draft (`eagle.safetensors` + `eagle_config.json`).

Provides `generate_eagle()` for K-step chain speculative decoding.
"""
from __future__ import annotations

import json
import os
import time

import torch
import torch.nn as nn

from .eagle_model import EAGLE


def _patch_autogptq_for_gemma2():
    """auto-gptq < 0.8 doesn't list gemma2; register it as alias of gemma."""
    from auto_gptq.modeling._const import SUPPORTED_MODELS
    if "gemma2" not in SUPPORTED_MODELS:
        SUPPORTED_MODELS.append("gemma2")
    from auto_gptq.modeling.auto import GPTQ_CAUSAL_LM_MODEL_MAP
    from auto_gptq.modeling.gemma import GemmaGPTQForCausalLM
    GPTQ_CAUSAL_LM_MODEL_MAP["gemma2"] = GemmaGPTQForCausalLM


class MERaLiON2EAGLEForASR(nn.Module):
    """Wrapper that bundles a quantized MERaLiON-2-3B with an EAGLE draft.

    Use `from_pretrained(repo)` to construct.  Use `generate_eagle(**inputs)`
    to run K-step chain speculative decoding.
    """

    def __init__(self, model, processor, eagle, rotary_emb, eagle_config):
        super().__init__()
        self.model         = model         # MERaLiON2ForConditionalGeneration
        self.processor     = processor
        self.eagle         = eagle
        self.rotary_emb    = rotary_emb
        self.eagle_config  = eagle_config

    @property
    def config(self):
        return self.model.config

    @property
    def text_decoder(self):
        return self.model.text_decoder

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        base_model: str = "MERaLiON/MERaLiON-2-3B",
        torch_dtype: torch.dtype = torch.float16,
        gptq_kernel: str = "exllama",
        device: str = "cuda",
        local_files_only: bool = False,
    ):
        """Args:
            pretrained_model_name_or_path:  HF repo id or local dir holding
                eagle.safetensors / text_decoder_w4a16/ / meralion2_bl/.
            base_model:        Source for the BF16 speech encoder + audio
                               adapter (default: the original MERaLiON-2-3B).
            torch_dtype:       Must be torch.float16 — required by all auto-
                               gptq W4A16 kernels.
            gptq_kernel:       'exllama' (default, fastest at batch=1) |
                               'exllamav2' | 'marlin' (best at batch>=4).
            device:            Target device ('cuda' / 'cuda:0' / etc.)
        """
        if torch_dtype != torch.float16:
            raise ValueError(
                "auto-gptq W4A16 kernels require torch_dtype=torch.float16")

        # Resolve to local dir (download if needed)
        if os.path.isdir(pretrained_model_name_or_path):
            local_dir = pretrained_model_name_or_path
        else:
            from huggingface_hub import snapshot_download
            local_dir = snapshot_download(
                repo_id=pretrained_model_name_or_path,
                local_files_only=local_files_only,
            )

        td_dir   = os.path.join(local_dir, "text_decoder_w4a16")
        eag_path = os.path.join(local_dir, "eagle.safetensors")
        eag_cfg  = os.path.join(local_dir, "eagle_config.json")
        for p in (td_dir, eag_path, eag_cfg):
            if not os.path.exists(p):
                raise FileNotFoundError(f"missing: {p}")

        # 1. Load the W4A16 Gemma2 text decoder via auto-gptq.
        _patch_autogptq_for_gemma2()
        from auto_gptq import AutoGPTQForCausalLM

        if gptq_kernel == "marlin":
            kw = dict(use_marlin=True)
        elif gptq_kernel == "exllama":
            kw = dict(disable_exllama=False, disable_exllamav2=True)
        elif gptq_kernel == "exllamav2":
            kw = dict(disable_exllama=True, disable_exllamav2=False)
        else:
            raise ValueError(f"unknown gptq_kernel: {gptq_kernel}")

        print(f"Loading W4A16 text_decoder (kernel={gptq_kernel}) …")
        t0 = time.time()
        qmodel = AutoGPTQForCausalLM.from_quantized(
            td_dir, torch_dtype=torch_dtype, trust_remote_code=False, **kw)
        qmodel = qmodel.to(device)
        print(f"  done in {time.time()-t0:.1f}s")

        # 2. Load BF16 base model from HF for speech_encoder + adapter.
        print(f"Loading BF16 base model: {base_model}")
        from .meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            base_model, trust_remote_code=True,
            local_files_only=local_files_only)
        full_model = MERaLiON2ForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=torch_dtype, use_safetensors=True,
            local_files_only=local_files_only)
        full_model.text_decoder = qmodel.model
        full_model = full_model.to(device)

        # 3. Build EAGLE and load its weights.
        from safetensors.torch import load_file
        with open(eag_cfg) as f:
            eag_meta = json.load(f)
        n_layers = eag_meta.get("num_layers", 1)

        td       = full_model.text_decoder
        verif_cfg = td.model.config
        layer_cls = type(td.model.layers[0])
        # The marlin/exllama text_decoder's Gemma2DecoderLayer class won't
        # have plain Linear projs — but we need a vanilla Gemma2DecoderLayer
        # for EAGLE.  Import from the vendored modeling code.
        from .meralion2_bl.modeling_gemma2 import Gemma2DecoderLayer
        eagle = EAGLE(verif_cfg, td.base_model.embed_tokens, td.lm_head,
                      Gemma2DecoderLayer, num_layers=n_layers)
        eagle = eagle.to(device).to(torch_dtype)
        # Re-bind shared refs after .to() (module.to() clones non-parameter refs)
        object.__setattr__(eagle, "_embed",   td.base_model.embed_tokens)
        object.__setattr__(eagle, "_lm_head", td.lm_head)

        sd = load_file(eag_path)
        missing, unexpected = eagle.load_state_dict(sd, strict=False)
        missing = [m for m in missing
                   if not (m.startswith("_embed") or m.startswith("_lm_head"))]
        if missing:
            print(f"  EAGLE: missing keys (truncated): {missing[:3]}")
        if unexpected:
            print(f"  EAGLE: unexpected keys (truncated): {unexpected[:3]}")
        eagle.eval()
        for p in eagle.parameters():
            p.requires_grad_(False)

        rotary_emb = td.model.rotary_emb
        print(f"  EAGLE: {sum(p.numel() for p in eagle.parameters())/1e6:.1f} M params")

        return cls(full_model, processor, eagle, rotary_emb, eag_meta)

    @torch.inference_mode()
    def generate_eagle(
        self,
        input_ids,
        attention_mask,
        input_features,
        feature_attention_mask,
        max_new_tokens: int = 128,
        K: int = 4,
    ):
        """Greedy K-step chain speculative decoding.  Returns the full
        token-id sequence (prompt + generated).
        """
        from transformers.cache_utils import HybridCache, DynamicCache

        device = input_ids.device
        tok    = self.processor.tokenizer
        td_dtype = next(p.dtype for p in self.text_decoder.parameters()
                        if p.dtype in (torch.float16, torch.bfloat16))
        seq_len   = input_ids.shape[1]
        max_cache = seq_len + max_new_tokens
        verifier_kv = HybridCache(
            self.text_decoder.model.config, max_batch_size=1,
            max_cache_len=max_cache, dtype=td_dtype, device=device)

        eos_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")}
        eos_ids.discard(None)

        # Prefill (full multimodal forward)
        out = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            past_key_values=verifier_kv, use_cache=True,
            cache_position=torch.arange(0, seq_len, device=device),
            output_hidden_states=True, return_dict=True,
        )
        h_last   = out.hidden_states[-1][0, -1:, :]
        next_tok = int(out.logits[0, -1].argmax())
        generated = list(input_ids[0].tolist()) + [next_tok]
        cur_pos   = seq_len

        while len(generated) - seq_len < max_new_tokens and next_tok not in eos_ids:
            # Draft: K sequential EAGLE steps, fresh KV cache each round
            draft_kv = DynamicCache()
            last_tok = next_tok
            last_h   = h_last
            draft_tokens = []
            K_eff = min(K, max_cache - cur_pos - 1)
            for k in range(K_eff):
                ids_d  = torch.tensor([[last_tok]], dtype=torch.long, device=device)
                prev_h = last_h.unsqueeze(0)
                pos_id = torch.tensor([[k]], dtype=torch.long, device=device)
                pos_eb = self.rotary_emb(prev_h, pos_id)
                cpos   = torch.tensor([k], device=device)
                logits_d, h_new, _ = self.eagle(
                    input_ids=ids_d, prev_hidden=prev_h,
                    position_ids=pos_id, attention_mask=None,
                    cache_position=cpos, past_key_value=draft_kv,
                    position_embeddings=pos_eb)
                d_tok = int(logits_d[0, -1].argmax())
                draft_tokens.append(d_tok)
                last_tok = d_tok
                last_h   = h_new[0]
                if d_tok in eos_ids:
                    break
            K_act = len(draft_tokens)
            if K_act == 0:
                break

            # Verify K_act+1 tokens in one verifier forward pass
            spec_ids  = torch.tensor([[next_tok] + draft_tokens],
                                     dtype=torch.long, device=device)
            spec_attn = torch.ones(1, cur_pos + K_act + 1, dtype=torch.long, device=device)
            spec_cpos = torch.arange(cur_pos, cur_pos + K_act + 1, device=device)
            v_out = self.text_decoder(
                input_ids=spec_ids, attention_mask=spec_attn,
                past_key_values=verifier_kv, use_cache=True,
                cache_position=spec_cpos,
                output_hidden_states=True, return_dict=True,
            )

            # Greedy chain accept
            n_acc, stopped = 0, False
            for i in range(K_act):
                if len(generated) - seq_len >= max_new_tokens:
                    stopped = True; break
                pred = int(v_out.logits[0, i].argmax())
                if pred == draft_tokens[i]:
                    generated.append(draft_tokens[i]); n_acc += 1
                    next_tok = draft_tokens[i]
                    if draft_tokens[i] in eos_ids:
                        stopped = True; break
                else:
                    generated.append(pred); n_acc += 1
                    next_tok = pred; break
            if n_acc == K_act and not stopped \
                    and len(generated) - seq_len < max_new_tokens:
                bonus = int(v_out.logits[0, K_act].argmax())
                generated.append(bonus); n_acc += 1
                next_tok = bonus

            # Trim verifier KV to accepted length, update h_last
            valid_end = cur_pos + n_acc
            for i in range(len(verifier_kv.key_cache)):
                kc = verifier_kv.key_cache[i]
                vc = verifier_kv.value_cache[i]
                if kc is not None and kc.shape[2] > valid_end:
                    kc[:, :, valid_end:, :].zero_()
                if vc is not None and vc.shape[2] > valid_end:
                    vc[:, :, valid_end:, :].zero_()
            verifier_kv._seen_tokens = valid_end
            idx_h  = min(n_acc - 1, v_out.hidden_states[-1].size(1) - 1)
            h_last = v_out.hidden_states[-1][0, idx_h:idx_h + 1, :]
            cur_pos = valid_end

        return torch.tensor([generated], dtype=torch.long, device=device)
