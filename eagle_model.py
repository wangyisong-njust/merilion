"""EAGLE draft model for MERaLiON-2-3B (ASR, audio-conditioned).

EAGLE (https://arxiv.org/abs/2401.15077) replaces Medusa's parallel per-offset
heads with a SINGLE small auto-regressive draft:

    Draft step t:
        input  = concat( embed(token_t), h_{t-1} )   # previous token's embed
                                                      # + verifier's last-layer
                                                      # hidden state at t-1
        fused  = Linear_fuse(input)                   # → (hidden,)
        h_t_draft = DecoderLayer(fused, <cache_from_previous_draft_steps>)
        logits = lm_head(h_t_draft)                   # verifier's lm_head (shared)

At inference, the draft model generates K tokens sequentially, each
conditioned on the previous draft step's h.  It reuses the verifier's
`embed_tokens` and `lm_head` (frozen), so only the fusion layer + the single
decoder layer are trainable (~90 M params for Gemma2-2B config).

Training (in train_eagle.py): supervise with next-token CE from the
(hidden, token) pairs captured by collect_medusa_data.py.  Unlike Medusa,
EAGLE's recurrence means acc rate doesn't degrade nearly as fast with
depth — paper reports 2.5-3× speedup.
"""
import torch
import torch.nn as nn


class EAGLE(nn.Module):
    """Minimal EAGLE draft.

    Reuses the verifier's Gemma2DecoderLayer class (which understands RoPE,
    softcap, sliding window, GQA, etc.) — we only add a fusion linear and
    an output layernorm, and borrow the verifier's embed_tokens / lm_head.

    Keeps `_embed`/`_lm_head` as non-module attributes so `state_dict()`
    stores only the trainable part (fuse + layer + final_ln).
    """
    def __init__(self, verifier_config, verifier_embed_tokens,
                 verifier_lm_head, layer_cls, num_layers=1):
        super().__init__()
        self.config = verifier_config
        self.num_layers = num_layers
        H = verifier_config.hidden_size

        # fuse(cat(emb, h)) → H
        self.fuse = nn.Linear(2 * H, H, bias=False)
        nn.init.zeros_(self.fuse.weight)                       # init as "pass-through h"
        with torch.no_grad():
            # Start by outputting h (copy h-slice of the fused input, ignoring emb).
            self.fuse.weight[:, H:] = torch.eye(H, dtype=self.fuse.weight.dtype,
                                                device=self.fuse.weight.device)

        # Decoder layers — use same class as the verifier.
        # layer_idx=1,3,5,... → is_sliding=False (Gemma2 uses sliding window on
        # even-indexed layers); we want full global attention in the draft.
        self.layers = nn.ModuleList([
            layer_cls(verifier_config, layer_idx=2 * i + 1)
            for i in range(num_layers)
        ])
        # Gemma2 normalises outputs at the end of its decoder stack.
        self.final_ln = nn.RMSNorm(H, eps=verifier_config.rms_norm_eps)

        # Non-parameter references (don't register as submodules so they
        # don't land in state_dict()).
        object.__setattr__(self, "_embed",   verifier_embed_tokens)
        object.__setattr__(self, "_lm_head", verifier_lm_head)

    # ── helpers shared with verifier ──────────────────────────────────────────
    def _apply_rotary_and_project(self, fused, position_ids, cache_position,
                                  past_key_value, attention_mask):
        """Wrap our decoder_layer forward to match the verifier's interface
        (need position_embeddings)."""
        # Compute rotary position embeddings using the layer's rotary_emb
        # object — Gemma2 exposes it on the outer Gemma2Model.  Since our
        # draft is standalone, replicate it on construction.
        raise NotImplementedError("use forward(); see inference wiring")

    def forward(self, input_ids, prev_hidden,
                position_ids=None, attention_mask=None,
                cache_position=None, past_key_value=None,
                position_embeddings=None):
        """
        input_ids:    (B, T)           — the previous token at each position
        prev_hidden:  (B, T, H)        — h produced at that position by verifier
                                          (for step t, this is h_{t-1} from the
                                          verifier's last-layer output)
        position_embeddings: (cos, sin) tuple for RoPE at each position.  We
                                         require the caller to pre-compute it
                                         from the verifier's rotary_emb so
                                         EAGLE stays lightweight.
        past_key_value:                cache accumulated from prior draft steps.
        attention_mask:                optional 4D mask (used by training
                                        code that batches a whole sequence).
        Returns (logits, next_hidden, past_key_value).
        """
        emb  = self._embed(input_ids)                      # (B, T, H)
        h    = self.fuse(torch.cat([emb, prev_hidden], dim=-1))  # (B, T, H)

        for layer in self.layers:
            layer_out = layer(
                hidden_states=h,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=False,
                use_cache=(past_key_value is not None),
                cache_position=cache_position,
            )
            h = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        h_next = self.final_ln(h)
        logits = h_next @ self._lm_head.weight.t()         # shared lm_head
        return logits, h_next, past_key_value

    # ── save/load only trainable weights ──────────────────────────────────────
    def trainable_state_dict(self):
        out = {}
        for n, p in self.named_parameters():
            out[n] = p.detach().cpu()
        return out

    def load_trainable_state_dict(self, sd):
        missing, unexpected = self.load_state_dict(sd, strict=False)
        # Ignore any missing lm_head / embed keys — those live on the verifier.
        missing = [m for m in missing
                   if not m.startswith("_embed") and not m.startswith("_lm_head")]
        if missing:
            print(f"  EAGLE.load: missing keys: {missing[:5]}")
        if unexpected:
            print(f"  EAGLE.load: unexpected keys: {unexpected[:5]}")


def attach_eagle(verifier_model, device, dtype=torch.bfloat16, num_layers=1):
    """Build an EAGLE draft for the given MERaLiON-2 verifier and place it
    on `device`.  Returns (eagle, rotary_emb) — the rotary module is the
    verifier's own, reused to compute position_embeddings without
    duplicating params."""
    td = verifier_model.text_decoder
    cfg = td.model.config
    layer_cls = type(td.model.layers[0])          # Gemma2DecoderLayer
    eagle = EAGLE(cfg, td.base_model.embed_tokens, td.lm_head, layer_cls,
                  num_layers=num_layers)
    eagle = eagle.to(device).to(dtype)
    # Re-bind shared references post-.to() (module.to() clones them).
    object.__setattr__(eagle, "_embed",   td.base_model.embed_tokens)
    object.__setattr__(eagle, "_lm_head", td.lm_head)
    rotary = td.model.rotary_emb
    return eagle, rotary


if __name__ == "__main__":
    # Smoke test construction on tiny config.
    from types import SimpleNamespace
    cfg = SimpleNamespace(hidden_size=32, num_attention_heads=2,
                          num_key_value_heads=1, head_dim=16,
                          intermediate_size=64, rms_norm_eps=1e-6,
                          midblock_start=-1, midblock_end=-1,
                          midblock_ratio=1.0, attention_dropout=0.0,
                          attn_logit_softcapping=None,
                          query_pre_attn_scalar=None,
                          sliding_window=None, max_position_embeddings=2048,
                          _attn_implementation="eager")
    print(f"(toy construction sanity test — full e2e test needs real verifier)")
    import torch.nn as nn
    # Fake layer + lm_head
    emb = nn.Embedding(100, 32)
    lm_head = nn.Linear(32, 100, bias=False)
    class FakeLayer(nn.Module):
        def __init__(self, c, layer_idx=0):
            super().__init__()
            self.identity = nn.Linear(c.hidden_size, c.hidden_size)
        def forward(self, hidden_states, **kw):
            return (self.identity(hidden_states),)
    e = EAGLE(cfg, emb, lm_head, FakeLayer)
    x = torch.randint(0, 100, (1, 3))
    h = torch.randn(1, 3, 32)
    out = e(x, h)
    print(f"  logits shape: {out[0].shape}  h_next shape: {out[1].shape}")
