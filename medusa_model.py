"""Medusa heads for MERaLiON-2-3B speculative decoding.

Attaches K small MLP heads on top of the verifier's last-layer hidden state h.
Head i predicts the token at offset +i (1-indexed). At inference:
    verifier.forward(...) -> h            # one full verifier forward
    draft_i = argmax(lm_head(head_i(h)))  # K cheap MLP forwards
    verifier.forward([next_tok, draft_1..draft_K]) -> K+1 logits  # verify in one pass

Paper: https://arxiv.org/abs/2401.10774
Reference impl: https://github.com/FasterDecoding/Medusa

Design choices (v1):
  * Share verifier's lm_head (avoid K * hidden_size * vocab_size extra params).
  * Single ResBlock per head (num_layers=1).
  * Frozen verifier during training; only heads trainable.
  * Greedy linear acceptance at inference (no tree attention).
"""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """y = x + SiLU(W x).  Init W to 0 so head starts as identity."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act    = nn.SiLU()
        # Zero init makes head initially predict the same token as verifier's
        # next-token logits (via shared lm_head on unchanged hidden state) —
        # a sensible starting point for training.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return x + self.act(self.linear(x))


class MedusaHead(nn.Module):
    """Single Medusa head: residual MLP that refines hidden state.  The shared
    verifier lm_head is applied *externally* to the output to produce logits."""
    def __init__(self, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.mlp = nn.Sequential(
            *[ResBlock(hidden_size) for _ in range(num_layers)]
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)


class MedusaHeads(nn.Module):
    """Bundle of K heads plus a reference to the verifier's lm_head.

    Usage (training):
        heads = MedusaHeads(K=4, hidden_size=H, lm_head=verifier.text_decoder.lm_head)
        # Freeze verifier; only heads.mlps.parameters() are trainable.
        logits_k = heads(h)   # (K, B, T, vocab_size)

    Usage (inference):
        # h is (B, 1, H) last hidden state of verifier at current position
        draft_logits = heads(h)   # (K, 1, 1, V)
        draft_toks   = draft_logits.argmax(-1).squeeze(-1)  # (K, 1)
    """
    def __init__(self, num_heads: int, hidden_size: int,
                 lm_head: nn.Linear, num_layers: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.heads = nn.ModuleList([
            MedusaHead(hidden_size, num_layers=num_layers)
            for _ in range(num_heads)
        ])
        # lm_head is SHARED with the verifier — keep as an attribute, not a
        # submodule, so it's not re-registered when we save/load just the heads.
        self._lm_head = lm_head

    @property
    def lm_head(self):
        return self._lm_head

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, H) → logits (K, B, T, V).  lm_head applied via @ weight
        so external dtypes don't leak into heads' autograd graph."""
        lm_weight = self._lm_head.weight  # (V, H), dtype inherited
        outs = []
        for head in self.heads:
            refined = head(h)                # (B, T, H)
            logits  = refined @ lm_weight.t()  # (B, T, V)
            outs.append(logits)
        return torch.stack(outs, dim=0)       # (K, B, T, V)

    # Convenience: save/load only the trainable head params (not lm_head).
    def trainable_state_dict(self):
        return {f"head_{i}": h.state_dict() for i, h in enumerate(self.heads)}

    def load_trainable_state_dict(self, sd):
        for i, h in enumerate(self.heads):
            h.load_state_dict(sd[f"head_{i}"])


def attach_medusa(model, num_heads: int = 4, num_layers: int = 1) -> MedusaHeads:
    """Instantiate heads sized for the given MERaLiON2 model, place them on the
    lm_head's device, return the MedusaHeads bundle."""
    td = model.text_decoder
    hidden_size = td.lm_head.in_features
    heads = MedusaHeads(
        num_heads=num_heads,
        hidden_size=hidden_size,
        lm_head=td.lm_head,
        num_layers=num_layers,
    )
    heads = heads.to(td.lm_head.weight.device).to(td.lm_head.weight.dtype)
    # Re-bind shared lm_head after .to(): module.to() clones the underlying
    # linear, we want to keep the exact same parameter object.
    heads._lm_head = td.lm_head
    return heads


if __name__ == "__main__":
    # Smoke test
    lm = nn.Linear(2304, 256000, bias=False)
    heads = MedusaHeads(num_heads=4, hidden_size=2304, lm_head=lm)
    h = torch.randn(1, 10, 2304)
    out = heads(h)
    print("output shape:", out.shape)  # expect (4, 1, 10, 256000)
    n_params = sum(p.numel() for p in heads.heads.parameters())
    print(f"trainable params (heads only): {n_params/1e6:.2f} M")
