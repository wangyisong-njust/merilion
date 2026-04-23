from dataclasses import dataclass
from typing import List, Tuple

import torch
from transformers.cache_utils import Cache

def _round_heads(x: float, *, min_val: int = 1, multiple_of: int = 1) -> int:
    # round to nearest int, clamp, and optionally enforce divisibility
    v = int(round(x))
    v = max(min_val, v)
    if multiple_of > 1:
        v = max(multiple_of, (v // multiple_of) * multiple_of)
    return v

def _round_size(x: float, *, min_val: int = 1, multiple_of: int = 1) -> int:
    v = int(round(x))
    v = max(min_val, v)
    if multiple_of > 1:
        v = max(multiple_of, (v // multiple_of) * multiple_of)
    return v

@dataclass
class LayerwiseDims:
    num_attention_heads: List[int]
    num_key_value_heads: List[int]
    intermediate_size: List[int]

def build_layerwise_dims(
    *,
    num_layers: int,
    orig_num_attention_heads: int,
    orig_num_key_value_heads: int,
    orig_intermediate_size: int,
    midblock_start: int,
    midblock_end: int,
    ratio_attn: float,
    ratio_mlp: float,
    heads_multiple_of: int = 1,   # set to 8 if you need TensorCore-friendly multiples
    mlp_multiple_of: int = 1,     # e.g., 8 or 16 depending on your pruning constraints
) -> LayerwiseDims:
    attn_heads = []
    kv_heads = []
    mlp_sizes = []

    for i in range(num_layers):
        in_mid = (midblock_start <= i < midblock_end)

        if in_mid:
            attn_heads.append(_round_heads(orig_num_attention_heads * ratio_attn,
                                           multiple_of=heads_multiple_of))
            kv_heads.append(_round_heads(orig_num_key_value_heads * ratio_attn,
                                         multiple_of=heads_multiple_of))
            mlp_sizes.append(_round_size(orig_intermediate_size * ratio_mlp,
                                         multiple_of=mlp_multiple_of))
        else:
            attn_heads.append(orig_num_attention_heads)
            kv_heads.append(orig_num_key_value_heads)
            mlp_sizes.append(orig_intermediate_size)

    return LayerwiseDims(attn_heads, kv_heads, mlp_sizes)





class LayerwiseKVCache(Cache):
    """
    Layer-wise KV cache with per-layer num_kv_heads.
    Stores K/V as [B, H_kv(layer), T, D].
    """
    def __init__(
        self,
        *,
        num_kv_heads_per_layer,
        head_dim: int,
        max_batch_size: int,
        max_cache_len: int,
        device,
        dtype,
    ):
        self.num_layers = len(num_kv_heads_per_layer)
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
        self.device = device
        self.dtype = dtype
        self.head_dim = head_dim

        self.k = []
        self.v = []
        for h_kv in num_kv_heads_per_layer:
            self.k.append(torch.zeros(
                (max_batch_size, h_kv, max_cache_len, head_dim),
                device=device, dtype=dtype
            ))
            self.v.append(torch.zeros(
                (max_batch_size, h_kv, max_cache_len, head_dim),
                device=device, dtype=dtype
            ))

        # single global filled length (Gemma2 uses same token positions across layers)
        self._seq_len = 0

    def get_seq_length(self, layer_idx: int = 0):
        return self._seq_len

    def get_max_cache_shape(self):
        # Gemma2Model._update_causal_mask uses this as target_length
        return self.max_cache_len

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """
        key_states/value_states: [B, H_kv, T_new, D]
        """
        b, h, t_new, d = key_states.shape
        if d != self.head_dim:
            raise ValueError(f"head_dim mismatch: got {d}, expected {self.head_dim}")
        if h != self.k[layer_idx].shape[1]:
            raise ValueError(
                f"KV heads mismatch at layer {layer_idx}: got {h}, "
                f"cache expects {self.k[layer_idx].shape[1]}"
            )

        cache_position = None
        if cache_kwargs is not None:
            cache_position = cache_kwargs.get("cache_position", None)

        if cache_position is None:
            # append mode
            start = self._seq_len
            end = start + t_new
            if end > self.max_cache_len:
                raise ValueError(f"Cache overflow: end={end} > max_cache_len={self.max_cache_len}")
            self.k[layer_idx][:b, :h, start:end, :] = key_states
            self.v[layer_idx][:b, :h, start:end, :] = value_states
            self._seq_len = end
        else:
            # cache_position: shape [T_new]
            # correctness-first implementation
            pos_list = cache_position.tolist()
            for i, pos in enumerate(pos_list):
                self.k[layer_idx][:b, :h, pos:pos+1, :] = key_states[:, :, i:i+1, :]
                self.v[layer_idx][:b, :h, pos:pos+1, :] = value_states[:, :, i:i+1, :]
            self._seq_len = max(self._seq_len, int(pos_list[-1]) + 1)

        return (
            self.k[layer_idx][:b, :h, :self._seq_len, :],
            self.v[layer_idx][:b, :h, :self._seq_len, :],
        )



def make_midblock_cache_for_gemma2(
    *,
    model,                 # Gemma2Model / Gemma2ForCausalLM .model etc.
    batch_size: int,
    max_cache_len: int,
    midblock_start: int,
    midblock_end: int,
    ratio_attn: float,
    ratio_mlp: float,
    heads_multiple_of: int = 1,
    mlp_multiple_of: int = 1,
):
    cfg = model.config
    num_layers = cfg.num_hidden_layers

    # originals (global) from config
    orig_h = cfg.num_attention_heads
    orig_kv = cfg.num_key_value_heads
    orig_mlp = cfg.intermediate_size

    dims = build_layerwise_dims(
        num_layers=num_layers,
        orig_num_attention_heads=orig_h,
        orig_num_key_value_heads=orig_kv,
        orig_intermediate_size=orig_mlp,
        midblock_start=midblock_start,
        midblock_end=midblock_end,
        ratio_attn=ratio_attn,
        ratio_mlp=ratio_mlp,
        heads_multiple_of=heads_multiple_of,
        mlp_multiple_of=mlp_multiple_of,
    )

    # head_dim is normally fixed (hidden_size / num_attention_heads original)
    # In Gemma/Gemma2 this is typically cfg.hidden_size // cfg.num_attention_heads
    head_dim = cfg.hidden_size // orig_h

    cache = LayerwiseKVCache(
        num_kv_heads_per_layer=dims.num_key_value_heads,
        head_dim=head_dim,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    )

    return cache, dims
