import torch
from transformers.cache_utils import Cache

class LayerwiseKVCache(Cache):
    """
    Layer-wise KV cache with per-layer num_kv_heads support.
    Stores K/V as [B, H_kv(layer), T, D].
    """
    def __init__(self, num_layers, num_kv_heads_per_layer, head_dim,
                 max_batch_size, max_cache_len, device, dtype):
        self.num_layers = num_layers
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

        # track how many tokens are currently filled (global, same across layers)
        self._seq_len = 0

    def get_seq_length(self, layer_idx: int = 0):
        return self._seq_len

    def get_max_cache_shape(self):
        # Gemma2Model._update_causal_mask uses this as target_length for 4D mask
        return self.max_cache_len

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """
        key_states/value_states arrive as [B, H_kv, T_new, D] in Gemma2Attention (after transpose).
        """
        b, h, t_new, d = key_states.shape
        assert d == self.head_dim, (d, self.head_dim)

        # write position: either use cache_position passed via cache_kwargs (static-style),
        # or append at current seq_len (dynamic-append).
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
            # cache_position is [T_new] positions (Gemma2Model builds it as arange)
            # write each token to the correct absolute slot
            # NOTE: This simple loop is fine for correctness; you can vectorize later.
            for i, pos in enumerate(cache_position.tolist()):
                self.k[layer_idx][:b, :h, pos:pos+1, :] = key_states[:, :, i:i+1, :]
                self.v[layer_idx][:b, :h, pos:pos+1, :] = value_states[:, :, i:i+1, :]
            self._seq_len = max(self._seq_len, int(cache_position[-1].item()) + 1)

        # return the "full" kv (up to seq_len) in the same shape expected by attention backend
        return (
            self.k[layer_idx][:b, :h, :self._seq_len, :],
            self.v[layer_idx][:b, :h, :self._seq_len, :],
        )
