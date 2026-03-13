"""Custom vLLM Gemma2 model with per-layer dimension support for pruned MERaLiON-2 models.

Problem:
  Standard vLLM Gemma2Model creates all layers with uniform dimensions from config.
  Pruned models (midblock system) have different attention heads and MLP sizes per layer:
    - Layers outside [midblock_start, midblock_end): original full dimensions
    - Layers inside [midblock_start, midblock_end): pruned dimensions

Solution:
  PrunedGemma2Model reads midblock_ratio/midblock_start/midblock_end from config and
  creates each decoder layer with the correct per-layer dimensions.

KV Cache Compatibility:
  - config.num_key_value_heads stays at the MAXIMUM value (unpruned layers)
  - vLLM allocates KV cache blocks with this maximum size
  - Pruned layers write/read fewer KV heads; extra cache slots are unused
  - Wastes some memory but is functionally correct with no CacheEngine changes

Compatible with vLLM v0.6.5 - v0.7.3 (V0 engine API).
"""

import copy
from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

import inspect
from vllm.attention import Attention
try:
    from vllm.attention import AttentionMetadata
except ImportError:
    AttentionMetadata = None  # vLLM 0.8.5+ removed from public API

# vLLM 0.8.5+ removed kv_cache/attn_metadata from Attention.forward();
# they are now retrieved from the execution context internally.
_ATTN_FWD_TAKES_KV = 'kv_cache' in inspect.signature(Attention.forward).parameters
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors


def get_layer_dims(text_config, layer_idx: int):
    """Compute per-layer attention/MLP dimensions from midblock config.

    Returns:
        (num_attention_heads, num_key_value_heads, intermediate_size)
    """
    midblock_start = getattr(text_config, "midblock_start", -1)
    midblock_end = getattr(text_config, "midblock_end", -1)
    midblock_ratio = getattr(text_config, "midblock_ratio", 1.0)
    mlp_midblock_ratio = getattr(text_config, "text_mlp_midblock_ratio", midblock_ratio)

    num_attention_heads = text_config.num_attention_heads
    num_key_value_heads = text_config.num_key_value_heads
    intermediate_size = text_config.intermediate_size

    if midblock_start >= 0 and midblock_start <= layer_idx < midblock_end:
        num_attention_heads = int(num_attention_heads * midblock_ratio)
        num_key_value_heads = int(num_key_value_heads * midblock_ratio)
        intermediate_size = int(intermediate_size * mlp_midblock_ratio)

    return num_attention_heads, num_key_value_heads, intermediate_size


def is_pruned_model(text_config) -> bool:
    """Check if the config indicates a pruned model with non-uniform layers."""
    midblock_start = getattr(text_config, "midblock_start", -1)
    midblock_end = getattr(text_config, "midblock_end", -1)
    midblock_ratio = getattr(text_config, "midblock_ratio", 1.0)
    return midblock_start >= 0 and midblock_end > midblock_start and midblock_ratio < 1.0


class PrunedGemma2MLP(nn.Module):
    """Gemma2 MLP layer with configurable intermediate_size."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = GeluAndMul(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class PrunedGemma2Attention(nn.Module):
    """Gemma2 Attention with configurable per-layer head counts."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        layer_idx: int,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.total_num_heads = num_attention_heads
        self.total_num_kv_heads = num_key_value_heads

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = config.query_pre_attn_scalar ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Gemma2: even layers use sliding window, odd layers use global attention
        is_sliding = not bool(layer_idx % 2)
        _sw = getattr(config, 'sliding_window', None) or getattr(config, 'sliding_window_size', None)
        sliding_window = _sw if is_sliding else None

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            is_neox_style=True,
        )

        attn_kwargs = dict(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        import inspect
        _attn_sig = inspect.signature(Attention.__init__).parameters
        if 'per_layer_sliding_window' in _attn_sig:
            attn_kwargs['per_layer_sliding_window'] = sliding_window
        if 'logits_soft_cap' in _attn_sig:
            attn_kwargs['logits_soft_cap'] = config.attn_logit_softcapping
        self.attn = Attention(**attn_kwargs)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache=None,
        attn_metadata=None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        if _ATTN_FWD_TAKES_KV:
            attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        else:
            attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class PrunedGemma2DecoderLayer(nn.Module):
    """Gemma2 decoder layer with per-layer attention/MLP dimensions."""

    def __init__(
        self,
        config,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        num_attn_heads, num_kv_heads, intermediate_size = get_layer_dims(config, layer_idx)

        self.self_attn = PrunedGemma2Attention(
            hidden_size=config.hidden_size,
            num_attention_heads=num_attn_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=config.head_dim,
            layer_idx=layer_idx,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = PrunedGemma2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache=None,
        attn_metadata=None,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-attention norm (fused add+norm when residual is not None)
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self attention
        hidden_states = self.self_attn(positions, hidden_states, kv_cache, attn_metadata)
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Pre-MLP norm (fused add+norm)
        hidden_states, residual = self.pre_feedforward_layernorm(hidden_states, residual)

        # MLP
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)

        return hidden_states, residual


class PrunedGemma2Model(nn.Module):
    """Gemma2 model with per-layer dimension support for pruned models.

    Drop-in replacement for vLLM's Gemma2Model that handles non-uniform
    layer dimensions from midblock pruning.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        # Build layers with per-layer dimensions
        self.layers = nn.ModuleList([
            PrunedGemma2DecoderLayer(
                config=config,
                layer_idx=i,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{i}",
            )
            for i in range(config.num_hidden_layers)
        ])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Log per-layer dimensions for debugging
        midblock_start = getattr(config, "midblock_start", -1)
        midblock_end = getattr(config, "midblock_end", -1)
        if midblock_start >= 0:
            full_heads, full_kv, full_mlp = get_layer_dims(config, 0)
            mid_heads, mid_kv, mid_mlp = get_layer_dims(config, midblock_start)
            print(f"[PrunedGemma2Model] Layers 0-{midblock_start-1}, {midblock_end}-{config.num_hidden_layers-1}: "
                  f"heads={full_heads}, kv={full_kv}, mlp={full_mlp}")
            print(f"[PrunedGemma2Model] Layers {midblock_start}-{midblock_end-1}: "
                  f"heads={mid_heads}, kv={mid_kv}, mlp={mid_mlp}")

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata=None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        elif inputs_embeds is not None:
            hidden_states = inputs_embeds
            residual = None
        else:
            hidden_states = self.get_input_embeddings(input_ids)
            residual = None

        # Gemma2: normalize embeddings by sqrt(hidden_size)
        normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for i, layer in enumerate(self.layers):
            kv = kv_caches[i] if kv_caches is not None else None
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv,
                attn_metadata,
                residual,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states": torch.zeros(
                (batch_size, self.config.hidden_size), dtype=dtype, device=device
            ),
            "residual": torch.zeros(
                (batch_size, self.config.hidden_size), dtype=dtype, device=device
            ),
        })
