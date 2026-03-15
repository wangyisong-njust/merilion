"""vLLM-compatible MERaLiON2 model with support for pruned (non-uniform) layer dimensions.

This is a drop-in replacement for the standard vllm_plugin_meralion2 model class.
It detects midblock pruning config and uses PrunedGemma2Model instead of vLLM's
built-in Gemma2Model when non-uniform layers are detected.

Usage:
    # Register this model class instead of the standard one:
    from vllm import ModelRegistry
    from meralion2_vllm_pruned import MERaLiON2PrunedForConditionalGeneration
    ModelRegistry.register_model(
        "MERaLiON2ForConditionalGeneration",
        MERaLiON2PrunedForConditionalGeneration
    )
"""
from functools import lru_cache
from typing import Iterable, List, Mapping, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torch.nn as nn

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs, DummyData,
                         InputContext, token_inputs)
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
try:
    from vllm.model_executor.model_loader.weight_utils import maybe_remap_kv_scale_name
except ImportError:
    def maybe_remap_kv_scale_name(name, params_dict):
        return name  # no-op on older vLLM versions
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
try:
    from vllm.multimodal.utils import consecutive_placeholder_ranges
except ImportError:
    # Fallback for older vLLM versions
    def consecutive_placeholder_ranges(num_items, item_size):
        return [range(i * item_size, (i + 1) * item_size) for i in range(num_items)]
from vllm.sequence import IntermediateTensors, SequenceData
from transformers.models.whisper.modeling_whisper import WhisperEncoder

try:
    from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsLoRA, SupportsPP
except ImportError:
    # Older vLLM: define stub mixins so the class definition doesn't fail
    class SupportsMultiModal:
        pass
    class SupportsLoRA:
        pass
    class SupportsPP:
        pass
from vllm.model_executor.models.utils import maybe_prefix

from pruned_gemma2_vllm import PrunedGemma2Model, is_pruned_model

# Re-use modules from existing plugin
try:
    from vllm_plugin_meralion2.transformers_utils.modules import (
        autoset_attn_implementation_for_whisper,
        MERaLiON2Inputs, MERaLiON2SpeechAudioAdaper)
except ImportError:
    # Inline fallback if plugin is not installed
    from transformers import WhisperConfig

    def autoset_attn_implementation_for_whisper(config):
        config._attn_implementation = "sdpa"
        return config

    class MERaLiON2Inputs(dict):
        pass

    class MERaLiON2SpeechAudioAdaper(nn.Module):
        def __init__(self, speech_dim, text_dim):
            super().__init__()
            scale = 15
            self.scale = scale
            self.mlp_adapter = nn.Sequential(
                nn.Linear(speech_dim * scale, speech_dim * 5),
                nn.SiLU(),
                nn.Dropout(0.01),
            )
            self.gate_proj = nn.Linear(speech_dim * 5, speech_dim * 5)
            self.pool_proj = nn.Linear(speech_dim * 5, speech_dim * 5)
            self.act_fn = nn.SiLU()
            self.out_proj = nn.Linear(speech_dim * 5, text_dim)

        def forward(self, speech_embeds):
            B, T, C = speech_embeds.shape
            speech_embeds = self.mlp_adapter(
                speech_embeds.reshape(B, T // self.scale, C * self.scale))
            speech_embeds = self.act_fn(self.gate_proj(speech_embeds)) * self.pool_proj(speech_embeds)
            return self.out_proj(speech_embeds)


logger = init_logger(__name__)


# Weight name remapping: checkpoint uses "text_decoder.model.*", vLLM uses "model.*"
_KEYS_TO_MODIFY_MAPPING = {
    "text_decoder.model": "model",
}

# === Audio processing constants === #
DEFAULT_SAMPLE_RATE = 16000
FEATURE_CHUNK_SIZE = DEFAULT_SAMPLE_RATE * 30
OUTPUT_CHUNK_SIZE = 100
MAX_NUMBER_CHUNKS = 8


def dummy_data_for_meralion(ctx: InputContext, seq_len: int,
                               mm_counts: Mapping[str, int]):
    num_audios = mm_counts["audio"]
    max_tokens_per_audio = get_max_meralion_audio_tokens(ctx)
    max_llm_audio_tokens = max_tokens_per_audio * num_audios
    if seq_len - max_llm_audio_tokens - 2 < 0:
        raise RuntimeError(
            f"MERaLiON-AudioLLM cannot process {num_audios} audios in a prompt, "
            "please increase max_model_len or reduce audio limit by "
            "--limit-mm-per-prompt.")

    speech_token_index = ctx.model_config.hf_config.speech_token_index

    dummy_seqdata = SequenceData.from_prompt_token_counts(
        (speech_token_index, max_llm_audio_tokens),
        (0, seq_len - max_llm_audio_tokens),
    )
    dummy_audio = np.full((max_llm_audio_tokens * 15 * 2 * 160, ), 0.)
    return DummyData(
        dummy_seqdata, {"audio": [(dummy_audio, DEFAULT_SAMPLE_RATE)] * num_audios}, {
            "audio":
            consecutive_placeholder_ranges(num_items=num_audios,
                                           item_size=max_tokens_per_audio)
        })


def get_processor(processor_name, *args, trust_remote_code=True, **kwargs):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(
        processor_name, *args, trust_remote_code=trust_remote_code, **kwargs)


cached_get_processor = lru_cache(get_processor)


def _get_number_chunks(audios):
    audio_lengths = np.array([_.shape[0] for _ in audios])
    number_chunks = ((audio_lengths - 1) // FEATURE_CHUNK_SIZE) + 1
    return np.clip(number_chunks, a_min=None, a_max=MAX_NUMBER_CHUNKS)


def _get_feat_extract_output_lengths(audios):
    return _get_number_chunks(audios) * OUTPUT_CHUNK_SIZE


def _get_chunked_audios(audios):
    audio_number_chunks = _get_number_chunks(audios)
    chunked = []
    for audio_idx, audio in enumerate(audios):
        for cid in range(audio_number_chunks[audio_idx]):
            chunked.append(audio[cid * FEATURE_CHUNK_SIZE: (cid + 1) * FEATURE_CHUNK_SIZE])
    return chunked


def get_max_meralion_audio_tokens(ctx: InputContext) -> int:
    return MAX_NUMBER_CHUNKS * OUTPUT_CHUNK_SIZE


def input_processor_for_meralion(
        ctx: InputContext, inputs: DecoderOnlyInputs) -> DecoderOnlyInputs:
    multi_modal_data = inputs.get("multi_modal_data")
    if multi_modal_data is None or "audio" not in multi_modal_data:
        return inputs

    audios = multi_modal_data["audio"]
    if not isinstance(audios, list):
        audios = [audios]
    if len(audios) == 0:
        return inputs

    processor = cached_get_processor(ctx.model_config.model)

    resampled_audios = [
        librosa.resample(audio, orig_sr=sr, target_sr=processor.feature_extractor.sampling_rate)
        for audio, sr in audios
    ]

    audio_output_lengths = _get_feat_extract_output_lengths(resampled_audios)
    speech_token_index = ctx.model_config.hf_config.speech_token_index

    input_ids = inputs['prompt_token_ids']
    new_input_ids = []
    audio_num = input_ids.count(speech_token_index)
    assert len(audio_output_lengths) == audio_num

    start = 0
    for audio_idx in range(audio_num):
        end = input_ids.index(speech_token_index, start)
        new_input_ids.extend(input_ids[start:end])
        new_input_ids.extend([speech_token_index] * audio_output_lengths[audio_idx])
        start = end + 1
    new_input_ids.extend(input_ids[start:])

    return token_inputs(
        prompt_token_ids=new_input_ids,
        prompt=inputs.get('prompt'),
        multi_modal_data=multi_modal_data,
    )


def input_mapper_for_meralion(ctx: InputContext, multi_modal_data):
    if not isinstance(multi_modal_data, list):
        multi_modal_data = [multi_modal_data]
    if len(multi_modal_data) == 0:
        return MultiModalKwargs()

    processor = cached_get_processor(ctx.model_config.model)
    fe = processor.feature_extractor
    target_sr = fe.sampling_rate

    resampled = [
        librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        if sr != target_sr else audio
        for audio, sr in multi_modal_data
    ]
    resampled = _get_chunked_audios(resampled)

    batch_data = fe(resampled, sampling_rate=target_sr, return_attention_mask=True,
                    padding="max_length", return_tensors="pt", do_normalize=True).data
    batch_data["feature_attention_mask"] = batch_data.pop("attention_mask")

    return MultiModalKwargs(batch_data)


def _patch_whisper_attn_projections(encoder, pruned_embed_dim: int, d_model: int,
                                    start: int = 0, end: int = -1):
    """Patch WhisperEncoder attention Linear layers to their pruned (non-square) shapes.

    After Whisper attention pruning, q/k/v projections have shape
    [pruned_embed_dim, d_model] and out_proj has shape [d_model, pruned_embed_dim].
    WhisperEncoder.__init__ creates all of them as [d_model, d_model], so we must
    replace them before load_weights to avoid shape-mismatch errors.
    """
    layers = encoder.layers
    if end < 0:
        end = len(layers)
    for i in range(start, min(end, len(layers))):
        attn = layers[i].self_attn
        has_k_bias = (attn.k_proj.bias is not None)
        attn.q_proj = nn.Linear(d_model, pruned_embed_dim, bias=True)
        attn.k_proj = nn.Linear(d_model, pruned_embed_dim, bias=has_k_bias)
        attn.v_proj = nn.Linear(d_model, pruned_embed_dim, bias=True)
        attn.out_proj = nn.Linear(pruned_embed_dim, d_model, bias=True)
        attn.embed_dim = pruned_embed_dim
        attn.head_dim = pruned_embed_dim // attn.num_heads


def _patch_whisper_ffn_layers(encoder, pruned_ffn_dim: int, d_model: int,
                               start: int = 0, end: int = -1):
    """Patch WhisperEncoder fc1/fc2 layers to their pruned shapes.

    After Whisper FFN (MLP) pruning, fc1 has shape [pruned_ffn_dim, d_model]
    and fc2 has shape [d_model, pruned_ffn_dim].
    WhisperEncoder.__init__ creates them as [encoder_ffn_dim, d_model] /
    [d_model, encoder_ffn_dim] using the original (unpruned) config value.
    """
    layers = encoder.layers
    if end < 0:
        end = len(layers)
    for i in range(start, min(end, len(layers))):
        layer = layers[i]
        layer.fc1 = nn.Linear(d_model, pruned_ffn_dim, bias=True)
        layer.fc2 = nn.Linear(pruned_ffn_dim, d_model, bias=True)


def _maybe_apply(registry, method_name, *args):
    """Apply registry.method_name(*args) as a decorator if the method exists, else no-op."""
    method = getattr(registry, method_name, None)
    if method is not None:
        return method(*args)
    return lambda cls: cls


@_maybe_apply(INPUT_REGISTRY, 'register_dummy_data', dummy_data_for_meralion)
@_maybe_apply(INPUT_REGISTRY, 'register_input_processor', input_processor_for_meralion)
@_maybe_apply(MULTIMODAL_REGISTRY, 'register_input_mapper', "audio", input_mapper_for_meralion)
@_maybe_apply(MULTIMODAL_REGISTRY, 'register_max_multimodal_tokens', "audio", get_max_meralion_audio_tokens)
class MERaLiON2PrunedForConditionalGeneration(nn.Module, SupportsMultiModal,
                                              SupportsLoRA, SupportsPP):
    """MERaLiON2 model for vLLM with support for non-uniform (pruned) layer dimensions.

    Automatically detects midblock pruning from config and uses PrunedGemma2Model
    for the text decoder. Falls back to standard Gemma2Model for unpruned models.
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    supported_lora_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        # Speech encoder (Whisper)
        config.speech_config = autoset_attn_implementation_for_whisper(config.speech_config)
        self.speech_encoder = WhisperEncoder(config.speech_config)
        # If whisper attention or FFN was pruned, the linear layers have non-square
        # shapes that differ from what WhisperEncoder.__init__ creates. Patch them
        # now so load_weights sees matching shapes.
        _ws = getattr(config.speech_config, 'whisper_midblock_start', 0)
        _we = getattr(config.speech_config, 'whisper_midblock_end', -1)
        _pruned_attn_dim = getattr(config.speech_config, 'whisper_pruned_attn_embed_dim', None)
        if _pruned_attn_dim is not None and _pruned_attn_dim != config.speech_config.d_model:
            _patch_whisper_attn_projections(
                self.speech_encoder,
                pruned_embed_dim=_pruned_attn_dim,
                d_model=config.speech_config.d_model,
                start=_ws, end=_we,
            )
        _pruned_ffn_dim = getattr(config.speech_config, 'whisper_pruned_ffn_dim', None)
        if _pruned_ffn_dim is not None and _pruned_ffn_dim != config.speech_config.encoder_ffn_dim:
            _patch_whisper_ffn_layers(
                self.speech_encoder,
                pruned_ffn_dim=_pruned_ffn_dim,
                d_model=config.speech_config.d_model,
                start=_ws, end=_we,
            )
        self.ln_speech = nn.LayerNorm(config.speech_config.d_model)
        self.speech_audio_adapter = MERaLiON2SpeechAudioAdaper(
            config.speech_config.d_model, config.text_config.hidden_size)

        self.quant_config = quant_config

        # Text decoder: always use PrunedGemma2Model, which handles both pruned
        # (non-uniform layers) and uniform models correctly, and is compatible
        # with all vLLM versions.
        text_config = config.text_config
        if is_pruned_model(text_config):
            logger.info("Detected midblock pruning config — using PrunedGemma2Model")
        else:
            logger.info("No midblock pruning detected — using PrunedGemma2Model with uniform layers")
        self.model = PrunedGemma2Model(
            vllm_config=vllm_config.with_hf_config(text_config),
            prefix=maybe_prefix(prefix, "model"),
        )

        # LM head
        self.unpadded_vocab_size = text_config.vocab_size
        if text_config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                text_config.vocab_size, text_config.hidden_size, quant_config=quant_config)

        logit_scale = getattr(config, "logit_scale", 1.0)
        import inspect as _inspect
        _lp_sig = _inspect.signature(LogitsProcessor.__init__).parameters
        _lp_kwargs = {}
        if 'logit_softcapping' in _lp_sig:
            _lp_kwargs['logit_softcapping'] = getattr(text_config, "final_logit_softcapping", None)
        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size, text_config.vocab_size, logit_scale,
            **_lp_kwargs,
        )
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def _validate_and_reshape_mm_tensor(self, mm_input, name):
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            result = torch.concat(list(mm_input))
        else:
            result = torch.concat(mm_input)
        # Flatten any leading batch/chunk dims: [..., mel, time] -> [N, mel, time]
        return result.view(-1, result.size(-2), result.size(-1))

    def _parse_and_validate_audio_input(self, **kwargs):
        input_features = kwargs.pop('input_features', None)
        feature_attention_mask = kwargs.pop('feature_attention_mask', None)
        if input_features is None:
            return None
        input_features = self._validate_and_reshape_mm_tensor(input_features, 'input_features')
        feature_attention_mask = self._validate_and_reshape_mm_tensor(
            feature_attention_mask, 'feature_attention_mask')
        return MERaLiON2Inputs(input_features=input_features,
                                feature_attention_mask=feature_attention_mask)

    def _process_audio_input(self, audio_input):
        input_features = audio_input["input_features"].to(self.speech_encoder.dtype)
        feature_attention_mask = audio_input["feature_attention_mask"]

        audio_outputs = self.speech_encoder(
            input_features, attention_mask=feature_attention_mask)
        audio_features = audio_outputs.last_hidden_state
        audio_features = self.ln_speech(audio_features)
        audio_features = self.speech_audio_adapter(audio_features)
        audio_features = audio_features.view(-1, audio_features.size(-1))
        return audio_features

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata=None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None
        elif inputs_embeds is None:
            # V0 path: compute embeddings here from audio kwargs
            audio_input = self._parse_and_validate_audio_input(**kwargs)
            if audio_input is not None:
                inputs_embeds = self.model.embed_tokens(input_ids)
                processed_audio = self._process_audio_input(audio_input)
                mask = (input_ids == self.config.speech_token_index)
                inputs_embeds[mask, :] = processed_audio
                input_ids = None
        # else: inputs_embeds already provided (V1 path)

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(self, hidden_states, sampling_metadata):
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def sample(self, logits, sampling_metadata):
        return self.sampler(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.text_config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            # Remap checkpoint keys: text_decoder.model.* -> model.*
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)

            # Handle stacked params (QKV, gate_up) — skip for speech encoder
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name or 'speech_' in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


# ---------------------------------------------------------------------------
# Newer vLLM (0.7+) requires _processor_factories for max-token allocation.
# The pruned model only differs in text-decoder layer dims; all multimodal
# processing (audio encoding, token counting) is identical to the original
# MERaLiON-2.  Reuse the plugin's processor classes when available.
# ---------------------------------------------------------------------------
def _register_processor_factory():
    """Register MERaLiON2PrunedForConditionalGeneration in vLLM's
    _processor_factories (required by vLLM 0.8+ even with V0 engine).

    Safe to call multiple times — skips silently if already registered.
    Must be called AFTER ModelRegistry.register_model for the pruned class,
    because some vLLM versions validate against ModelRegistry at registration.
    """
    if not hasattr(MULTIMODAL_REGISTRY, '_processor_factories'):
        return  # Old vLLM — already covered by _maybe_apply decorators above

    # Idempotent: skip if already registered.
    if MERaLiON2PrunedForConditionalGeneration in MULTIMODAL_REGISTRY._processor_factories:
        return

    try:
        from vllm_plugin_meralion2.vllm085 import (
            MERaLiON2MultiModalProcessor,
            MERaLiON2ProcessingInfo,
            MERaLiON2DummyInputsBuilder,
        )
        MULTIMODAL_REGISTRY.register_processor(
            MERaLiON2MultiModalProcessor,
            info=MERaLiON2ProcessingInfo,
            dummy_inputs=MERaLiON2DummyInputsBuilder,
        )(MERaLiON2PrunedForConditionalGeneration)
        logger.info("Registered MERaLiON2MultiModalProcessor for pruned model "
                    "(reused from vllm_plugin_meralion2.vllm085)")
    except Exception as e:
        import traceback
        logger.warning(f"Could not register processor factory for pruned model:\n"
                       f"{traceback.format_exc()}")
        raise RuntimeError(
            "Failed to register MERaLiON2PrunedForConditionalGeneration in "
            "MULTIMODAL_REGISTRY._processor_factories. "
            "Call _register_processor_factory() again after "
            "ModelRegistry.register_model(...)."
        ) from e


# Attempt early registration.  May fail if ModelRegistry hasn't registered
# MERaLiON2PrunedForConditionalGeneration yet — callers should call again
# after ModelRegistry.register_model to guarantee success.
try:
    _register_processor_factory()
except RuntimeError:
    pass  # Will be retried by the caller after ModelRegistry registration
