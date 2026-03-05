"""PyTorch MERaLiON2 model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import os
import glob
from safetensors.torch import load_file

from .modeling_gemma2 import Gemma2ForCausalLM
from .modeling_whisper import WhisperEncoder
from transformers.cache_utils import HybridCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from .configuration_meralion2 import MERaLiON2Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MERaLiON2Config"


# Copied from transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_with_cache_position
def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    if attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
    return causal_mask


@dataclass
class MERaLiON2OutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.FloatTensor] = None


MERALION_START_DOCSTRING = """
    Meralion-2 model architecture.
"""


# @add_start_docstrings(
#     "The bare MERaLiON2 Model outputting raw hidden-states without any specific head on top.",
#     MERALION_START_DOCSTRING,
# )
class MERaLiON2PreTrainedModel(PreTrainedModel):
    config_class = MERaLiON2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WhisperEncoderLayer", "WhisperDecoderLayer", "Gemma2DecoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.init_std if hasattr(self.config, "init_std") else self.config.speech_config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class MERaLiON2SpeechAudioAdaper(nn.Module):
    def __init__(self, config, **kwargs):
        super(MERaLiON2SpeechAudioAdaper, self).__init__()
        speech_audio_encoder_output_dim = config.speech_config.d_model
        llm_input_hidden_size = config.text_config.hidden_size
        speech_mlp_scale_factor = config.speech_mlp_scale_factor
        self.speech_mlp_scale_factor = speech_mlp_scale_factor
        self.mlp_adapter = nn.Sequential(
            nn.Linear(in_features=speech_audio_encoder_output_dim * speech_mlp_scale_factor, out_features=speech_audio_encoder_output_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        self.speech_llm_proj = nn.Sequential(
                nn.Linear(speech_audio_encoder_output_dim, speech_audio_encoder_output_dim * 4),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(speech_audio_encoder_output_dim * 4, llm_input_hidden_size),
            )

    def forward(self, speech_embeds, **kwargs):
        B, T, C = speech_embeds.shape
        speech_embeds = self.mlp_adapter(speech_embeds.reshape(B, T // self.speech_mlp_scale_factor, C * self.speech_mlp_scale_factor))
        return self.speech_llm_proj(speech_embeds)
    

class MERaLiON2SpeechAudioAdaperLarge(nn.Module):
    def __init__(self, config, **kwargs):
        super(MERaLiON2SpeechAudioAdaperLarge, self).__init__()
        speech_audio_encoder_output_dim = config.speech_config.d_model
        llm_input_hidden_size = config.text_config.hidden_size
        speech_mlp_scale_factor = config.speech_mlp_scale_factor
        self.speech_mlp_scale_factor = speech_mlp_scale_factor
        self.mlp_adapter = nn.Sequential(
            nn.Linear(in_features=speech_audio_encoder_output_dim * speech_mlp_scale_factor, out_features=speech_audio_encoder_output_dim * 5),
            nn.SiLU(),
            nn.Dropout(0.01),
        )
        self.gate_proj = nn.Linear(in_features=speech_audio_encoder_output_dim * 5, out_features=speech_audio_encoder_output_dim * 5)
        self.pool_proj = nn.Linear(in_features=speech_audio_encoder_output_dim * 5, out_features=speech_audio_encoder_output_dim * 5)
        self.act_fn = nn.SiLU()
        self.out_proj = nn.Linear(speech_audio_encoder_output_dim * 5, llm_input_hidden_size)

    def forward(self, speech_embeds, **kwargs):
        B, T, C = speech_embeds.shape
        speech_embeds = self.mlp_adapter(speech_embeds.reshape(B, T // self.speech_mlp_scale_factor, C * self.speech_mlp_scale_factor))
        speech_embeds = self.act_fn(self.gate_proj(speech_embeds)) * self.pool_proj(speech_embeds)
        speech_embeds = self.out_proj(speech_embeds)
        return speech_embeds


MERALION_INPUTS_DOCSTRING = """
    input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
        Indices of input sequence tokens.
"""

# @add_start_docstrings(
#     """The MERALION model which consists of a audio backbone and a language model.""",
#     MERALION_START_DOCSTRING,
# )
class MERaLiON2ForConditionalGeneration(MERaLiON2PreTrainedModel, GenerationMixin):
    def __init__(self, config: MERaLiON2Config):
        config.text_config._attn_implementation = config._attn_implementation
        config.speech_config._attn_implementation = config._attn_implementation
        super().__init__(config)
        self.speech_encoder = WhisperEncoder(config.speech_config)
        self.ln_speech = nn.LayerNorm(config.speech_config.d_model)
        self.speech_audio_adapter = MERaLiON2SpeechAudioAdaperLarge(config)
        self.vocab_size = config.text_config.vocab_size
        self.text_decoder = Gemma2ForCausalLM(config.text_config)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self._padding_side = "left"
        self.post_init()

    @property
    def padding_side(self):
        return self._padding_side

    @padding_side.setter
    def padding_side(self, padding_side: str):
        if padding_side not in ["left", "right"]:
            raise ValueError(f"{padding_side} is not `left` or `right`.")
        self._padding_side = padding_side

    def get_input_embeddings(self):
        return self.text_decoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_decoder.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.text_decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.text_decoder.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.text_decoder.set_decoder(decoder)

    def get_decoder(self):
        return self.text_decoder.get_decoder()

    def tie_weights(self):
        return self.text_decoder.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.text_decoder.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    # @add_start_docstrings_to_model_forward(MERALION_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=MERaLiON2OutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_features: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MERaLiON2OutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        speech_encoder_device = self.speech_encoder.device
        if input_features is not None:
            input_features = input_features.to(speech_encoder_device)
            feature_attention_mask = feature_attention_mask.to(speech_encoder_device)
            if inputs_embeds is None:
                speech_contexts_embeds = self.speech_encoder(input_features, attention_mask=feature_attention_mask).last_hidden_state
                speech_contexts_embeds = self.ln_speech(speech_contexts_embeds)
                speech_audio_contexts_embeds = self.speech_audio_adapter(speech_contexts_embeds)
                inputs_embeds = self.text_decoder.base_model.embed_tokens(input_ids)
                speech_mask = (input_ids == self.config.speech_token_index).unsqueeze(-1)
                speech_mask = speech_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(speech_mask, speech_audio_contexts_embeds)
                input_ids = None
        outputs = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels
        )
        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=None,
        **kwargs,
    ):
        is_first_step = cache_position[0].item() == 0
        if past_key_values is not None:
            if inputs_embeds is not None:
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)
        if inputs_embeds is not None and is_first_step:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}
        if (
            isinstance(past_key_values, HybridCache)
            and attention_mask.ndim == 2
            and not self.config._attn_implementation == "flash_attention_2"
        ):
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device
            dtype = self.text_decoder.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min
            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache
            }
        )
        if is_first_step:
            model_inputs["input_features"] = input_features
            model_inputs["feature_attention_mask"] = feature_attention_mask
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.text_decoder._reorder_cache(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.get("config", None)
        if config is None:
            config_file = os.path.join(pretrained_model_name_or_path, "config.json")
            if os.path.exists(config_file):
                config = MERaLiON2Config.from_json_file(config_file)
            else:
                config = MERaLiON2Config.from_pretrained(pretrained_model_name_or_path)
        
        model = cls(config)
        
        print(f"Loading state dict from {pretrained_model_name_or_path} to detect pruned shapes...")
        state_dict = {}
        sf_files = glob.glob(os.path.join(pretrained_model_name_or_path, "*.safetensors"))
        if not sf_files:
            bin_files = glob.glob(os.path.join(pretrained_model_name_or_path, "*.bin"))
            for f in bin_files:
                state_dict.update(torch.load(f, map_location="cpu"))
        else:
            for f in sf_files:
                state_dict.update(load_file(f, device="cpu"))
        
        if hasattr(model, "speech_encoder"):
            model.speech_encoder.resize_to_match(state_dict, "speech_encoder")
        if hasattr(model, "text_decoder"):
            model.text_decoder.resize_to_match(state_dict, "text_decoder")
            
        print("Loading weights into resized layers...")
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Load state dict result: {msg}")
        
        if "text_decoder.lm_head.weight" not in state_dict:
            print("Tying lm_head weights to embed_tokens...")
            model.tie_weights()
        
        torch_dtype = kwargs.get("torch_dtype", config.torch_dtype)
        if torch_dtype is not None:
            model.to(dtype=torch_dtype)
        
        return model
