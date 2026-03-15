"""Inference-only MERaLiON AudioLLM model compatible with HuggingFace weights."""
from typing import Any, Optional, Set, Tuple, TypedDict, Union, List

import torch
import torch.nn as nn
from transformers.utils.import_utils import is_torch_sdpa_available, is_flash_attn_2_available

# === Audio Inputs === #
class MERaLiON2Inputs(TypedDict):
    input_features: torch.Tensor
    """Shape: 
    `(num_audios, num_mel_bins, 3000)`
    """

    feature_attention_mask: torch.Tensor
    """Shape: `(num_audios, 3000)`
    """


# === Audio Encoder === #
class MERaLiON2SpeechAudioAdaper(nn.Module):
    def __init__(self, audio_hidden_size: int, text_hidden_size: int):
        super(MERaLiON2SpeechAudioAdaper, self).__init__()
        speech_mlp_scale_factor = 15

        self.speech_mlp_scale_factor = speech_mlp_scale_factor
        self.mlp_adapter = nn.Sequential(
            nn.Linear(
                in_features=audio_hidden_size * speech_mlp_scale_factor,
                out_features=audio_hidden_size * 5,
            ),
            nn.SiLU(),
        )

        self.gate_proj = nn.Linear(
                in_features=audio_hidden_size * 5,
                out_features=audio_hidden_size * 5,
            )
        
        self.pool_proj = nn.Linear(
                in_features=audio_hidden_size * 5,
                out_features=audio_hidden_size * 5,
        )
        self.act_fn = nn.SiLU()
        self.out_proj = nn.Linear(
                audio_hidden_size * 5,
                text_hidden_size,
        )

    def forward(self, speech_embeds, **kwargs):
        B, T, C = speech_embeds.shape
        speech_embeds = self.mlp_adapter(
            speech_embeds.reshape(
                B,
                T // self.speech_mlp_scale_factor,
                C * self.speech_mlp_scale_factor,
            )
        )
        speech_embeds = self.act_fn(self.gate_proj(speech_embeds)) * self.pool_proj(speech_embeds)
        speech_embeds = self.out_proj(speech_embeds)
        return speech_embeds
    

def autoset_attn_implementation_for_whisper(config):
    _implementation = "eager"
    if is_torch_sdpa_available():
        _implementation = "sdpa"
    if is_flash_attn_2_available():
        _implementation = "flash_attention_2"

    config._attn_implementation = _implementation
    return config
