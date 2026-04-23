"""End-to-end example: audio file → Medusa-accelerated transcription.

Usage:
    python example_inference.py <audio.wav> [--repo YOUR_USERNAME/MERaLiON-2-3B-Medusa]

Requires: transformers, safetensors, librosa (or soundfile), huggingface_hub.
"""
import argparse
import os
import sys

import numpy as np
import torch
from transformers import AutoProcessor

from modeling_medusa import MERaLiON2MedusaForASR


SAMPLE_RATE = 16000
CHUNK_SIZE = SAMPLE_RATE * 30
SPEECH_TOKENS_PER_CHUNK = 100
MAX_CHUNKS = 8


def load_audio(path, target_sr=SAMPLE_RATE):
    try:
        import soundfile as sf
        audio, sr = sf.read(path, always_2d=False, dtype="float32")
    except Exception:
        import librosa
        audio, sr = librosa.load(path, sr=None, mono=True)
    if audio.ndim == 2:
        audio = audio.mean(axis=-1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def prepare_audio(audio: np.ndarray, processor):
    fe = processor.feature_extractor
    target_sr = fe.sampling_rate
    chunks = []
    for i in range(0, len(audio), CHUNK_SIZE):
        chunk = audio[i:i + CHUNK_SIZE]
        if len(chunk) < target_sr:
            chunk = np.pad(chunk, (0, target_sr - len(chunk)), "constant")
        chunks.append(chunk)
    chunks = chunks[:MAX_CHUNKS]
    out = fe(
        chunks, sampling_rate=target_sr, return_attention_mask=True,
        padding="max_length", return_tensors="pt", do_normalize=True,
    )
    n_speech = len(chunks) * SPEECH_TOKENS_PER_CHUNK
    return out.input_features, out.attention_mask, n_speech


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_path")
    ap.add_argument("--repo", default="YOUR_HF_USERNAME/MERaLiON-2-3B-Medusa",
                    help="HF repo id of this Medusa adapter")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--instruction", default="Transcribe the speech")
    args = ap.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"audio not found: {args.audio_path}")
        sys.exit(1)

    print(f"Loading Medusa model from {args.repo} …")
    model = MERaLiON2MedusaForASR.from_pretrained(
        args.repo, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device)

    processor = AutoProcessor.from_pretrained(
        model.adapter_config["base_model_name_or_path"], trust_remote_code=True)
    tokenizer = processor.tokenizer

    print(f"Loading audio: {args.audio_path}")
    audio = load_audio(args.audio_path)
    input_features, feature_attention_mask, n_speech = prepare_audio(audio, processor)
    input_features = input_features.to(args.device).to(torch.bfloat16)
    feature_attention_mask = feature_attention_mask.to(args.device)

    speech_token_id = model.config.speech_token_index
    conversation = [{
        "role": "user",
        "content": (f"Instruction: {args.instruction} \n"
                    "Follow the text instruction based on the "
                    "following audio: <SpeechHere>"),
    }]
    prompt = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True)
    raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
    pos = raw_ids.index(speech_token_id)
    input_ids = torch.tensor(
        [raw_ids[:pos] + [speech_token_id] * n_speech + raw_ids[pos + 1:]],
        dtype=torch.long, device=args.device)
    attention_mask = torch.ones_like(input_ids)

    print("Transcribing with Medusa …")
    out_ids = model.generate_medusa(
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
        max_new_tokens=args.max_new_tokens,
    )
    gen = out_ids[0, input_ids.shape[1]:].tolist()
    text = tokenizer.decode(gen, skip_special_tokens=True)
    text = text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()
    print(f"\nTranscription:\n  {text}")


if __name__ == "__main__":
    main()
