"""End-to-end example: audio file → EAGLE+W4A16-accelerated transcription.

Usage:
    python example_inference.py <audio.wav> [--repo YOUR_USERNAME/...]

Requires: transformers, auto_gptq (compiled with marlin+exllama+exllamav2),
soundfile or librosa, huggingface_hub.  See `requirements.txt`.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

# Local imports — when installed via `from_pretrained` the package is
# unpacked alongside this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modeling_eagle import MERaLiON2EAGLEForASR


SAMPLE_RATE = 16000
CHUNK_SIZE  = SAMPLE_RATE * 30
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


def prepare_audio(audio, processor):
    fe = processor.feature_extractor
    target_sr = fe.sampling_rate
    chunks = []
    for i in range(0, len(audio), CHUNK_SIZE):
        chunk = audio[i:i + CHUNK_SIZE]
        if len(chunk) < target_sr:
            chunk = np.pad(chunk, (0, target_sr - len(chunk)), "constant")
        chunks.append(chunk)
    chunks = chunks[:MAX_CHUNKS]
    out = fe(chunks, sampling_rate=target_sr, return_attention_mask=True,
             padding="max_length", return_tensors="pt", do_normalize=True)
    n_speech = len(chunks) * SPEECH_TOKENS_PER_CHUNK
    return out.input_features, out.attention_mask, n_speech


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_path")
    ap.add_argument("--repo",
                    default="YOUR_HF_USERNAME/MERaLiON-2-3B-EAGLE-W4A16",
                    help="HF repo id (or local path) of this package")
    ap.add_argument("--base", default=None,
                    help="(optional) Override BF16 base path. By default we "
                         "use the bundled base_bf16/ inside --repo.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--K",      type=int, default=4)
    ap.add_argument("--gptq_kernel", default="exllama",
                    choices=["exllama", "exllamav2", "marlin"])
    ap.add_argument("--instruction", default="Transcribe the speech")
    args = ap.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"audio not found: {args.audio_path}"); sys.exit(1)

    model = MERaLiON2EAGLEForASR.from_pretrained(
        args.repo, base_model=args.base,
        torch_dtype=torch.float16,
        gptq_kernel=args.gptq_kernel, device=args.device)
    processor = model.processor
    tokenizer = processor.tokenizer

    print(f"Loading audio: {args.audio_path}")
    audio = load_audio(args.audio_path)
    input_features, feature_attention_mask, n_speech = prepare_audio(audio, processor)
    input_features = input_features.to(args.device).to(torch.float16)
    feature_attention_mask = feature_attention_mask.to(args.device)

    speech_token_id = model.config.speech_token_index
    conv = [{"role": "user",
             "content": (f"Instruction: {args.instruction} \n"
                         "Follow the text instruction based on the "
                         "following audio: <SpeechHere>")}]
    prompt = tokenizer.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=True)
    raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
    pos = raw_ids.index(speech_token_id)
    input_ids = torch.tensor(
        [raw_ids[:pos] + [speech_token_id] * n_speech + raw_ids[pos + 1:]],
        dtype=torch.long, device=args.device)
    attention_mask = torch.ones_like(input_ids)

    # Warmup pass — first call eats CUDA kernel JIT + allocator init time;
    # don't include it in the timing.  Same audio / prompt, short max_new.
    print("Warmup …")
    _ = model.generate_eagle(
        input_ids=input_ids, attention_mask=attention_mask,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
        max_new_tokens=16, K=args.K,
    )
    torch.cuda.synchronize()

    print(f"Transcribing with EAGLE+W4A16 (K={args.K}, kernel={args.gptq_kernel}) …")
    torch.cuda.synchronize()
    t0 = time.time()
    out_ids = model.generate_eagle(
        input_ids=input_ids, attention_mask=attention_mask,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
        max_new_tokens=args.max_new_tokens, K=args.K,
    )
    torch.cuda.synchronize()
    dt = time.time() - t0
    n_gen = out_ids.shape[1] - input_ids.shape[1]

    text = tokenizer.decode(out_ids[0, input_ids.shape[1]:],
                             skip_special_tokens=True)
    text = text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()
    print(f"\n  decode: {dt:.2f}s  ({n_gen / dt:.1f} tok/s)")
    print(f"\nTranscription:\n  {text}")


if __name__ == "__main__":
    main()
