---
language:
  - en
license: apache-2.0
library_name: transformers
tags:
  - eagle
  - speculative-decoding
  - quantization
  - w4a16
  - gptq
  - asr
  - meralion
base_model: MERaLiON/MERaLiON-2-3B
pipeline_tag: automatic-speech-recognition
---

# MERaLiON-2-3B + EAGLE + W4A16 (Speculative Decoding, Quantized)

**1.83× decode-throughput speedup, 1.90× latency speedup, ~19% less VRAM, no WER regression.**

This repository combines two acceleration techniques on top of
[`MERaLiON/MERaLiON-2-3B`](https://huggingface.co/MERaLiON/MERaLiON-2-3B):

1. **EAGLE speculative decoding** — a small (~88 M param) auto-regressive
   draft model that predicts K tokens ahead of the verifier. Uses a single
   shared `Gemma2DecoderLayer` plus a fusion linear, conditioned on the
   verifier's last-layer hidden state. K=4 chain mode by default.
2. **W4A16 GPTQ quantization** of the verifier's text decoder, dispatched
   to the **ExllamaV1 CUDA kernel** at inference. The non-text branches
   (Whisper speech encoder + audio adapter) stay BF16 / FP16.

## Benchmark

IMDA_PART1, 20 held-out samples, A100-SXM4-40GB (sm_80), greedy decode,
K=4 chain:

| Verifier              | tok/s  | Latency speedup | VRAM    | WER    | Accept rate |
|-----------------------|-------:|----------------:|--------:|-------:|------------:|
| BF16 (no spec)        | 24.78  | 1.00×           | 7.16 GB | 7.12 % | —           |
| BF16 + EAGLE          | 43.84  | 1.81×           | 7.16 GB | 6.74 % | 46.8 %      |
| **W4A16 + EAGLE**     | **45.35** | **1.90×**    | **5.83 GB** | **6.74 %** | **48.4 %** |

The W4A16 + EAGLE combination beats BF16 + EAGLE on tok/s **and** saves
~1.3 GB VRAM with no quality loss.

## Quick start

```python
import torch
from modeling_eagle import MERaLiON2EAGLEForASR

model = MERaLiON2EAGLEForASR.from_pretrained(
    "YOUR_HF_USERNAME/MERaLiON-2-3B-EAGLE-W4A16",
    base_model="MERaLiON/MERaLiON-2-3B",   # for the BF16 speech encoder
    torch_dtype=torch.float16,             # required by W4A16 kernels
    gptq_kernel="exllama",                 # exllama | exllamav2 | marlin
).to("cuda")

# Standard MERaLiON-2 audio prompt — see example_inference.py for the
# full pipeline (audio loading, prompt template, tokenization).
out_ids = model.generate_eagle(
    input_ids=input_ids,
    attention_mask=attention_mask,
    input_features=input_features,
    feature_attention_mask=feature_attention_mask,
    max_new_tokens=128,
    K=4,
)
```

A complete end-to-end audio → transcript example is in
`example_inference.py`.

## Method (brief)

**EAGLE draft.** A single `Gemma2DecoderLayer` (full attention, no sliding
window) plus a `nn.Linear(2H → H)` that fuses `[embed(prev_tok),
prev_h]`. The fused state is normalized and projected through the
verifier's tied `lm_head`. Trained with cross-entropy + a small hidden-
state MSE term, on (last-layer h, next token) pairs collected by
running the frozen BF16 verifier over real ASR audio. Multi-step
unrolled training (D=4) plus scheduled sampling is used so the draft
keeps making sensible predictions when fed its own previous hidden
state at inference time.

**W4A16 quantization.** GPTQ on the text decoder only (Gemma2 stack,
~2.6 B params). Group size 128, symmetric, `desc_act=False` for kernel
compatibility. Speech encoder + audio adapter + `lm_head` are excluded
and stay in their BF16/FP16 source dtype. At load, the GPTQ
`qweight`/`qzeros`/`scales`/`g_idx` are repacked into ExllamaV1's
internal layout and the verifier dispatches to ExllamaV1's batch=1
fused W4A16 GEMM, which on A100 runs slightly faster than BF16
cuBLAS at single-token decode while reading 4× less weight memory.

**Inference loop.** Standard chain speculative decoding: each round the
draft proposes K=4 tokens auto-regressively (each step conditioned on
the previous step's hidden state plus token embedding); the verifier
runs once on `[next_tok, d_0, …, d_{K-1}]` in a single batched forward
to score them; we accept the longest matching prefix.

## Files

| File | Purpose |
|---|---|
| `eagle.safetensors` | EAGLE draft weights (~88 M params, FP16) |
| `eagle_config.json` | EAGLE hyperparameters (num_layers, hidden_size, …) |
| `text_decoder_w4a16/` | GPTQ-quantized Gemma2 text decoder (qweight + scales) |
| `text_decoder_w4a16/quantize_config.json` | bits, group_size, sym, desc_act |
| `meralion2_bl/` | Custom MERaLiON-2 modeling code (vendored) |
| `eagle_model.py` | EAGLE class |
| `modeling_eagle.py` | `MERaLiON2EAGLEForASR` wrapper with `generate_eagle()` |
| `example_inference.py` | End-to-end audio → transcript example |
| `requirements.txt` | Pinned env (auto-gptq compiled with marlin+exllama+exllamav2) |

## Requirements

A100 / H100 / RTX 30xx (sm_80+) is required for the ExllamaV1 / Marlin
W4A16 kernel. CUDA 12.x. See `requirements.txt`. The auto-gptq library
must be built from source with the W4A16 CUDA kernels — install
instructions are in the requirements file.

## Caveats

- **Greedy decoding only.** Sampling / beam not implemented for EAGLE.
- **Single batch.** The decode loop is batch-1; batched ASR would need
  per-sample EAGLE state.
- **English ASR tested.** Trained and benchmarked on IMDA_PART1; other
  languages or non-ASR tasks should still work if the verifier hidden-
  state distribution is close enough, but are not validated here.

## Citation

- Base: [MERaLiON/MERaLiON-2-3B](https://huggingface.co/MERaLiON/MERaLiON-2-3B)
- EAGLE: [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)
- GPTQ: [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- Marlin / ExLlama kernels: [auto-gptq](https://github.com/AutoGPTQ/AutoGPTQ)
