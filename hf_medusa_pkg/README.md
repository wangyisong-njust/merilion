---
language:
  - en
license: apache-2.0
library_name: transformers
tags:
  - medusa
  - speculative-decoding
  - asr
  - meralion
base_model: MERaLiON/MERaLiON-2-3B
pipeline_tag: automatic-speech-recognition
---

# MERaLiON-2-3B + Medusa (Speculative Decoding Adapter)

**~1.5× greedy-decoding speedup for ASR, WER unchanged, +40 MB VRAM.**

This repository contains **Medusa heads** trained on top of
[`MERaLiON/MERaLiON-2-3B`](https://huggingface.co/MERaLiON/MERaLiON-2-3B).
K=4 small residual-MLP heads predict the next-next, next-next-next, …
tokens from the last-layer hidden state of the text decoder, so each
decode round produces up to 5 accepted tokens from a single batched
verifier forward pass.

The weights here are **adapter-only (~42 MB)** — the base model is
pulled from its own HF repo at load time.

## Benchmark

IMDA_PART1, 20 samples, L40 GPU, BF16 + FlashAttention-2, greedy decode:

| Config           | Latency (s/sample) | Throughput (tok/s) | WER    | Accept rate |
|------------------|-------------------:|-------------------:|-------:|------------:|
| Base (no Medusa) | 0.53               | 38.0               | 1.51 % | —           |
| **+ Medusa (K=4)** | **0.35**        | **57.3**           | 1.51 % | 38.8 %      |
| **Speedup**      | **1.51×**          | **1.51×**          | same   |             |

Also tested with quantized verifiers (heads trained only in BF16):

| Verifier | Base tps | Medusa tps | Speedup | WER    |
|----------|---------:|-----------:|--------:|-------:|
| BF16     | 38.0     | 57.3       | 1.51×   | 1.51 % |
| FP16     | 37.5     | 56.3       | 1.51×   | 1.51 % |
| MLX-INT4 | 22.0     | 35.7       | 1.62×   | 1.51 % |
| BnB-INT8 | 10.3     | 17.5       | 1.70×   | 1.51 % |
| BnB-INT4 | 20.1     | 27.4       | 1.37×   | 2.64 % † |

† INT4 WER hit is a pre-existing bitsandbytes property, not Medusa-induced.

## Usage

```python
import torch
from transformers import AutoProcessor
from modeling_medusa import MERaLiON2MedusaForASR

model = MERaLiON2MedusaForASR.from_pretrained(
    "YOUR_HF_USERNAME/MERaLiON-2-3B-Medusa",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")

processor = AutoProcessor.from_pretrained(
    "MERaLiON/MERaLiON-2-3B", trust_remote_code=True,
)

# Prepare audio + prompt exactly like the base model expects
# (see MERaLiON-2-3B model card for the full template).
audio, sr = ...  # numpy float32, target sample rate
input_features = processor.feature_extractor(
    audio, sampling_rate=sr, return_tensors="pt",
).input_features.to("cuda").to(torch.bfloat16)

conversation = [{
    "role": "user",
    "content": ("Instruction: Transcribe the speech \n"
                "Follow the text instruction based on the "
                "following audio: <SpeechHere>"),
}]
prompt = processor.tokenizer.apply_chat_template(
    conversation, tokenize=False, add_generation_prompt=True)
# ... see included `example_inference.py` for the full pipeline ...

out_ids = model.generate_medusa(
    input_ids=input_ids, attention_mask=attention_mask,
    input_features=input_features, feature_attention_mask=fam,
    max_new_tokens=128,
)
hyp = processor.tokenizer.decode(out_ids[0, input_ids.shape[1]:],
                                 skip_special_tokens=True)
```

A complete end-to-end example is in `example_inference.py`.

## How the heads were trained

1. **Collect hidden states** by running the frozen BF16 base model
   (speech encoder + adapter + text decoder) over 10 000 IMDA_PART1
   training audios and recording `(last_layer_hidden_state, next_token)`
   at every decoded position.  This avoids the train/inference
   distribution mismatch that kills heads trained on pure text.
2. **Train the 21 M parameter MLP heads** (4 heads × 1 residual block
   each, sharing the base `lm_head`).  3 epochs, batch 16, lr 1e-3,
   cosine schedule, ~3 min on a single L40.  Best checkpoint picked on a
   5 % held-out validation split.
3. Per-head held-out next-token accuracy:
   - head 0 (offset +1): 76 %
   - head 1 (offset +2): 47 %
   - head 2 (offset +3): 35 %
   - head 3 (offset +4): 27 %

## Files

| File | Purpose |
|---|---|
| `adapter_config.json` | Base-model pointer and head hyperparameters |
| `medusa_heads.safetensors` | Trained head weights (BF16, ~42 MB) |
| `modeling_medusa.py` | `MERaLiON2MedusaForASR` class with `generate_medusa()` |
| `example_inference.py` | End-to-end audio → transcript example |

## Caveats

- **Greedy decoding only.** Sampling / beam search not yet implemented.
- **Single batch.** Decode loop is batch-size-1; batched inference needs
  tree attention (not implemented).
- Heads trained and tested on English ASR (IMDA_PART1). Other languages
  / tasks untested but should work if hidden-state distribution matches.

## Citation / Acknowledgement

- Base model: [MERaLiON/MERaLiON-2-3B](https://huggingface.co/MERaLiON/MERaLiON-2-3B)
- Medusa paper: [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
