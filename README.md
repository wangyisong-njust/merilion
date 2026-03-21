# MERaLiON-2 Model Compression and Inference Optimization

This project applies structural pruning, quantization, and inference optimization to the [MERaLiON-2](https://huggingface.co/MERaLiON) multimodal ASR (Automatic Speech Recognition) model. The goal is to significantly reduce parameter count and inference latency while preserving transcription accuracy, enabling efficient deployment on CPU and other resource-constrained edge devices.

---

## Model Architecture

### MERaLiON-2 Base Architecture

MERaLiON-2 is a multimodal large language model developed with support from Singapore's National Research Foundation, focused on English speech recognition. Its architecture consists of three core components:

```
Audio Input
     ↓
Speech Encoder (Whisper architecture)
     ↓
Audio Adapter (linear projection)
     ↓
Text Decoder (Gemma2 architecture)
     ↓
Transcription Output
```

**Speech Encoder**
- Transformer encoder based on OpenAI Whisper
- Maps raw audio features (mel spectrogram) to audio embeddings
- Processes long audio in 30-second chunks
- Kept in FP32/FP16 during inference — not quantized

**Audio Adapter**
- Aligns audio embeddings to the text decoder's dimension space
- Lightweight linear layer with minimal parameters

**Text Decoder (MERaLiON-2-3B)**
- Decoder-only Transformer based on Gemma2
- Default configuration (before pruning):
  - Layers: 26
  - Hidden size: `hidden_size = 2048`
  - Attention heads: `num_attention_heads = 8` (GQA, `num_key_value_heads = 4`)
  - FFN intermediate size: `intermediate_size = 8192`
  - Activation: GeGLU (Gated Linear Unit)

---

## Compression Pipeline and Technical Architecture

### 1. Structural Pruning (LLM-Pruner)

The text decoder of MERaLiON-2 is structurally pruned using the [LLM-Pruner](https://arxiv.org/abs/2305.11627) framework.

#### Pruning Framework

Pruning proceeds in three stages:

**Stage 1 — Discovery Stage**
- Automatically analyzes inter-layer coupling via a `DependencyGraph`
- Identifies the minimally removable unit (Group) to ensure valid model structure after pruning
- Attention head pruning must operate on whole heads (Q/K/V are linked)

**Stage 2 — Estimation Stage**
- Uses **Taylor importance estimation** to measure each group's contribution to model performance
- Computes first-order gradients and approximate second-order Hessian from a small calibration set (~20 audio samples)
- Options: `param_mix` (combined first+second order, default), `param_first`, `l1`, `l2`

**Stage 3 — Recover Stage**
- Recovers model performance after pruning via LoRA fine-tuning (see below)

#### Mid-Block Pruning Strategy

Pruning is applied only to the middle Transformer layers, preserving the full capacity of the first and last layers:

```
Layer 0 ~ start-1     ← kept intact (input feature extraction)
Layer start ~ end-1   ← pruned at pruning_ratio (mid-block)
Layer end ~ N-1       ← kept intact (output layers)
```

Typical configuration (25% pruning, 3B model with 26 layers):
- `--block_mlp_layer_start 3 --block_mlp_layer_end 23`
- `--block_attention_layer_start 3 --block_attention_layer_end 23`

After pruning, layer dimensions are no longer uniform. The custom model code (`meralion2_bl/`) uses `DynamicCache` instead of the original `HybridCache` to support variable KV cache dimensions.

#### Pruning Command

```bash
python meralion.py \
  --base_model MERaLiON/MERaLiON-2-3B \
  --pruning_ratio 0.25 \
  --block_wise \
  --block_mlp_layer_start 3 --block_mlp_layer_end 23 \
  --block_attention_layer_start 3 --block_attention_layer_end 23 \
  --pruner_type taylor \
  --num_examples 20 \
  --device cuda \
  --save_model \
  --save_ckpt_log_name meralion_pruned
```

---

### 2. LoRA Post-Training (Recovery Stage)

After pruning, Low-Rank Adaptation (LoRA) fine-tuning is used to efficiently recover model performance:

- LoRA adapters are inserted into the linear layers of the text decoder (Gemma2 part) only
- Dataset: IMDA English ASR data (Part 1) and mixed corpora
- Supports multi-GPU training (DeepSpeed / gradient accumulation)

```bash
python post_training_meralion.py \
  --prune_model meralion_checkpoints/meralion_pruned \
  --data_path imda_asr \
  --lora_r 16 \
  --learning_rate 5e-5 \
  --num_epochs 3 \
  --output_dir meralion_tune_log/my_tune
```

Merge LoRA weights after fine-tuning:

```bash
python merge_meralion.py \
  --ckpt meralion_checkpoints/meralion_pruned \
  --lora_ckpt meralion_tune_log/my_tune/checkpoint-final \
  --save_path merged_model
```

---

### 3. Quantization Schemes

Multiple quantization schemes are implemented for CPU inference (`infer_cpu.py`), allowing a precision/speed trade-off:

| Scheme | Precision | Speedup | Hardware | Flag |
|--------|-----------|---------|----------|------|
| None | FP32 | 1× | Any | (default) |
| INT8 Dynamic | Weights INT8, activations FP32 | ~1.5–2× | Any | `--int8` |
| INT8 Weight-only (torchao) | Weights INT8, activations FP32 | ~1.3–1.8× | Any, torch.compile compatible | `--int8ao` |
| **W8A8** (INT8 × INT8) | Weights + activations INT8 | ~2–3× | AVX-512 VNNI / AMX | `--w8a8` |
| INT4 Weight-only | Weights INT4, activations FP32 | ~2–4× | Any (experimental) | `--int4` |

**W8A8 quantization** is the primary optimization target. It leverages VNNI/AMX instructions on modern x86 CPUs for true INT8 matrix multiplication:

- Only the linear layers in `text_decoder.model` (Gemma2 Transformer blocks) are quantized — both weights and runtime activations
- `speech_encoder`, `audio_adapter`, and `lm_head` remain FP32 to avoid accuracy loss
- Implemented via [torchao](https://github.com/pytorch/ao) `Int8DynamicActivationInt8WeightConfig`

---

### 4. torch.compile Inference Acceleration

`torch.compile` is used to compile and optimize the inference computation graph:

- **`max-autotune` mode** (default): Auto-tunes CPU GEMM kernels; best for INT8/FP32 inference
- **`reduce-overhead` mode**: Reduces Python dispatch overhead; better for smaller batch sizes

```bash
# Enable compile with mode selection
python infer_cpu.py --model merged_model --w8a8 --compile --compile_mode max-autotune
python infer_cpu.py --model merged_model --int8ao --compile --compile_mode reduce-overhead
```

Note: The core compute of INT4 and W8A8 is handled by torchao kernels and accelerates even without `--compile`; combining both can further improve throughput.

---

### 5. .mera Packed Format (Edge Distribution)

To simplify model distribution and loading on edge devices, the project implements the `.mera` single-file packed format (`pack_model.py`):

**Format Specification (MERA v1)**

```
Bytes  0–3:    Magic: b"MERA"
Bytes  4–7:    Version: uint32 LE = 1
Bytes  8–15:   Header length: uint64 LE (including alignment padding)
Bytes 16–N:    Header JSON (UTF-8), zero-padded to 64-byte alignment
Bytes  N+:     Tensor data blocks, each aligned to 64 bytes
```

**Header JSON Structure**

```json
{
  "format_version": 1,
  "model_config": { ... },           // config.json contents
  "configs": {                        // auxiliary configs (tokenizer, etc.)
    "tokenizer.json": { ... },
    "processor_config.json": { ... }
  },
  "source_files": {                   // model Python source (e.g. processor class)
    "processing_meralion2.py": "..."
  },
  "storage": "int8",                  // or "float16"
  "tensors": {                        // tensor index table
    "text_decoder.model.layers.0.mlp.gate_proj.weight": {
      "dtype": "int8", "shape": [1024, 2048],
      "offset": 1234, "nbytes": 2097152
    },
    "text_decoder.model.layers.0.mlp.gate_proj.weight_scale": {
      "dtype": "float32", "shape": [1024],
      "offset": 3331586, "nbytes": 4096
    }
  }
}
```

**Quantization Strategy**
- `text_decoder.model.layers.*.*.weight`: INT8 per-output-channel symmetric quantization + FP32 scale
- All other tensors (speech encoder, audio adapter, etc.): FP16

Pack and load:

```bash
# Pack model
python pack_model.py --model merged_model --output model.mera

# Load and run inference
python infer_cpu.py --model model.mera --w8a8 --compile --audio sample.wav
```

---

## Project Structure

```
merilion/
├── LLMPruner/                    # Core pruning framework library
│   ├── torch_pruning/            # Dependency graph, importance estimation, pruning algorithms
│   ├── models/                   # LLM architecture adapters (LLaMA, ChatGLM, etc.)
│   ├── pruner/                   # MERaLiON-specific pruner
│   ├── evaluator/                # PPL evaluation module
│   └── peft/                     # LoRA implementation
├── meralion2_bl/                 # Modified HuggingFace model code
│   ├── modeling_gemma2.py        # Gemma2 with DynamicCache support for pruned models
│   └── modeling_whisper.py       # Whisper audio encoder
├── scripts/                      # Shell scripts
│   └── meralion.sh               # MERaLiON pruning + fine-tuning pipeline
├── audiobench/                   # ASR evaluation dataset utilities
├── vllm_inference/               # vLLM inference integration
├── meralion_checkpoints/         # Pruned model checkpoints
├── meralion_tune_log/            # LoRA fine-tuning logs and weights
├── meralion.py                   # MERaLiON pruning main script
├── post_training_meralion.py     # MERaLiON LoRA fine-tuning
├── merge_meralion.py             # Merge pruned model + LoRA adapters
├── infer_cpu.py                  # CPU inference (quantization + compile)
├── pack_model.py                 # .mera format packing tool
├── make_demo_html.py             # Interactive HTML demo generator
└── run_demo.sh                   # One-shot benchmark + demo generation
```

---

## Quick Start

### Dependencies

```bash
pip install torch torchao transformers peft deepspeed
pip install jiwer  # for WER computation
```

### End-to-End Pipeline

```bash
# 1. Prune (25% parameters, mid-block strategy)
python meralion.py \
  --base_model MERaLiON/MERaLiON-2-3B \
  --pruning_ratio 0.25 \
  --block_wise \
  --block_mlp_layer_start 3 --block_mlp_layer_end 23 \
  --block_attention_layer_start 3 --block_attention_layer_end 23 \
  --save_model --save_ckpt_log_name meralion_pruned

# 2. LoRA fine-tuning to recover performance
python post_training_meralion.py \
  --prune_model meralion_checkpoints/meralion_pruned \
  --output_dir meralion_tune_log/my_tune

# 3. Merge weights
python merge_meralion.py \
  --ckpt meralion_checkpoints/meralion_pruned \
  --lora_ckpt meralion_tune_log/my_tune/checkpoint-final \
  --save_path merged_model

# 4. Pack into .mera format
python pack_model.py --model merged_model --output model.mera

# 5. CPU inference (W8A8 + compile)
python infer_cpu.py \
  --model model.mera \
  --w8a8 --compile \
  --audio sample.wav
```

### One-Shot Demo

```bash
bash run_demo.sh
```

Generates an interactive HTML demo page with Pareto charts and audio sample comparisons.

---

## Changelog

| Commit | Change |
|--------|--------|
| `a4f0b5d` | Add W8A8 quantization (INT8 weights × INT8 dynamic activations) with VNNI/AMX CPU acceleration |
| `809ce90` | Implement `.mera` single-file packed format for edge device distribution |
| `db8381a` | Switch torch.compile default mode to `max-autotune` for better CPU GEMM performance |
| `267c4a9` | Add `--compile_mode` CLI flag for runtime compile mode selection |
| `1f208c7` | Fix cache type: use `HybridCache` for original model, `DynamicCache` for pruned |
| `6c2ce7c` | Fix missing `<bos>`: use `apply_chat_template` in both inference paths |
| `8617a66` | Add WER text normalization (lowercase + strip punctuation) for consistent evaluation |
| `ba8f208` | Fix `HybridCache` overflow: pre-allocate with prefill + max_new_tokens capacity |
| `e8a3622` | Add `run_demo.sh`: one-shot benchmark + HTML demo generation |

---

## Evaluation

Inference is evaluated on WER (Word Error Rate) and latency using the IMDA Part 1 ASR test set (indices 11000–15999).

```bash
python infer_cpu.py \
  --model model.mera \
  --w8a8 --compile \
  --dataset /path/to/imda_test \
  --output wer+latency
```

---

## Acknowledgements

- [LLM-Pruner](https://arxiv.org/abs/2305.11627) (Ma et al., NeurIPS 2023) — structural pruning framework
- [MERaLiON-2](https://huggingface.co/MERaLiON) — base model
- [torchao](https://github.com/pytorch/ao) — INT8/INT4 quantization kernels
- [Torch-Pruning](https://github.com/VainF/Torch-Pruning) — dependency-graph pruning engine
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — evaluation toolkit

---

## Citation

If you use the pruning methods from this project, please cite:

```bibtex
@inproceedings{ma2023llmpruner,
  title={LLM-Pruner: On the Structural Pruning of Large Language Models},
  author={Xinyin Ma and Gongfan Fang and Xinchao Wang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023},
}
```
