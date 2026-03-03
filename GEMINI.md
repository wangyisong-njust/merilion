# GEMINI.md

## Project Overview

This project, `LLM-Pruner`, is a comprehensive framework for applying structural pruning to Large Language Models (LLMs). Its goal is to reduce the size and computational cost of models like LLaMA, Llama-2, Llama-3, BLOOM, and various vision-language models (VLMs) while preserving their performance.

The pruning process involves three main stages:
1.  **Discovery:** Analyzing the model's architecture to identify inter-dependencies and define minimally-removable units called "groups".
2.  **Estimation:** Using an importance criterion (e.g., L1/L2 magnitude, Taylor series approximation) to score these groups.
3.  **Recovery:** Fine-tuning the pruned model, typically using Low-Rank Adaptation (LoRA), to recover any lost performance.

The core technologies used are Python, PyTorch, and the Hugging Face ecosystem (`transformers`, `peft`).

## Key Files

-   `README.md`: The main entry point for understanding the project's purpose, features, and evaluation results.
-   `requirement.txt`: A list of all the necessary Python packages.
-   `hf_prune.py`: The main script for pruning standard Hugging Face language models.
-   `llama3.py`: A dedicated script for pruning Llama-3 and Llama-3.1 models.
-   `meralion.py` / `smolvlm.py`: Scripts for pruning more complex or custom vision-language models.
-   `post_training.py`: The script used for the post-pruning fine-tuning (recovery) stage.
-   `generate.py`: Provides a Gradio-based web interface for generating text with pre-trained, pruned, or fine-tuned models.
-   `test_speedup.py`: A utility to calculate and compare the MACs (Multiply-Accumulate operations) and parameter counts of models.
-   `LLMPruner/pruner/hf_llama_pruner.py`: Contains the core implementation of the pruning functions and importance measures for LLaMA-style models.

## Building and Running

### 1. Installation

First, install the required Python dependencies:

```bash
pip install -r requirement.txt
```

### 2. Pruning

The pruning process is initiated via the `hf_prune.py` or `llama3.py` scripts. The command specifies the base model, pruning strategy (block-wise, channel-wise), pruning ratio, and importance criterion.

**Example: Pruning LLaMA-7B**

```bash
python hf_prune.py \
      --base_model decapoda-research/llama-7b-hf \
      --pruning_ratio 0.25 \
      --pruner_type taylor \
      --block_wise \
      --block_mlp_layer_start 4 \
      --block_mlp_layer_end 30 \
      --block_attention_layer_start 4 \
      --block_attention_layer_end 30 \
      --save_ckpt_log_name llama_pruned \
      --save_model
```

This will save the pruned model checkpoint to `prune_log/llama_pruned/pytorch_model.bin`.

### 3. Post-Training (Fine-tuning)

After pruning, you can fine-tune the model to recover performance using LoRA.

**Example: Fine-tuning the pruned LLaMA**

```bash
python post_training.py \
      --prune_model prune_log/llama_pruned/pytorch_model.bin \
      --data_path yahma/alpaca-cleaned \
      --lora_r 8 \
      --num_epochs 2 \
      --learning_rate 1e-4 \
      --batch_size 64 \
      --output_dir tune_log/llama_pruned_tuned
```

### 4. Generation and Evaluation

You can test the models using the Gradio UI or evaluate them for performance.

**Run Gradio UI for a fine-tuned model:**

```bash
python generate.py \
      --model_type tune_prune_LLM \
      --ckpt prune_log/llama_pruned/pytorch_model.bin \
      --lora_ckpt tune_log/llama_pruned_tuned
```

**Test computational complexity:**

```bash
# For the original model
python test_speedup.py --model_type pretrain --base_model decapoda-research/llama-7b-hf

# For the pruned model
python test_speedup.py --model_type pruneLLM --ckpt prune_log/llama_pruned/pytorch_model.bin
```

## Development Conventions

-   The project relies heavily on command-line arguments for configuration.
-   Pruning logic is modular, with different scripts for different model families (`hf_prune.py`, `llama3.py`, `meralion.py`).
-   The core pruning implementation resides within the `LLMPruner/` directory, particularly in `LLMPruner/pruner/`.
-   Fine-tuning is handled by the PEFT library (LoRA).
-   Logging is managed through a custom `LoggerWithDepth` class, with logs and checkpoints saved to the `prune_log/` and `tune_log/` directories.
-   Evaluation is performed using established benchmarks like `lm-evaluation-harness` and perplexity metrics on datasets like `wikitext2` and `ptb`.
