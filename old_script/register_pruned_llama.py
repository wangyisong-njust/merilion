import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from safetensors.torch import load_file
import os

# Paths
pruned_model_path = "/home/kaixin/LLMPruner/LLM-Pruner/merged_llama2_pruned_tuned"
output_path = "/home/kaixin/LLMPruner/LLM-Pruner/llama2-7b-padded-hf"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(pruned_model_path)
print(type(tokenizer))  # Should output <class 'transformers.models.auto.tokenization_auto.AutoTokenizer'>

# ✅ Load all safetensors shards (NOT torch.load)
state_dict = {}
for file in os.listdir(pruned_model_path):
    if file.endswith(".safetensors"):
        shard_path = os.path.join(pruned_model_path, file)
        print(f"Loading shard: {shard_path}")
        state_dict.update(load_file(shard_path))

# Load full-size LLaMA2 config (assumes 7B, adjust if different)
config = LlamaConfig.from_pretrained(pruned_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_config(config)

# Save in Hugging Face-compatible sharded safetensors format
model.save_pretrained(output_path, safe_serialization=True)
tokenizer.save_pretrained(output_path)

print(f"✅ Saved padded Hugging Face model to: {output_path}")
