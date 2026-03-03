import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from safetensors.torch import load_file
import os

# Paths
pruned_model_path = "/home/kaixin/LLMPruner/LLM-Pruner/merged_llama2_pruned_tuned"
output_path = "/home/kaixin/LLMPruner/LLM-Pruner/llama2-7b-padded-hf"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(pruned_model_path)

# ✅ Load all safetensors shards
state_dict = {}
for file in os.listdir(pruned_model_path):
    if file.endswith(".safetensors"):
        shard_path = os.path.join(pruned_model_path, file)
        print(f"Loading shard: {shard_path}")
        state_dict.update(load_file(shard_path))

# Load full-size LLaMA2 config
config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_config(config)

# Pad function
def pad_to_shape(weight, target_shape):
    if weight.shape == target_shape:
        return weight
    print(f"Patching: {weight.shape} -> {target_shape}")
    new_weight = torch.zeros(target_shape, dtype=weight.dtype)
    new_weight[:weight.shape[0], :weight.shape[1]] = weight
    return new_weight

# Patch all weights
model_sd = model.state_dict()
new_sd = {}
for k, v in model_sd.items():
    if k in state_dict:
        w = state_dict[k]
        if w.shape != v.shape:
            w = pad_to_shape(w, v.shape)
        new_sd[k] = w
    else:
        print(f"Missing {k}, keeping default zeros")
        new_sd[k] = v

# Load patched weights
model.load_state_dict(new_sd, strict=False)

# Save padded HF model
model.save_pretrained(output_path, safe_serialization=True)
tokenizer.save_pretrained(output_path)

print(f"✅ Saved padded Hugging Face model to: {output_path}")
