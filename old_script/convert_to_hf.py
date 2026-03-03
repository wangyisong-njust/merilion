import torch
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
import os

# ==========
# 1. PATHS
# ==========
pruned_model_path = "/home/kaixin/LLMPruner/LLM-Pruner/merged_llama2_pruned_tuned/pytorch_model.bin"
save_dir = "./merged_llama2_pruned_hf"
base_model = "meta-llama/Llama-2-7b-hf"  # Base model to copy tokenizer/config from

# ==========
# 2. LOAD PRUNED STATE_DICT
# ==========
print(f"Loading pruned state_dict from {pruned_model_path}...")
state_dict = torch.load(pruned_model_path, map_location="cpu")

# ==========
# 3. CREATE CONFIG & ADJUST TO PRUNED SIZE
# ==========
print("Creating modified HF config...")
config = LlamaConfig.from_pretrained(base_model)

# ---- Detect pruned hidden_size automatically ----
# Find a representative weight (e.g., first MLP up_proj)
for k, v in state_dict.items():
    if "mlp.up_proj.weight" in k:
        config.intermediate_size = v.shape[0]
        config.hidden_size = v.shape[1]
        break

print(f"Modified config: hidden_size={config.hidden_size}, intermediate_size={config.intermediate_size}")

model = LlamaForCausalLM(config)

# ==========
# 4. FILTER INCOMPATIBLE WEIGHTS
# ==========
filtered_state_dict = {}
skipped = []

for key, weight in state_dict.items():
    if key in model.state_dict() and model.state_dict()[key].shape == weight.shape:
        filtered_state_dict[key] = weight
    else:
        skipped.append((key, weight.shape, model.state_dict().get(key, None).shape if key in model.state_dict() else None))

print(f"✅ Loaded {len(filtered_state_dict)} compatible tensors")
print(f"⚠️ Skipped {len(skipped)} incompatible tensors")

# Load filtered weights
missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
print("Missing keys (initialized randomly):", missing)
print("Unexpected keys (ignored):", unexpected)

# ==========
# 5. SAVE HF FORMAT
# ==========
os.makedirs(save_dir, exist_ok=True)
print(f"Saving HF-compatible pruned model to {save_dir}...")
model.save_pretrained(save_dir, safe_serialization=True)

# Save tokenizer (use base model tokenizer)
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.save_pretrained(save_dir)

print("\n🎉 Conversion complete! You can now push to Hugging Face:")
print(f"    from transformers import AutoModelForCausalLM; model = AutoModelForCausalLM.from_pretrained('{save_dir}')")
