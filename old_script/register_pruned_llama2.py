'''
import os
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
from huggingface_hub import login
#from transformers import AutoModelForCausalLM


# Step 1: Login to Hugging Face
login()  # Paste your Hugging Face token when prompted

# Step 2: Define paths and model ID
model_path = "/home/kaixin/LLMPruner/LLM-Pruner/merged_llama2_pruned_tuned"  # Path to your pruned LLaMA model
model_id = "Mikidokido/LLaMA2-7B-custom"  # Replace with your desired Hugging Face repo name

# Step 3: Register LLaMA config and model for AutoClass (optional but recommended)
LlamaConfig.register_for_auto_class()
LlamaForCausalLM.register_for_auto_class("AutoModelForCausalLM")

# Step 4: Load the model and tokenizer
config = LlamaConfig.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path, config=config, ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Step 5: Set environment variables (disable DeepSpeed if not needed)
os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"

# Step 6: Push the model and tokenizer to Hugging Face Hub
model.push_to_hub(model_id, safe_serialization=True, private=True, trust_remote_code=False)
tokenizer.push_to_hub(model_id)

print(f"✅ Model and tokenizer uploaded to https://huggingface.co/{model_id}")

'''

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Step 1: Login to Hugging Face
login()  # Paste your Hugging Face token when prompted

model_id = "Mikidokido/LLaMA2-7B-custom"  # change as needed

model = AutoModelForCausalLM.from_pretrained("./llama2-7b-padded-hf")
tokenizer = AutoTokenizer.from_pretrained("./llama2-7b-padded-hf")

model.push_to_hub(model_id, private=True)
tokenizer.push_to_hub(model_id)
