from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from transformers import AutoTokenizer, pipeline
from datasets import load_dataset
import torch
import argparse

parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

# Model Type&Path
parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
parser.add_argument('--prune_model', type=str, help='prune model name')
parser.add_argument('--data_path', type=str, default="yahma/alpaca-cleaned", help='data path')
parser.add_argument('--cache_dataset', action="store_true", default=False)
parser.add_argument('--extra_val_dataset', type=str, default=None, help='validation datasets. Split with ","')
parser.add_argument('--output_dir', type=str, default="./lora-alpaca", help='output directory')

# Training Hyperparameters
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')
parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
parser.add_argument('--cutoff_len', type=int, default=256, help='cutoff length')
parser.add_argument('--val_set_size', type=int, default=2000, help='validation set size')
parser.add_argument('--prompt_template_name', type=str, default="alpaca", help="The prompt template to use, will default to alpaca.")
parser.add_argument('--no_instruction', action='store_true', default=False, help="Whether to use the instruction template or not.")

# Lora Configuration
parser.add_argument('--lora_r', type=int, default=8, help='lora r')
parser.add_argument('--lora_alpha', type=int, default=16, help='lora alpha')
parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
parser.add_argument('--lora_target_modules', type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj", help='lora target modules')

# llm hyperparameters
parser.add_argument('--train_on_inputs', default=False, action="store_true", help='Train on inputs. If False, masks out inputs in loss')
parser.add_argument('--add_eos_token', default=False, action="store_true")
parser.add_argument('--group_by_length', default=False, action="store_true", help="faster, but produces an odd training loss curve")

# wandb params
parser.add_argument('--wandb_project', type=str, default="")
parser.add_argument('--resume_from_checkpoint', type=str, help="either training checkpoint or final adapter")

#ddp
parser.add_argument('--local_rank', type=int, default=-1)

# PPL
parser.add_argument('--max_seq_len', type=int, default=2048)

args = parser.parse_args()

# 1. Configuration
config = PPOConfig(
    model_name=args.base_model,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    log_with="wandb",
)

# 2. Load policy model with value head
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_model = create_reference_model(model)  # Used to calculate KL penalty

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# 3. Reward function (placeholder: sentiment analysis here)
reward_pipe = pipeline("sentiment-analysis")

def compute_reward(samples):
    texts = [s["generated_text"] for s in samples]
    results = reward_pipe(texts)
    rewards = [r["score"] if r["label"] == "POSITIVE" else 1 - r["score"] for r in results]
    return rewards

# 4. Prompt dataset
dataset = load_dataset("imdb", split="train[:1%]")  # just a small subset for demo
prompts = [f"Review: {x['text'][:100]}" for x in dataset]

# 5. PPO Trainer
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)

# 6. RLHF loop
for epoch in range(3):  # Few training steps for demo
    for i in range(0, len(prompts), config.batch_size):
        batch_prompts = prompts[i:i + config.batch_size]
        query_tensors = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).input_ids

        # Generate responses
        response_tensors = []
        for q in query_tensors:
            output = model.generate(q.unsqueeze(0), max_new_tokens=50, pad_token_id=tokenizer.eos_token)
            response_tensors.append(output[0][len(q):])

        # Decode and compute rewards
        samples = [{
            "prompt": tokenizer.decode(q, skip_special_tokens=True),
            "generated_text": tokenizer.decode(r, skip_special_tokens=True)
        } for q, r in zip(query_tensors, response_tensors)]
        
        rewards = compute_reward(samples)

        # PPO step
        ppo_trainer.step(query_tensors, response_tensors, torch.tensor(rewards))
