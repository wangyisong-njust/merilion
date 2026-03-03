from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer, load_quantized_model
import torch
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="./MiniCPM-checkpoints/MiniCPM-2B-128k-pruned-bl-0.3-taylor")
parser.add_argument("--output", type=str, default="./MiniCPM-checkpoints/MiniCPM-2B-128k-pruned-bl-0.3-taylor_int4")
parser.add_argument("--bit", type=int, default=4)
args = parser.parse_args()



# Update main export code
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, trust_remote_code=True)

quantizer = GPTQQuantizer(bits=args.bit, dataset="c4", block_name_to_quantize="model.layers", model_seqlen=2048)
quantized_model = quantizer.quantize_model(model, tokenizer)
quantizer.save(model, args.output)
