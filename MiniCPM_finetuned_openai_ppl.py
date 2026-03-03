import random
import argparse
from typing import Tuple

import os
import inspect
import torch
import torch.nn as nn 
import numpy as np
from LLMPruner.peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from LLMPruner.models.hf_MiniCPM.modeling_minicpm import MiniCPMForCausalLM
from LLMPruner.evaluator.ppl import PPLMetric, PPLMetricOpenAI
from LLMPruner.templates.prompts import prompts

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    set_random_seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True) 
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True, truncation=True, max_length=model.config.max_length)
    model.to(args.device)
    model = PeftModel.from_pretrained(
        model,
        args.lora_ckpt
    )

    model.merge_and_unload()

    model = model.get_base_model()

    model.config.pad_token_id = tokenizer.pad_token_id = 0 
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    
    base_model = MiniCPMForCausalLM(model.config)
    base_model.load_state_dict(model.state_dict())       
    model = base_model

    if args.eval_device != "cpu":
        model.half()
    model.to(args.eval_device)
    
    model.eval()

    ppl = PPLMetricOpenAI(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.eval_device)
    print("PPL OpenAI: {}".format(ppl))

    ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.eval_device)
    print("PPL: {}".format(ppl))
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--ckpt', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--lora_ckpt', type=str, default="./MiniCPM-checkpoints/MiniCPM-2B-128k-pruned-bl-0.3-taylor-finetuned")

    # argument for generation
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')

    # general argument
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')

    parser.add_argument('--seed', type=int, default=42, help='seed')

    #ddp
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
