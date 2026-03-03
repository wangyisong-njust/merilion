'''
Refer to
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
'''

import os
import sys
import argparse
from typing import List
from pathlib import Path

import torch
import transformers
from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from LLMPruner.peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from LLMPruner.utils.prompter import Prompter, ZeroPrompter
from LLMPruner.datasets.ppl_dataset import get_loaders
from LLMPruner.peft import PeftModel
from LLMPruner.evaluator.ppl import PPLMetric

device = "cuda" if torch.cuda.is_available() else "cpu"

def measure_decode_speed(model, prompt, tokenizer, attention_mask=None, max_length=30):
    """
    Measure the decode speed for generating the rest of the tokens.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_length,
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output

def merge_base_and_lora_weight(args):
    pruned_dict = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']


    model = PeftModel.from_pretrained(
        model,
        args.lora_ckpt,
        torch_dtype="bfloat16",
    )

    model.merge_and_unload()

    if args.lora_ckpt_2 is not None:
        model = PeftModel.from_pretrained(
            model.base_model.model,
            args.lora_ckpt_2,
            torch_dtype="bfloat16",
        )

        model.merge_and_unload()

    if device == 'cuda':
        model.half()

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"



    ori_model = torch.load(args.ckpt, map_location='cpu', weights_only=False)['model']
    
    ori_model.load_state_dict(model.base_model.model.state_dict())

    model = ori_model.to(device)


    prompts= ["Hey, are you conscious? Can you talk to me?",
              "将下⾯的句⼦翻译成中⽂：It's a beautiful day to learn something new.",
              "描述优秀的领导者应具备的五个特质，并解释每个特质为什么重要",
              "计算8乘以12",
              "近年来，随着技术的快速发展和全球化的深入推进，数字经济已成为推动世界经济增长的新引擎。数字经济不仅改变了人们的生活方式，促进了信息和资源的快速流通，还重塑了传统行业的业务模式和竞争格局。尽管数字经济的发展为全球经济增长提供了新的动能，但同时也带来了数据安全、隐私保护、数字鸿沟和市场垄断等一系列挑战。考虑到这些背景，请详细分析数字经济在促进世界经济增长方面的作用，包括但不限于数字经济对提高生产效率、创造就业机会和促进可持续发展的贡献。同时，探讨如何应对数字经济发展过程中出现的挑战，具体包括如何保护个人数据安全和隐私、缩小数字鸿沟以确保数字经济的包容性和公平性，以及如何制定有效政策以避免市场垄断情况的出现，最终实现数字经济的健康和可持续发展。"
            #   "What is the capital of France?",
            #   "Explain the theory of relativity in simple terms.",
            #   "Write a short story about a robot learning to love."
    ]


    for prompt in prompts:

        # messages = [
        #     {"role": "system", "content": "You are a helpful assistant."},
        #     {"role": "user", "content": prompt}
        # ]
        # text = tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )

        # alpaca template 
        prompter = Prompter(template_name="alpaca")
        text = prompter.generate_prompt(instruction=prompt, input=None, label=None)

        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            # do_sample=True,
            # temperature=0.7,
            # top_k=50,
            # top_p=0.95
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)

    # PPLMetric(model, tokenizer, ['wikitext2', "ptb"], 2048, device="cuda")


    if args.save_path is not None and not os.path.exists(os.path.split(args.save_path)[0]):
        os.mkdir(os.path.split(args.save_path)[0])
    
    if args.save_path is not None:
        # torch.save(model.base_model.model, args.save_path)
        print(model)
        model.base_model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    # Model Type&Path
    # parser.add_argument('--base_model', type=str, default="/home/kaixin/programs/MiniCPM-checkpoints/MiniCPM-2B-128k", help='base model name')
    parser.add_argument('--ckpt', type=str, default='./MiniCPM-checkpoints/MiniCPM-2B-128k-pruned-bl-0.3-taylor', help='pruned model path', required=True)
    parser.add_argument('--lora_ckpt', type=str, default=None, required=True)
    parser.add_argument('--lora_ckpt_2', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None, required=True)

    args = parser.parse_args()

    merge_base_and_lora_weight(args)


    
