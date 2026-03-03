import os
import sys
import argparse
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from LLMPruner.peft import PeftModel


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
torch_version = int(torch.__version__.split('.')[1])

def load_model(args):
    if args.model_type == 'pretrain':
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = BenchmarkAutoModelForCausalLM.from_pretrained(
            args.base_model,
            low_cpu_mem_usage=True if torch_version >=9 else False
        )
        description = "Model Type: {}\n Base Model: {}".format(args.model_type, args.base_model)
    elif args.model_type == 'pruneLLM':
        pruned_dict = torch.load(args.ckpt, map_location='cpu')
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
        description = "Model Type: {}\n Pruned Model: {}".format(args.model_type, args.ckpt)
    elif args.model_type == 'tune_prune_LLM':
        pruned_dict = torch.load(args.ckpt, map_location='cpu')
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
        model = PeftModel.from_pretrained(
            model,
            args.lora_ckpt,
            torch_dtype=torch.float16,
        )
        description = "Model Type: {}\n Pruned Model: {}\n LORA ckpt: {}".format(args.model_type, args.ckpt, args.lora_ckpt)
    else:
        raise NotImplementedError
    
    if device == "cuda":
        model.half()
        model = model.cuda()

    # unwind
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()

    return tokenizer, model


def chat_with_model(tokenizer, model, input_text, temperature=0.1, top_p=0.75, top_k=40, max_new_tokens=128):
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    # Measure prefill time
    with torch.no_grad():
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": input_text}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_seq_len,
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return output

def main(args):
    tokenizer, model = load_model(args)
    print("Model loaded successfully. You can start chatting with the model now.")
    while True:
        input_text = input("You: ")
        if input_text.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break
        response = chat_with_model(tokenizer, model, input_text)
        print("Model: ", response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chat with Tuned Pruned LLaMA Model')

    parser.add_argument('--ckpt', type=str, required=True, help='Path to the pruned model checkpoint')
    parser.add_argument('--lora_ckpt', type=str, required=True, help='Path to the LORA checkpoint')

    parser.add_argument('--model_type', type=str, required=True, help = 'choose from ')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--eval_device', type=str, default='cuda')


    args = parser.parse_args()
    main(args)