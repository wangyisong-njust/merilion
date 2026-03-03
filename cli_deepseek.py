import os
import sys
import argparse
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from LLMPruner.peft import PeftModel


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
torch_version = int(torch.__version__.split('.')[1])

def load_model(args):
    if args.model_type == 'pretrain':
        tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt,
            low_cpu_mem_usage=True if torch_version >=9 else False,
            torch_dtype=torch.bfloat16,
        ).to(device)
        description = "Model Type: {}\n Base Model: {}".format(args.model_type, args.ckpt)
    elif args.model_type == 'pruneLLM':
        pruned_dict = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model'].to(device)
        description = "Model Type: {}\n Pruned Model: {}".format(args.model_type, args.ckpt)
    elif args.model_type == 'tune_prune_LLM':
        pruned_dict = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
        model = PeftModel.from_pretrained(
            model,
            args.lora_ckpt,
            torch_dtype=torch.float16,
        ).to(device)
        model.merge_and_unload()

        
        description = "Model Type: {}\n Pruned Model: {}\n LORA ckpt: {}".format(args.model_type, args.ckpt, args.lora_ckpt)
    else:
        raise NotImplementedError
    
    # if device == "cuda":
    #     model.half()
    #     model = model.cuda()

    # unwind

    print(model)

    if "deepseek" not in args.ckpt:
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
    else:
        model.generation_config = GenerationConfig.from_pretrained(args.ckpt)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    model.eval()

    
    if args.push_to_hub is not None:
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)

    return tokenizer, model

@torch.no_grad()
def chat_with_model(tokenizer, model, prompt, messages=[]):

    # Measure prefill time
    messages += [
        {"role": "user", "content": prompt}
    ]
    input_tensor = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True, 
        return_tensors="pt"
    )

    outputs  = model.generate(
        input_tensor.to(device),
        max_new_tokens=args.max_seq_len,
    )
    
    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    messages += [
        {"role": "system", "content": result}
    ]
    return result

def main(args):
    tokenizer, model = load_model(args)
    print("Model loaded successfully. You can start chatting with the model now.")

    messages = []
    while True:
        input_text = input("You: ")
        if input_text.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break
        response = chat_with_model(tokenizer, model, input_text, messages)
        print("Model: ", response)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chat with Tuned Pruned LLaMA Model')

    parser.add_argument('--ckpt', type=str, required=True, help='Path to the pruned model checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default=None, help='Path to the LORA checkpoint')

    parser.add_argument('--push-to-hub', type=str, default=None, help='Push the pruned model to Hugging Face Hub')

    parser.add_argument('--model_type', type=str, required=True, help = 'choose from ')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--eval_device', type=str, default='cuda')


    args = parser.parse_args()
    main(args)