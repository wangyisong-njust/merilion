import os
import sys
import argparse
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, BitsAndBytesConfig
from optimum.gptq import GPTQQuantizer, load_quantized_model
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.peft import PeftModel
from LLMPruner.utils.prompter import Prompter, ZeroPrompter
import json
import pandas as pd
from datasets import load_dataset
import Levenshtein
import tqdm

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
            # low_cpu_mem_usage=True if torch_version >=9 else False
            torch_dtype=torch.float16,
            trust_remote_code=True, device_map="cuda",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True)
        )#.to(device)
        description = "Model Type: {}\n Base Model: {}".format(args.model_type, args.ckpt)
    elif args.model_type == 'pruneLLM':
        pruned_dict = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        model = pruned_dict.get('model', pruned_dict).to(device)
        if "tokenizer" in pruned_dict:
            tokenizer = pruned_dict['tokenizer']
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_ckpt)
        description = "Model Type: {}\n Pruned Model: {}".format(args.model_type, args.ckpt)
    elif args.model_type == 'tune_prune_LLM':
        pruned_dict = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        model = pruned_dict['model'].to(device)
        if "tokenizer" in pruned_dict:
            tokenizer = pruned_dict['tokenizer']
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_ckpt)
        model = PeftModel.from_pretrained(
            model,
            args.lora_ckpt,
            torch_dtype="auto",
        ).to(device)
        model.merge_and_unload()

        save_path = "./merged_model"
        model.save_pretrained(save_path)
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            save_path,
            # quantization_config=bnb_config,
            device_map="cuda",
            torch_dtype=torch.float16
        )
        description = "Model Type: {}\n Pruned Model: {}\n LORA ckpt: {}".format(args.model_type, args.ckpt, args.lora_ckpt)
    elif args.model_type == 'tune_tune_prune_LLM':
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        pruned_dict = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        model = pruned_dict['model'].to(device)
        if "tokenizer" in pruned_dict:
            tokenizer = pruned_dict['tokenizer']
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_ckpt)
        model = PeftModel.from_pretrained(
            model,
            args.lora_ckpt,
            torch_dtype="auto",
            device_map="cuda"
        )
        model.merge_and_unload()
        save_path = "./merged_model"

        model = PeftModel.from_pretrained(
            model,
            args.lora_ckpt2,
            torch_dtype="auto",
            device_map="cuda"
        )
        model.merge_and_unload().cuda()

        # model.save_pretrained(save_path)
        # model = AutoModelForCausalLM.from_pretrained(
        #     save_path,
        #     quantization_config=bnb_config,
        #     device_map="cuda"
        # )

        description = "Model Type: {}\n Pruned Model: {}\n LORA ckpt: {} & {}".format(args.model_type, args.ckpt, args.lora_ckpt, args.lora_ckpt2)
    else:
        raise NotImplementedError
    
    print(description)
    

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()

    return tokenizer, model

@torch.no_grad()
def main(args):
    tokenizer, model = load_model(args)
    print(model)

    # test_data = [{"input": ent["instruction"] + "\n" + ent["instances"][0]["input"], "output": ent["instances"][0]["output"]} for ent in json.load(open("/home/kaixin/programs/LLM-Pruner/zh_seed_tasks.json", "r"))] 
    valdata = load_dataset(
        'yahma/alpaca-cleaned', split='train'
    ).select(range(200))

    

    # test_data = [{"input": ent["instruction"] + "\n" + ent["input"], "output": ent["output"]} for ent in valdata] 
    test_data = [{"input": "Write a function to convert a given string to uppercase unless it starts with the string \"Rejected\"", "output": "asdf"}]
    df = pd.DataFrame(columns=["loss", "input", "output", "label"])


    # prompter = Prompter("alpaca")
    # for data in test_data:
    #     inputs = tokenizer([data["input"]], return_tensors="pt").to(device)
    #     # output = model(inputs)
    #     # lm_logits = output.logits
    
    #     # shift_logits = lm_logits[:, :-1, :].contiguous()
    #     # shift_labels = inputs[:, 1:].contiguous()
        
    #     # loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    #     # loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
    #     outputs = model.generate(
    #         **inputs,
    #         use_cache=False,
    #     )

    #     outputs = outputs[:, inputs.input_ids.shape[1]:]
    #     output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     score = Levenshtein.distance(output_text.lower(), data["output"].lower())
    #     df.loc[len(df)] = [score, data["input"].replace("\n", "/n"), output_text.replace("\n", "/n"), data["output"].replace("\n", "/n")]

    batch_size = 1  # Define the batch size for batched inference
    for i in tqdm.tqdm(range(0, len(test_data), batch_size)):
        batch_data = test_data[i:i + batch_size]
        inputs = tokenizer([tokenizer.apply_chat_template([
                # {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": data["input"]}
            ], tokenize=False, add_generation_prompt=True) for data in batch_data], return_tensors="pt", padding=True, truncation=True).to(device)
        # inputs = tokenizer([prompter.generate_prompt(instruction=data["input"]) for data in batch_data], return_tensors="pt", padding=True, truncation=True).to(device)
        # inputs = tokenizer([data["input"] for data in batch_data], return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(
            **inputs,
            use_cache=False,
            max_new_tokens=args.max_seq_len,
        )

        for j, output in enumerate(outputs):
            output = output[inputs.input_ids.shape[1]:]
            output_text = tokenizer.decode(output, skip_special_tokens=True)
            score = Levenshtein.distance(output_text.lower(), batch_data[j]["output"].lower()) / len(batch_data[j]["output"])
            df.loc[len(df)] = [score, batch_data[j]["input"].replace("\n", "/n"), output_text.replace("\n", "/n"), batch_data[j]["output"].replace("\n", "/n")]

        

        # df = df.sort_values(by="loss", ascending=True).reset_index(drop=True)
        df.to_csv(args.testcase_csv, index=False, sep="\t")
        print(f"save to {args.testcase_csv}")
        
    # ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb', 'alpaca_cn','c4'], args.max_seq_len, device=device) #
    # ppl = PPLMetric(model, tokenizer, ['alpaca_cn'], args.max_seq_len, device=device) 
    # print("PPL: {}".format(ppl))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chat with Tuned Pruned LLaMA Model')

    # parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--ckpt', type=str, help='Path to the pruned model checkpoint')
    parser.add_argument('--lora_ckpt', type=str, help='Path to the LORA checkpoint')
    parser.add_argument('--lora_ckpt2', type=str, help='Path to the LORA checkpoint')
    parser.add_argument('--tokenizer_ckpt', type=str, help='Path to the tokenizer checkpoint', default="Qwen/Qwen2.5-3B")

    parser.add_argument('--model_type', type=str, required=True, help = 'choose from ')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--eval_device', type=str, default='cuda')


    parser.add_argument('--int8', action="store_true", help="Use int8 precision for the model")
    parser.add_argument('--int4', action="store_true", help="Use int4 precision for the model")

    parser.add_argument('--testcase_csv', default="test.csv", type=str, help='Path to the test case CSV file')

    parser.add_argument('--push-to-hub', type=str, default=None, help='Push the pruned model to Hugging Face Hub')


    args = parser.parse_args()

    assert not (args.int8 and args.int4), "Cannot use both int8 and int4 at the same time"
    main(args)
