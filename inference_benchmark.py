import os
import sys
import argparse
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from optimum.gptq import GPTQQuantizer, load_quantized_model
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.peft import PeftModel
from LLMPruner.utils.prompter import Prompter, ZeroPrompter


def measure_ttft(model_for_causallm, input_ids, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
    """
    Measure the time-to-first-token (TTFT) for the model.
    """
    start_time = time.time()
    
    # Generate the first token
    outputs = model_for_causallm.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    
    logits = outputs[0]
    
    # Get the first token
    first_token_id = torch.argmax(logits[:, -1, :], dim=-1)
    
    end_time = time.time()
    ttft = end_time - start_time
    prefill_speed = len(input_ids[0]) / ttft
    return prefill_speed, first_token_id

def measure_decode_speed(model_for_causallm, input_ids, attention_mask=None, max_length=30):
    """
    Measure the decode speed for generating the rest of the tokens.
    """
    # Measure TTFT
    prefill_speed, first_token_id = measure_ttft(model_for_causallm, input_ids, attention_mask)
    print(f"Prefill speed: {prefill_speed} tokens/second")

    # Prepare for generating the rest of the tokens
    input_ids = torch.cat([input_ids, first_token_id.unsqueeze(-1)], dim=-1)
    
    start_time = time.time()

    for _ in range(max_length - 1):
        outputs = model_for_causallm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        logits = outputs[0]
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
    
    decode_time = time.time() - start_time

    decode_speed = (max_length - 1) / decode_time
    print(f"Average decode speed: {decode_speed} tokens/second")
    return input_ids


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
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device)
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
        description = "Model Type: {}\n Pruned Model: {}\n LORA ckpt: {}".format(args.model_type, args.ckpt, args.lora_ckpt)
    else:
        raise NotImplementedError
    
    print(description)
    

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()

    return tokenizer, model

@torch.no_grad()
def benchmark(tokenizer, model, prompt="Hey, are you conscious? Can you talk to me?", max_length=30):

    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    
    generated_ids = measure_decode_speed(model, inputs["input_ids"], max_length=max_length)
    print(f"Generated text: {tokenizer.decode(generated_ids[0], skip_special_tokens=True)}")

    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return output

def main(args):
    tokenizer, model = load_model(args)
    print(model)
    print(model.config)

    # prompts= ["Hey, are you conscious? Can you talk to me?",
    #           "将下⾯的句⼦翻译成中⽂：It's a beautiful day to learn something new.",
    #           "描述优秀的领导者应具备的五个特质，并解释每个特质为什么重要",
    #           "计算8乘以12"
    # ]

    # for prompt in prompts:
    #     # print(f"Prompt: {prompt}")

    #     prompter = Prompter(template_name="alpaca")
    #     text = prompter.generate_prompt(instruction=prompt, input=None, label=None)
       
    #     model_inputs = tokenizer([text], return_tensors="pt").to(device)

    #     generated_ids = model.generate(
    #         model_inputs.input_ids,
    #         max_new_tokens=512,
    #         # do_sample=True,
    #         # temperature=0.7,
    #         # top_k=50,
    #         # top_p=0.95
    #     )
        
    #     generated_ids = [
    #         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    #     ]

    #     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    #     print(response)

    if args.int8 or args.int4:
        print("Quantizing model to {}-bit precision using GPTQ".format(8 if args.int8 else 4))


        quantizer = GPTQQuantizer(bits=8 if args.int8 else 4, dataset="c4", 
                                  model_seqlen=args.max_seq_len,
                                  block_name_to_quantize="model.layers",
                                  modules_in_block_to_quantize =[["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"], ["self_attn.o_proj"]],
                                  backend="marlin",
                                  group_size=-1)
        quantized_folder = os.path.split(args.ckpt)[0] + f"quantized_{8 if args.int8 else 4}bit/" 

        if os.path.exists(quantized_folder):
            from accelerate import init_empty_weights

            with init_empty_weights(quantized_folder):
                _, empty_model = load_model(args)
            empty_model.tie_weights()
            quantized_model = load_quantized_model(empty_model, save_folder=quantized_folder, device_map="auto")

            if args.push_to_hub:
                gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
                quantized_model.save_pretrained(args.push_to_hub.split("/")[-1] + f"gptq-int{8 if args.int8 else 4}")
                tokenizer.save_pretrained(args.push_to_hub.split("/")[-1] + f"gptq-int{8 if args.int8 else 4}")

                quant_model = AutoModelForCausalLM.from_pretrained(args.push_to_hub.split("/")[-1] + f"gptq-int{8 if args.int8 else 4}", device_map="auto")
                quant_model.push_to_hub(args.push_to_hub + f"gptq-int{8 if args.int8 else 4}")
                tokenizer.push_to_hub(args.push_to_hub + f"gptq-int{8 if args.int8 else 4}")

        else:
            model.hf_device_map = getattr(model, "hf_device_map", {0: device})
            quantized_model = quantizer.quantize_model(model, tokenizer)

            quantizer.save(model,quantized_folder)

        model = quantized_model

    elif args.push_to_hub:    
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)  

    benchmark(tokenizer, model)

    print("GPU Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

    # ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb', 'alpaca_cn','c4'], args.max_seq_len, device=device) #
    ppl = PPLMetric(model, tokenizer, ['alpaca_cn'], args.max_seq_len, device=device) 
    print("PPL: {}".format(ppl))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chat with Tuned Pruned LLaMA Model')

    # parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--ckpt', type=str, help='Path to the pruned model checkpoint')
    parser.add_argument('--lora_ckpt', type=str, help='Path to the LORA checkpoint')
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
