import os
import sys
import argparse
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig
from LLMPruner.peft import PeftModel
from optimum.gptq import GPTQQuantizer, load_quantized_model
from LLMPruner.evaluator.ppl import PPLMetric



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
    
    hidden_states = outputs[0]
    logits = model_for_causallm.lm_head(hidden_states)
    
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
        hidden_states = outputs[0]
        logits = model_for_causallm.lm_head(hidden_states)
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

        if args.int8 or args.int4:
            print("Quantizing model to {}-bit precision using AWQ".format(8 if args.int8 else 4))
            quant_config = {
                "zero_point": True, 
                "q_group_size": 128, 
                "w_bit": 8 if args.int8 else 4,
                "version": "GEMM"
            }

            from awq import AutoAWQForCausalLM
            model = AutoAWQForCausalLM.from_pretrained(
                args.ckpt,
            )

            model.quantize(tokenizer, quant_config=quant_config)

            if args.push_to_hub:
                print(f"Saving quantized model to {args.push_to_hub}")
                model.save_pretrained(args.push_to_hub)
                tokenizer.save_pretrained(args.push_to_hub)
        else:
            
            model = AutoModelForCausalLM.from_pretrained(
                args.ckpt,
                torch_dtype="auto",
                device_map="auto"
            )
        
        description = "Model Type: {}\n Base Model: {}".format(args.model_type, args.ckpt)
    elif args.model_type == 'pruneLLM':
        if args.int8 or args.int4:
            print("Not supported")
            exit(1)

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
    
    # if device == "cuda" and not args.int8 and not args.int4:
    #     model.half()
    # model = model.to(device)

    # unwind
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
        
    benchmark(tokenizer, model)

    print("GPU Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

    ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=device)
    print("PPL: {}".format(ppl))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chat with Tuned Pruned LLaMA Model')

    # parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--ckpt', type=str, help='Path to the pruned model checkpoint')
    parser.add_argument('--lora_ckpt', type=str, help='Path to the LORA checkpoint')

    parser.add_argument('--model_type', type=str, required=True, help = 'choose from ')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--eval_device', type=str, default='cuda')


    parser.add_argument('--int8', action="store_true", help="Use int8 precision for the model")
    parser.add_argument('--int4', action="store_true", help="Use int4 precision for the model")

    parser.add_argument('--push-to-hub', type=str, default=None, help='Push the pruned model to Hugging Face Hub')


    args = parser.parse_args()

    assert not (args.int8 and args.int4), "Cannot use both int8 and int4 at the same time"
    main(args)
