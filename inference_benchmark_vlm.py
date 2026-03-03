import os
import sys
import argparse
import torch
import time
from transformers import AutoModelForVision2Seq, AutoProcessor, GPTQConfig
from optimum.gptq import GPTQQuantizer, load_quantized_model
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.peft import PeftModel
from transformers.image_utils import load_image

@torch.no_grad()
def measure_ttft(model, inputs):
    """
    Measure the time-to-first-token (TTFT) for the model.
    """
    start_time = time.time()
    
    # Generate the first token
    logits = model(
        **inputs, max_new_tokens=1, do_sample=False,
    ).logits
    
    first_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    
    end_time = time.time()
    ttft = end_time - start_time
    prefill_speed = len(inputs['input_ids'][0]) / ttft
    return prefill_speed, first_token

@torch.no_grad()
def measure_decode_speed(model, processor, inputs, max_length=30):
    """
    Measure the decode speed for generating the rest of the tokens.
    """
    # Measure TTFT

    prefill_speed, first_token = measure_ttft(model, inputs)
    print(f"Prefill speed: {prefill_speed} tokens/second")

    # Prepare for generating the rest of the tokens
    # decode_time = 0
    inputs['input_ids'] = torch.cat([inputs['input_ids'], first_token], dim=1)
    start_time = time.time()

    # for _ in range(max_length - 1):
    #     begin = time.time()
    #     logits = model(**inputs, max_new_tokens=1, do_sample=False).logits
    #     decode_time += time.time() - begin

    #     generated_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    #     inputs['input_ids'] = torch.cat([inputs['input_ids'], generated_ids], dim=1)

    logits = model(**inputs, max_new_tokens=max_length-1, do_sample=False).logits
    decode_time = time.time() - start_time

    generated_ids = logits.argmax(dim=-1)


    print(processor.batch_decode(inputs['input_ids'], skip_special_tokens=True))

    decode_speed = (max_length - 1) / decode_time
    print(f"Average decode speed: {decode_speed} tokens/second")
    return inputs['input_ids']


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
torch_version = int(torch.__version__.split('.')[1])

def load_model(args):
    if args.model_type == 'pretrain':
        processor = AutoProcessor.from_pretrained(args.ckpt)
        model = AutoModelForVision2Seq.from_pretrained(
            args.ckpt,
            # low_cpu_mem_usage=True if torch_version >=9 else False
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device)
        description = "Model Type: {}\n Base Model: {}".format(args.model_type, args.ckpt)
    else:
        raise NotImplementedError("Model type {} is not implemented".format(args.model_type))
    
    if args.lora_ckpt:
        print("Loading LORA checkpoint from {}".format(args.lora_ckpt))
        model.load_adapter(args.lora_ckpt, adapter_name="lora", device_map="auto")
    
    print(description)
    
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2

    model.eval()

    return processor, model

@torch.no_grad()
def benchmark(processor, model, inputs, max_length=30):
    generated_ids = measure_decode_speed(model, processor, inputs, max_length=max_length)
    output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Generated text: {output}")


def main(args):
    processor, model = load_model(args)
    print(model)
    # print(model.config)

    image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
    image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
    image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")


    # Create inputs
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, 
                        {"type": "text", "text": "In this image, we can see the city of New York, and more specifically the Statue of Liberty."},
                        {"type": "image"}, 
                        {"type": "text", "text": "What can we see in this image?"},
                        ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "In which city is that bridge located?"},
            ]
        }
    ]

    prompts = [processor.apply_chat_template([message], add_generation_prompt=True) for message in messages]
    images = [[image1, image2], [image3]]
    inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(model.device)

    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True, tokenizer=tokenizer)

    print(generated_texts[0])
    print(generated_texts[1])


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


        else:
            model.hf_device_map = getattr(model, "hf_device_map", {0: device})
            quantized_model = quantizer.quantize_model(model, processor)

            quantizer.save(model,quantized_folder)

        model = quantized_model

    benchmark(processor, model, inputs=inputs, max_length=args.max_seq_len)

    print("GPU Allocated memory: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

    # ppl = PPLMetric(model, processor.tokenizer, ['c4'], args.max_seq_len, device=device) #
    # print("PPL: {}".format(ppl))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chat with Tuned Pruned LLaMA Model')

    # parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--ckpt', type=str, help='Path to the pruned model checkpoint')
    parser.add_argument('--lora_ckpt', type=str, help='Path to the LORA checkpoint')
    parser.add_argument('--tokenizer_ckpt', type=str, help='Path to the tokenizer checkpoint', default="Qwen/Qwen2.5-3B")

    parser.add_argument('--model_type', type=str, default='pretrain', help = 'choose from ')
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--eval_device', type=str, default='cuda')


    parser.add_argument('--int8', action="store_true", help="Use int8 precision for the model")
    parser.add_argument('--int4', action="store_true", help="Use int4 precision for the model")

    parser.add_argument('--push-to-hub', type=str, default=None, help='Push the pruned model to Hugging Face Hub')


    args = parser.parse_args()

    assert not (args.int8 and args.int4), "Cannot use both int8 and int4 at the same time"
    main(args)
