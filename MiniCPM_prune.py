import os
import gc
import sys
import time
import json
import copy
import random
import argparse
from typing import Tuple

import torch
import torch.nn as nn 
import numpy as np
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
# from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP
from LLMPruner.models.hf_MiniCPM.modeling_minicpm import MiniCPMForCausalLM, MiniCPMRMSNorm, MiniCPMMLP

import LLMPruner.torch_pruning as tp 
from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.templates.prompts import prompts

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    pr_per_layer = []
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()
            print(f"  - {float((W==0).sum().item())/W.numel():.6f} {name}")

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")
        pr_per_layer.append([float(sub_count)/sub_params])

    model.config.use_cache = use_cache 
    return float(count)/total_params, pr_per_layer
    
def main(args):
    set_random_seed(args.seed)

    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name), 
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )

    # if "MiniCPM" in args.base_model: MiniCPMForCausalLM AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True) 
    # sdpa or flash_attention_2, no eager
    # torch_dtype=torch.bfloat16
    # check_sparsity(model)
    print(model)
    # input(model.config.max_length)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, truncation=True, max_length=model.config.max_length)

    # if "MiniCPM" in args.base_model:
    # if args.device != "cpu":
    #     model.half()
    model.to(args.device)

    if args.test_before_train:
        logger.log("\n==================Generation Results before Pruning================\n")
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(args.device)

                generation_output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_k=50,
                    max_length=args.max_seq_len,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
                
                result = tokenizer.decode(generation_output[0])
                logger.log(result)
    
        ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.device)
        logger.log("PPL before pruning: {}".format(ppl))

    pruner_type = args.pruner_type.lower()
    assert pruner_type in ['random', 'l2', 'l1', 'taylor']

    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    forward_prompts = torch.tensor([
        [    1,   306,  4658,   278,  6593,   310,  2834,   338],
        [    1,  3439, 17632,  1925, 29892,   278,  6368,   310],
    ]).to(args.device) # Only for building the dependency graph. Any input will be fine since the computation result are not taken into consideration.

    if pruner_type == 'random':
        imp = tp.importance.RandomImportance()
    elif pruner_type == 'l1':
        imp = llama_pruner.MagnitudeImportance(p=1)
    elif pruner_type == 'l2':
        imp = llama_pruner.MagnitudeImportance(p=2)
    elif pruner_type == 'taylor':
        imp = llama_pruner.TaylorImportance(group_reduction=args.grouping_strategy, taylor=args.taylor)
    else:
        raise NotImplementedError

    logger.log("Use {} pruner...".format(pruner_type))
    
    if args.block_wise:
        # get RMSNorm type for MiniCPM
        i=0
        for m in model.modules():
            if i == 16:
                MiniCPMRMSNorm = type(m)
                print(MiniCPMRMSNorm)
                break
            i += 1
        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio, 
            "ignored_layers":[],
            "channel_groups": {
            },
            "consecutive_groups": {
                layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers
            },
            "customized_pruners": {
                MiniCPMRMSNorm: llama_pruner.hf_rmsnorm_pruner,
            },
            "root_module_types": None, 
            "root_instances": [model.model.layers[i].self_attn.q_proj for i in range(args.block_attention_layer_start, args.block_attention_layer_end)] +
                              [model.model.layers[i].mlp.gate_proj for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end)]
        }
        logger.log("Pruning Attention Layer = {}".format(list(range(args.block_attention_layer_start, args.block_attention_layer_end))))
        logger.log("Pruning MLP Layer = {}".format(list(range(args.block_mlp_layer_start, args.block_mlp_layer_end))))

        pruner = tp.pruner.MetaPruner(
            model,
            forward_prompts,
            **kwargs
        )
        model.zero_grad()

        logger.log("Start Pruning")
        for i in range(args.iterative_steps):

            if pruner_type in ['taylor']:
                example_prompts = get_examples('bookcorpus', tokenizer, args.num_examples, seq_len = 64).to(args.device)
                logger.log("Start Backwarding in iterative steps = {}...".format(i))
                if args.taylor in ['param_mix', 'param_second']:
                    for j in range(args.num_examples):
                        batch_input = example_prompts[j].unsqueeze(0)
                        loss = model(batch_input, labels=batch_input).loss
                        logger.log("Loss = {}".format(loss))
                        loss.backward()

                        for module_param in model.parameters():
                            module_param.grad = module_param.grad * module_param.grad / args.num_examples
                            if hasattr(module_param, 'acc_grad'):
                                module_param.acc_grad += module_param.grad
                            else:
                                module_param.acc_grad = copy.deepcopy(module_param.grad)
                        model.zero_grad()
                        del loss.grad
                    
                loss = model(example_prompts, labels=example_prompts).loss
                logger.log("Loss = {}".format(loss))
                loss.backward()

            pruner.step()

            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))
        
            # modify inferece-related attributes
            for layer in model.model.layers:
                layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
                # for MiniCPM
                layer.self_attn.num_key_value_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim

        # Clean the gradient in the model
        model.zero_grad()
        for name, module in model.named_parameters():
            if 'weight' in name:
                module.grad = None

        del pruner

    elif args.channel_wise:
        # get RMSNorm type for MiniCPM
        i=0
        for m in model.modules():
            if i == 16:
                MiniCPMRMSNorm = type(m)
                print(MiniCPMRMSNorm)
                break
            i += 1
        # adjust hidden size to be a multiple of attention heads
        adjusted_pruning_ratio = (model.config.hidden_size * args.pruning_ratio // model.config.num_attention_heads + 1) \
                                * model.config.num_attention_heads / model.config.hidden_size
        print("adjusted pruning ratio:", adjusted_pruning_ratio)
        # print(model.config.hidden_size * adjusted_pruning_ratio)
        # input()
        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": adjusted_pruning_ratio, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            "ignored_layers":[],
            #"round_to": model.config.num_attention_heads * 2,
            "channel_groups": {
                #layer.self_attn: layer.self_attn.num_heads for layer in model.model.layers
            },
            "customized_pruners": {
                MiniCPMRMSNorm: llama_pruner.hf_rmsnorm_pruner, 
                # nn.MultiheadAttention: llama_pruner.hf_attention_pruner, # ori commented
                # MiniCPMMLP: llama_pruner.hf_linear_pruner # ori none
            },
            "root_module_types": [MiniCPMRMSNorm, 
                                #   nn.MultiheadAttention,
                                #   MiniCPMMLP # ori none
                                  ],
            # "root_instances": [layer.self_attn for layer in model.model.layers]
        }

        pruner = tp.pruner.MetaPruner(
            model,
            forward_prompts,
            **kwargs
        )
        model.zero_grad()
        
        logger.log("Start Pruning")
        for i in range(args.iterative_steps):

            if pruner_type in ['taylor']:
                example_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len = 64).to(args.device)
                logger.log("Start Backwarding in iterative steps = {}...".format(i))
                loss = model(example_prompts, labels=example_prompts).loss
                logger.log("Loss = {}".format(loss))
                loss.backward()

               

            pruner.step()
            
            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))

        model.zero_grad()
        
        # modify inferece-related attributes
        model.config.hidden_size = model.model.embed_tokens.weight.shape[1]
        #try
        # for layer in model.model.layers:
        #         layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
        #         # for MiniCPM
        #         layer.self_attn.num_key_value_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
        
        # Manually L1 pruning of Attention head channels\
        @torch.no_grad()
        def l1_norm_per_input_channel(W):
            return W.abs().mean(dim=-1)
        
        @torch.no_grad()
        def l1_norm_per_output_channel(W):
            return W.abs().mean(dim=-2)
        
        @torch.no_grad()
        def taylor_score_per_input_channel(W):
            return (W.data * W.grad).abs().mean(dim=-1)
        
        @torch.no_grad()
        def taylor_score_per_output_channel(W):
            return (W.data * W.grad).abs().mean(dim=-2)


        for layer in model.model.layers:
            
            layer.self_attn.q_proj.out_features = model.config.hidden_size
            layer.self_attn.k_proj.out_features = model.config.hidden_size
            layer.self_attn.v_proj.out_features = model.config.hidden_size
            layer.self_attn.o_proj.in_features = model.config.hidden_size

            l1_norm = 0.
            l1_norm += l1_norm_per_output_channel(layer.self_attn.q_proj.weight)
            l1_norm += l1_norm_per_output_channel(layer.self_attn.k_proj.weight)
            l1_norm += l1_norm_per_output_channel(layer.self_attn.v_proj.weight)
            l1_norm += l1_norm_per_input_channel(layer.self_attn.o_proj.weight)

            keep_inds = torch.argsort(l1_norm, descending=True)[:model.config.hidden_size]
            layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[keep_inds]
            layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[keep_inds]
            layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[keep_inds]
            layer.self_attn.o_proj.weight.data = layer.self_attn.o_proj.weight.data[:, keep_inds]
            # Assume no bias
            
            layer.self_attn.head_dim = model.config.hidden_size // model.config.num_attention_heads
            layer.self_attn._init_rope()

        # Clean the gradient in the model
        model.zero_grad()
        for name, module in model.named_parameters():
            if 'weight' in name:
                module.grad = None

        

        del pruner
            
    elif args.layer_wise:
        model.model.layers = model.model.layers[:args.layer]
        after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    else:
        raise NotImplementedError
    logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))
    
    gc.collect()
    torch.cuda.empty_cache()

    # if args.save_model_path:
    #     # model.half()
    #     torch.save({
    #         'model': model, 
    #         'tokenizer': tokenizer,
    #     }, logger.best_checkpoint_path)
    
    if args.save_model_path:
        model.save_pretrained(args.save_model_path)
        tokenizer.save_pretrained(args.save_model_path)
        print("Pruned MiniCPM model has been saved to", args.save_model_path)
    
    print(model)

    model.config.pad_token_id = tokenizer.pad_token_id = 0 
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if args.test_after_train:
        logger.log("\n==================Generation Results After Pruning================\n")

        def model_gen(prompt, logger, model, tokenizer, args):
            input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(args.eval_device)

            generation_output = model.generate(
                input_ids=input_ids,
                do_sample=True,
                top_k=50,
                max_length=args.max_seq_len,
                top_p=args.top_p,
                temperature=args.temperature,
            )
            
            result = tokenizer.decode(generation_output[0])
            logger.log(result)
        
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                model_gen(prompt, logger, model, tokenizer, args)
            
            
            print("Start chatting with the finetuned model!")
            while True:
                print("------------------------------")
                print("Please input the prompt text:")
                prompt = input()
                responds, history = model.chat(tokenizer, prompt, temperature=0.8, top_p=0.8)
                print(responds)
                # model_gen(prompt, logger, model, tokenizer, args)
                
        
        logger.log("\n==================Finish================\n")
    
    if args.eval_device != "cpu":
        model.half()
    model.to(args.eval_device)
    
    ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.eval_device)
    logger.log("PPL after pruning: {}".format(ppl))
    logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

    # Finetune
    from post_training_minicpm import finetune
    model, tokenizer = finetune(args, model, tokenizer)

    ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.eval_device)
    logger.log("PPL after finetuning: {}".format(ppl))
    logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

    print("Start chatting with the finetuned model!")
    while True:
        print("------------------------------")
        print("Please input the prompt text:")
        prompt = input()
        responds, history = model.chat(tokenizer, prompt, temperature=0.8, top_p=0.8)
        print(responds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--save_ckpt_log_name', type=str, default="llama_prune", help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='pruning ratio')
    parser.add_argument('--pruner_type', type=str, default='l2', help='pruner type')

    # argument for generation
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')

    # argument for layer-wise pruning/column-wise pruning
    parser.add_argument('--channel_wise', action='store_true', help='channel wise')
    parser.add_argument('--block_wise', action='store_true', help='block wise')
    parser.add_argument('--layer_wise', action='store_true', help='layer wise')
    parser.add_argument('--layer', type=int, default=12, help='remain the previous n layers')

    parser.add_argument('--block_attention_layer_start', type=int, help='start layer of block attention layers', default=3)
    parser.add_argument('--block_attention_layer_end', type=int, help='end layer of block attention layers', default=31)
    parser.add_argument('--block_mlp_layer_start', type=int, help='start layer of block mlp layers', default=3)
    parser.add_argument('--block_mlp_layer_end', type=int, help='end layer of block mlp layers', default=31)

    parser.add_argument('--iterative_steps', type=int, default=1, help="Iteration step for pruning. Default=1")
    parser.add_argument('--grouping_strategy', type=str, default='sum', help='Reduce method for grouping')
    parser.add_argument('--global_pruning', action='store_true', help='whether global pruning')
    parser.add_argument('--taylor', type=str, default='param_first', help='choose from [vectorize, param_second, param_first, param_mix]')
    parser.add_argument('--num_examples', type=int, default=10)

    # general argument
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--test_before_train', action='store_true', help='whether test before train')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
    parser.add_argument('--test_after_train', action='store_true', help='whether test after train')

    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_model', action='store_true', help='if save model')
    parser.add_argument('--save_model_path', type=str, help='path to save pruned model')

    # For finetuning
    # Model Type&Path
    parser.add_argument('--data_path', type=str, default="yahma/alpaca-cleaned", help='data path')
    parser.add_argument('--cache_dataset', action="store_true", default=False)
    parser.add_argument('--extra_val_dataset', type=str, default=None, help='validation datasets. Split with ","')
    parser.add_argument('--output_dir', type=str, default="./lora-alpaca", help='output directory')

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
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
    
    #ddp
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
