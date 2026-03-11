import os
import torch
torch._dynamo.disable()
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import gc
import sys
import time
import json
import copy
import random
import argparse
from typing import Tuple
from tqdm import tqdm


import numpy as np
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig, AutoTokenizer
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForSpeechSeq2Seq
from transformers.image_utils import load_image

import LLMPruner.torch_pruning as tp 
from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric
from eval_textvqa import eval_model
from LLMPruner.datasets.example_samples import get_examples

from audiobench.dataset import Dataset
from audiobench.model import Model
from audiobench.dataset_src.prompts.prompts import asr_instructions
import random
import shutil
from pathlib import Path

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def main(args):
    set_random_seed(args.seed)

    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name), 
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )
    # processor = AutoProcessor.from_pretrained(
    #     args.base_model, 
    #     trust_remote_code=True,
    # )
    # model = AutoModelForSpeechSeq2Seq.from_pretrained(
    #     args.base_model,
    #     use_safetensors=True,
    #     trust_remote_code=True,
    #     attn_implementation="flash_attention_2",
    #     torch_dtype=torch.bfloat16
    # ).to(args.device)
    model = Model(args.base_model)
    # model = net.model
    processor = model.processor

    # Ensure model is on CUDA for pruning.
    # The model loader may leave weights on CPU (no device_map, no .to("cuda")).
    first_param = next(model.model.parameters())
    if first_param.device.type == 'cpu':
        print("[FIX] Model is on CPU, moving to CUDA for pruning...")
        model.model = model.model.to('cuda')
    elif hasattr(model.model, 'hf_device_map'):
        from accelerate.hooks import remove_hook_from_submodules
        remove_hook_from_submodules(model.model)
        model.model = model.model.to('cuda')
        print("[FIX] Removed accelerate dispatch hooks, model moved to CUDA")
    print(f"[INFO] Model device: {next(model.model.parameters()).device}")

    # Monkey-patch get_loss to fix server's buggy do_sample_get_loss.
    # Server's version calls self.processor(text=reference) without audio, crashing at audio.ndim.
    # Also, reference may be a dict {'text': '...', 'audio': None} instead of a string.
    def _fixed_get_loss(input_data):
        audio_array = input_data["audio"]["array"]
        sampling_rate = input_data["audio"]["sampling_rate"]
        instruction = input_data["instruction"]
        reference = input_data.get("reference", "")
        if isinstance(reference, dict):
            reference = reference.get("text", "")
        audio_duration = len(audio_array) / sampling_rate
        if audio_duration < 1:
            import numpy as np
            audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')
        # Build prompt and process with audio
        prompt = "Instruction: {instruction} \nFollow the text instruction based on the following audio: <SpeechHere>"
        conversation = [{"role": "user", "content": prompt.format(instruction=instruction)}]
        chat_prompt = processor.tokenizer.apply_chat_template(
            conversation=conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=chat_prompt, audios=audio_array)
        for key in inputs:
            if not isinstance(inputs[key], torch.Tensor):
                inputs[key] = torch.tensor(inputs[key])
            inputs[key] = inputs[key].to('cuda')
            if inputs[key].dtype == torch.float32:
                inputs[key] = inputs[key].to(torch.bfloat16)
        loss = model.model(**inputs, labels=inputs['input_ids']).loss
        return loss
    model.get_loss = _fixed_get_loss
    print("[FIX] Patched model.get_loss to fix audio/reference handling")

    print(model.model)

    # model.config.pad_token_id = processor.pad_token_id = 0
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2

    DATASET_ID_CALIB = "imda_part2_asr_test"
    DATASET_ID_TEST = "imda_part1_asr_test"
    nsamples = args.num_examples + 1
    MAX_SEQUENCE_LENGTH = 256

    # Load calibration dataset with caching (avoids 8x parallel loading of 125k samples)
    # First process builds cache via audiobench.Dataset; other 7 wait via file lock, then load instantly.
    import pickle, fcntl
    CACHE_DIR = "/home/jinchao/runtao/meralion_datasets/ASR/_cached"
    calib_cache = os.path.join(CACHE_DIR, f"calib_{DATASET_ID_CALIB}_{nsamples}.pkl")
    cache_lock = os.path.join(CACHE_DIR, ".calib_cache.lock")

    if os.path.exists(calib_cache):
        print(f"[DATA] Loading cached calibration data from {calib_cache}")
        with open(calib_cache, 'rb') as f:
            calib_input_data = pickle.load(f)
    else:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_lock, 'w') as lf:
            try:
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if os.path.exists(calib_cache):
                    print(f"[DATA] Cache appeared, loading...")
                    with open(calib_cache, 'rb') as f:
                        calib_input_data = pickle.load(f)
                else:
                    print(f"[DATA] Building calibration cache (first run only)...")
                    calib_ds = Dataset(DATASET_ID_CALIB, nsamples)
                    calib_input_data = calib_ds.input_data
                    with open(calib_cache, 'wb') as f:
                        pickle.dump(calib_input_data, f)
                    print(f"[DATA] Calibration cache saved.")
                fcntl.flock(lf, fcntl.LOCK_UN)
            except BlockingIOError:
                print(f"[DATA] Waiting for another process to build calibration cache...")
                fcntl.flock(lf, fcntl.LOCK_EX)
                fcntl.flock(lf, fcntl.LOCK_UN)
                print(f"[DATA] Loading cached calibration data...")
                with open(calib_cache, 'rb') as f:
                    calib_input_data = pickle.load(f)

    # Normalize audio format: meralion_2_model_get_inputs expects input["audio"]["array"],
    # but audiobench.Dataset may return input["audio"] = sample['context'] which is
    # 3-level nested: {"audio": {"array": ..., "sampling_rate": ...}, ...}.
    # Flatten to 2-level so input["audio"]["array"] works directly.
    for item in calib_input_data:
        if "audio" in item and isinstance(item["audio"], dict) and "audio" in item["audio"]:
            item["audio"] = item["audio"]["audio"]

    forward_prompts = calib_input_data[-1]
    example_prompts = calib_input_data[:20]
    
    if args.test_before_train:
        logger.log("\n==================Generation Results before Pruning================\n")
        # model = model.to(args.eval_device)
        # model.eval()
        # with torch.no_grad():
        #     generated_texts = model.generate(forward_prompts)
        #     logger.log(generated_texts)
    
        # ppl = PPLMetric(model,  processor.tokenizer, ['c4'], args.max_seq_len, device=args.eval_device)
        # logger.log("PPL before pruning: {}".format(ppl))
        # logger.log("\n==================before pruning================\n")
        # eval_model(
        #     model=model,
        #     processor=processor,
        #     num_chunks=100
        #     # image_folder="./playground/data/eval/textvqa/train_images",
        #     # question_file="./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl",
        #     # annotation_file="./playground/data/eval/textvqa/TextVQA_0.5.1_val.json",
        #     # answers_file="answer.jsonl",
        #     # chunk_idx=0,
        #     # num_chunks=1,
        #     # max_new_tokens=500,
        #     # model_id=args.save_ckpt_log_name
        # )

        # eval with test dataset
        print(model.model)
        model.model.eval()
        nsamples = -1
        dataset = Dataset(DATASET_ID_TEST, nsamples)
        model.dataset_name = dataset.dataset_name
        file_save_folder = 'audiobench_log_for_all_models'
        batch_size = 1
        model_name = args.base_model
        dataset_name = DATASET_ID_TEST
        metrics = "wer"

        def do_model_prediction(input_data, model, batch_size):

            if batch_size not in [1, -1]:
                raise NotImplementedError("Batch size {} not implemented yet".format(batch_size))
            
            if batch_size == -1:
                model_predictions = model.generate(input_data)
            
            else:
                model_predictions = []
                for inputs in tqdm(input_data, leave=False):
                    outputs = model.generate(inputs)
                    if isinstance(outputs, list):
                        model_predictions.extend(outputs)
                    else:
                        model_predictions.append(outputs)
                        
            return model_predictions

        # Infer with model
        torch.cuda.synchronize()
        st = time.time()
        model_predictions           = do_model_prediction(dataset.input_data, model, batch_size=batch_size)
        torch.cuda.synchronize()
        et = time.time()
        if nsamples == -1:
            nsamples = len(dataset.input_data)
        print(nsamples, "samples took:", et-st, "s")
        data_with_model_predictions = dataset.dataset_processor.format_model_predictions(dataset.input_data, model_predictions)
        # input("inference ended")

    # model.to(args.device)
    pruner_type = args.pruner_type.lower()
    assert pruner_type in ['random', 'l2', 'l1', 'taylor']

    # for param in model.model.text_decoder.model.parameters():
    #     param.requires_grad_(True)
    for p in model.model.parameters():
        p.requires_grad_(False)
    for layer in model.model.text_decoder.model.layers:
        for p in layer.mlp.parameters():
            p.requires_grad_(True)
        for p in layer.self_attn.parameters():
            p.requires_grad_(True)
    # Also enable gradients for whisper encoder layers if whisper pruning is requested
    if args.whisper_block_layer_start >= 0 and args.whisper_block_layer_end > args.whisper_block_layer_start:
        for layer in model.model.speech_encoder.layers:
            for p in layer.self_attn.parameters():
                p.requires_grad_(True)
            for p in layer.fc1.parameters():
                p.requires_grad_(True)
            for p in layer.fc2.parameters():
                p.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    
    # forward_prompts = torch.tensor([
    #     [    1,   306,  4658,   278,  6593,   310,  2834,   338],
    #     [    1,  3439, 17632,  1925, 29892,   278,  6368,   310],
    # ]).to(args.device) # Only for building the dependency graph. Any input will be fine since the computation result are not taken into consideration.

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
        # Enable gradient checkpointing on the text decoder
        model.model.text_decoder.model.gradient_checkpointing_enable()
        # IMPORTANT: disable KV cache in training/backward
        if hasattr(model.model.text_decoder.model, "config"):
            model.model.text_decoder.model.config.use_cache = False
        if hasattr(model.model, "config"):
            model.model.config.use_cache = False

        for m in model.model.modules():
            if 'rms' in m.__class__.__name__.lower():
                LLamaRMSNorm = type(m)
                break

        for m in model.model.modules():
            if 'layernorm' in m.__class__.__name__.lower():
                LayerNorm = type(m)
                break
        
        # Resolve per-component pruning ratios (fall back to --pruning_ratio if not specified)
        text_mlp_ratio = args.text_mlp_pruning_ratio if args.text_mlp_pruning_ratio is not None else args.pruning_ratio
        text_attn_ratio = args.text_attn_pruning_ratio if args.text_attn_pruning_ratio is not None else args.pruning_ratio
        whisper_attn_ratio = args.whisper_attn_pruning_ratio if args.whisper_attn_pruning_ratio is not None else args.pruning_ratio
        whisper_mlp_ratio = args.whisper_mlp_pruning_ratio if args.whisper_mlp_pruning_ratio is not None else args.pruning_ratio

        # Build root_instances and per-module sparsity dict
        root_instances = []
        ch_sparsity_dict = {}

        for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end):
            m = model.model.text_decoder.model.layers[i].mlp.gate_proj
            root_instances.append(m)
            ch_sparsity_dict[m] = text_mlp_ratio

        for i in range(args.block_attention_layer_start, args.block_attention_layer_end):
            m = model.model.text_decoder.model.layers[i].self_attn.k_proj
            root_instances.append(m)
            ch_sparsity_dict[m] = text_attn_ratio

        if whisper_attn_ratio > 0:
            for i in range(args.whisper_block_layer_start, args.whisper_block_layer_end):
                m = model.model.speech_encoder.layers[i].self_attn.k_proj
                root_instances.append(m)
                ch_sparsity_dict[m] = whisper_attn_ratio

        if whisper_mlp_ratio > 0:
            for i in range(args.whisper_block_layer_start, args.whisper_block_layer_end):
                m = model.model.speech_encoder.layers[i].fc1
                root_instances.append(m)
                ch_sparsity_dict[m] = whisper_mlp_ratio

        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio,  # default fallback
            "ch_sparsity_dict": ch_sparsity_dict,  # per-component ratios
            "ignored_layers":[],
            "channel_groups": {
            },
            "consecutive_groups": {
                **{layer.self_attn.k_proj: layer.self_attn.head_dim for layer in model.model.text_decoder.model.layers},
                **{layer.self_attn.k_proj: layer.self_attn.head_dim for layer in model.model.speech_encoder.layers},
            },
            "customized_pruners": {
                LLamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
                LayerNorm: llama_pruner.hf_rmsnorm_pruner,
            },
            "root_module_types": None,
            "root_instances": root_instances,
            "forward_fn": lambda model, example_inputs: model(**example_inputs).logits
        }
        logger.log("Pruning Text Attention (ratio={}) Layer = {}".format(text_attn_ratio, list(range(args.block_attention_layer_start, args.block_attention_layer_end))))
        logger.log("Pruning Text MLP (ratio={}) Layer = {}".format(text_mlp_ratio, list(range(args.block_mlp_layer_start, args.block_mlp_layer_end))))
        logger.log("Pruning Whisper Attn (ratio={}) Layer = {}".format(whisper_attn_ratio, list(range(args.whisper_block_layer_start, args.whisper_block_layer_end))))
        logger.log("Pruning Whisper MLP (ratio={}) Layer = {}".format(whisper_mlp_ratio, list(range(args.whisper_block_layer_start, args.whisper_block_layer_end))))

        pruner = tp.pruner.MeralionPruner(
            model.model,
            model,
            forward_prompts,
            **kwargs
        )
        model.model.zero_grad()

        logger.log("Start Pruning")
        for i in range(args.iterative_steps):

            if pruner_type in ['taylor']:
                # example_prompts = get_examples('cauldron', processor, args.num_examples, seq_len = 512)
                # example_prompts = get_examples('c4', processor, args.num_examples, seq_len = 64).to(args.device)
                logger.log("Start Backwarding in iterative steps = {}...".format(i))
                if args.taylor in ['param_mix', 'param_second']:
                    for j in range(args.num_examples):
                        print(j)
                        # print(tokenizer.decode(example_prompts[j]))
                        # example_prompts[j] = example_prompts[j].to(args.device)
                        # loss = model.get_loss(**example_prompts[j], labels=example_prompts[j].input_ids).loss
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = model.get_loss(example_prompts[j])
                        logger.log("Loss = {}".format(loss))
                        loss.backward()

                        for module_param in model.model.parameters():
                            if module_param.grad is None:
                                continue
                            # module_param.grad = module_param.grad * module_param.grad / args.num_examples
                            g2_cpu = module_param.grad.detach() * module_param.grad.detach() / args.num_examples
                            g2_cpu = g2_cpu.cpu()
                            if hasattr(module_param, 'acc_grad'):
                                # module_param.acc_grad += module_param.grad
                                module_param.acc_grad += g2_cpu
                            else:
                                # module_param.acc_grad = copy.deepcopy(module_param.grad)
                                module_param.acc_grad = g2_cpu
                        # model.model.zero_grad()
                        model.model.zero_grad(set_to_none=True)
                        del loss.grad
                    
                # loss = model(**example_prompts[j], labels=example_prompts[j].input_ids).loss
                loss = model.get_loss(example_prompts[j])
                logger.log("Loss = {}".format(loss))
                loss.backward()

            pruner.step()

            after_pruning_parameters = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))
        
            # modify inferece-related attributes
            for layer in model.model.text_decoder.model.layers:
                layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
                layer.self_attn.num_key_value_heads = layer.self_attn.k_proj.weight.data.shape[0] // layer.self_attn.head_dim

            # Also update whisper encoder attention heads after pruning
            for layer in model.model.speech_encoder.layers:
                if hasattr(layer.self_attn, 'head_dim') and layer.self_attn.head_dim > 0:
                    layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
                    layer.self_attn.embed_dim = layer.self_attn.q_proj.weight.data.shape[0]

        # Clean the gradient in the model
        model.model.zero_grad()
        for name, module in model.model.named_parameters():
            if 'weight' in name:
                module.grad = None 

        del pruner


    elif args.channel_wise:
        for m in model.modules():
            if 'rms' in m.__class__.__name__.lower():
                print(m.__class__.__name__)
                Qwen2RMSNorm = type(m)
                break

        # adjust hidden size to be a multiple of attention heads
        # Adjust hidden size to be a multiple of attention heads and divisible by 32
        hidden_size = model.config.hidden_size
        num_attention_heads = model.config.num_attention_heads

        # Calculate the adjusted hidden size
        adjusted_hidden_size = round((hidden_size // num_attention_heads) * args.pruning_ratio / 16) * 16 * num_attention_heads
        # Calculate the adjusted pruning ratio
        adjusted_pruning_ratio = (adjusted_hidden_size / hidden_size)


        # Update the model configuration
        # adjusted_pruning_ratio = (model.config.hidden_size * args.pruning_ratio // model.config.num_attention_heads + 1) \
        #                         * model.config.num_attention_heads / model.config.hidden_size
        print("adjusted pruning ratio:", adjusted_pruning_ratio)
        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": adjusted_pruning_ratio, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            "ignored_layers":[],
            #"round_to": model.config.num_attention_heads * 2,
            "channel_groups": {
                # layer.self_attn: layer.self_attn.num_heads for layer in model.model.layers
            },
            "customized_pruners": {
                Qwen2RMSNorm: llama_pruner.hf_rmsnorm_pruner,
                # Qwen2SdpaAttention: llama_pruner.hf_attention_pruner,
            },
            "root_module_types": [Qwen2RMSNorm, Qwen2Attention],
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
                example_prompts = torch.cat((get_examples('c4', tokenizer, args.num_examples, seq_len = 64).to(args.device),
                                            get_examples('bookcorpus', tokenizer, args.num_examples, seq_len = 64).to(args.device),
                                            get_examples('wikitext', tokenizer, args.num_examples, seq_len = 64).to(args.device)))
                logger.log("Start Backwarding in iterative steps = {}...".format(i))
                loss = model(example_prompts, labels=example_prompts).loss
                logger.log("Loss = {}".format(loss))
                loss.backward()

            pruner.step()

            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))

        # Clean the gradient in the model
        model.zero_grad()
        for name, module in model.named_parameters():
            if 'weight' in name:
                module.grad = None

        # modify inferece-related attributes
        model.config.hidden_size = model.model.embed_tokens.weight.shape[1]
        model.zero_grad()
        
        del pruner
            
    elif args.layer_wise:
        model.model.layers = model.model.layers[:args.layer]
        after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    else:
        raise NotImplementedError
    logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))
    
    gc.collect()
    torch.cuda.empty_cache()

    # print(model.model)

    # if args.save_model:
    #     # model.half()
    #     torch.save({
    #         'model': model, 
    #         'processor': processor,
    #     }, logger.best_checkpoint_path)
    if args.save_model_path:
            # config_updates = {
            #     "midblock_ratio": 1-args.pruning_ratio,
            #     "midblock_start": args.block_mlp_layer_start,
            #     "midblock_end": args.block_mlp_layer_end,
            #     "text_config": {
            #         "midblock_ratio": 1-args.pruning_ratio,
            #         "midblock_start": args.block_mlp_layer_start,
            #         "midblock_end": args.block_mlp_layer_end
            #     },
            #     "speech_config": {
            #         "whisper_midblock_start": args.whisper_block_layer_start,
            #         "whisper_midblock_end": args.whisper_block_layer_end,
            #     },
            # }
            # model.model.config.update(config_updates)
            # Resolve per-component ratios for config saving
            _text_attn_r = args.text_attn_pruning_ratio if args.text_attn_pruning_ratio is not None else args.pruning_ratio
            _text_mlp_r = args.text_mlp_pruning_ratio if args.text_mlp_pruning_ratio is not None else args.pruning_ratio
            _whisper_attn_r = args.whisper_attn_pruning_ratio if args.whisper_attn_pruning_ratio is not None else args.pruning_ratio
            _whisper_mlp_r = args.whisper_mlp_pruning_ratio if args.whisper_mlp_pruning_ratio is not None else args.pruning_ratio

            # Text decoder config (use attn ratio for midblock_ratio since it affects head count)
            model.model.config.midblock_ratio = 1 - _text_attn_r
            model.model.config.text_config.midblock_ratio = 1 - _text_attn_r
            model.model.config.text_config.text_mlp_midblock_ratio = 1 - _text_mlp_r
            model.model.config.midblock_end = args.block_mlp_layer_end
            model.model.config.text_config.midblock_end = args.block_mlp_layer_end
            model.model.config.midblock_start = args.block_mlp_layer_start
            model.model.config.text_config.midblock_start = args.block_mlp_layer_start

            # Whisper encoder config (only write ratios when whisper pruning is active)
            model.model.config.speech_config.whisper_midblock_start = args.whisper_block_layer_start
            model.model.config.speech_config.whisper_midblock_end = args.whisper_block_layer_end
            if args.whisper_block_layer_start >= 0:
                model.model.config.speech_config.whisper_attn_midblock_ratio = 1 - _whisper_attn_r
                model.model.config.speech_config.whisper_mlp_midblock_ratio = 1 - _whisper_mlp_r

            
            model.model.save_pretrained(args.save_model_path)
            processor.save_pretrained(args.save_model_path)

            # copy code files
            def copy_files_only(src_dir, dst_dir):
                src_dir = Path(src_dir)
                dst_dir = Path(dst_dir)
                dst_dir.mkdir(parents=True, exist_ok=True)

                for file in src_dir.glob("*.py"):
                    if file.is_file():
                        shutil.copy2(file, dst_dir / file.name)

            # Example
            copy_files_only("./meralion2_bl", args.save_model_path)
            print("\n\nPruned model has been saved!\n\n")

            # --- WER evaluation immediately after pruning (before finetuning) ---
            # Aligned with post_training_meralion.py final WER evaluation:
            # same dataset, same samples, same prompt, same metric, same post-processing
            # Disabled by default; use --post_prune_eval to enable
            if args.post_prune_eval:
                import evaluate as hf_evaluate
                from datasets import load_from_disk

                logger.log("\n==================WER Evaluation After Pruning (Before Finetuning)================\n")
                model.model.eval()
                wer_metric = hf_evaluate.load("wer")

                test_data = load_from_disk("/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR")
                test_subset = test_data.shuffle(seed=42).select(range(10500, 11000))

                predictions = []
                references = []

                st = time.time()
                with torch.no_grad():
                    for sample in tqdm(test_subset, desc="Post-prune WER eval"):
                        audio_array = sample["context"]["audio"]["array"]
                        instruction = sample["instruction"]["text"] if isinstance(sample["instruction"], dict) else sample["instruction"]
                        ref = sample["other_attributes"]["Transcription"]

                        audio_array = np.asarray(audio_array, dtype=np.float32)
                        if audio_array.ndim == 2:
                            audio_array = audio_array.mean(axis=-1)
                        if len(audio_array) / 16000 < 1:
                            audio_array = np.pad(audio_array, (0, 16000), 'constant')

                        prompt = "Instruction: {instruction} \nFollow the text instruction based on the following audio: <SpeechHere>"
                        conversation = [{"role": "user", "content": prompt.format(instruction=instruction)}]
                        chat_prompt = processor.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

                        inputs = processor(text=chat_prompt, audios=audio_array, return_tensors="pt")
                        dev = next(model.model.parameters()).device
                        for k in inputs:
                            if isinstance(inputs[k], torch.Tensor):
                                inputs[k] = inputs[k].to(dev)
                                if inputs[k].dtype == torch.float32:
                                    inputs[k] = inputs[k].to(torch.bfloat16)

                        model_outputs = model.model.generate(**inputs, max_new_tokens=256, do_sample=False, num_beams=1)
                        input_len = inputs['input_ids'].shape[1]
                        generated_ids = model_outputs[:, input_len:]
                        pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        pred = pred.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()

                        predictions.append(pred)
                        references.append(ref)

                et = time.time()
                post_prune_wer = wer_metric.compute(predictions=predictions, references=references)
                logger.log("500 samples took: {:.1f}s".format(et - st))
                logger.log("Post-prune WER (before finetuning): {:.4f}".format(post_prune_wer))
                print("\n[POST-PRUNE EVAL] WER = {:.4f} (500 samples, {:.1f}s)\n".format(post_prune_wer, et - st))

                # Save results
                eval_save_name = os.path.basename(args.save_model_path)
                eval_save_dir = f'audiobench_log_for_all_models/{eval_save_name}-pruned-only'
                os.makedirs(eval_save_dir, exist_ok=True)
                with open(f'{eval_save_dir}/post_prune_wer.json', 'w') as f:
                    json.dump({
                        "wer": post_prune_wer,
                        "num_samples": len(predictions),
                        "dataset": "IMDA_PART1_mono_en_30_ASR",
                        "base_model": args.base_model,
                        "pruned_model": args.save_model_path,
                    }, f, indent=2)
                # Save per-sample details
                details = [{"prediction": p, "reference": r} for p, r in zip(predictions[:20], references[:20])]
                with open(f'{eval_save_dir}/post_prune_wer_samples.json', 'w') as f:
                    json.dump(details, f, indent=2, ensure_ascii=False)
                logger.log("Results saved to: {}".format(eval_save_dir))

            exit()
    
    # if args.eval_device != "cpu":
    #     model.half()
    # model.to(args.eval_device)

    # if args.test_after_train:
    #     logger.log("\n==================Generation Results After Pruning================\n")
        
    #     model.eval()
    #     with torch.no_grad():
    #         for prompt in prompts:
    #             input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(args.eval_device)

    #             generation_output = model.generate(
    #                 input_ids=input_ids,
    #                 do_sample=True,
    #                 top_k=50,
    #                 max_length=args.max_seq_len,
    #                 top_p=args.top_p,
    #                 temperature=args.temperature,
    #             )
                
    #             result = tokenizer.decode(generation_output[0])
    #             logger.log(result)
        
    #     logger.log("\n==================Finish================\n")

    # eval with test dataset
    model.model.eval()
    nsamples = -1
    dataset = Dataset(DATASET_ID_TEST, nsamples)
    model.dataset_name = dataset.dataset_name
    file_save_folder = 'audiobench_log_for_all_models'
    batch_size = 1
    model_name = args.base_model
    dataset_name = DATASET_ID_TEST
    metrics = "wer"

    def do_model_prediction(input_data, model, batch_size):

        if batch_size not in [1, -1]:
            raise NotImplementedError("Batch size {} not implemented yet".format(batch_size))
        
        if batch_size == -1:
            model_predictions = model.generate(input_data)
        
        else:
            model_predictions = []
            for inputs in tqdm(input_data, leave=False):
                outputs = model.generate(inputs)
                if isinstance(outputs, list):
                    model_predictions.extend(outputs)
                else:
                    model_predictions.append(outputs)
                    
        return model_predictions

    # Infer with model
    st = time.time()
    model_predictions           = do_model_prediction(dataset.input_data, model, batch_size=batch_size)
    et = time.time()
    print("50 samples took:", et-st, "s")
    data_with_model_predictions = dataset.dataset_processor.format_model_predictions(dataset.input_data, model_predictions)
    # input("inference ended")

    # Save the result with predictions
    os.makedirs(f'{file_save_folder}/{model_name}', exist_ok=True)
    with open(f'{file_save_folder}/{model_name}/{dataset_name}.json', 'w') as f:
        json.dump(data_with_model_predictions, f, indent=4, ensure_ascii=False)
    
    data_with_model_predictions = json.load(open(f'{file_save_folder}/{model_name}/{dataset_name}.json'))
    results = dataset.dataset_processor.compute_score(data_with_model_predictions, metrics=metrics)
    with open(f'{file_save_folder}/{model_name}/{dataset_name}_{metrics}_score.json', 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # Take only the first 100 samples for record.
    if 'details' in results:
        results['details'] = results['details'][:20]

    # Print the result with metrics
    logger.info('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =')
    logger.info('Dataset name: {}'.format(dataset_name.upper()))
    logger.info('Model name: {}'.format(model_name.upper()))
    logger.info(json.dumps({metrics: results[metrics]}, indent=4, ensure_ascii=False))
    logger.info('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =')

    # Save the scores
    with open(f'{file_save_folder}/{model_name}/{dataset_name}_{metrics}_score.json', 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    # input("Eval complete")
    
    # PPL
    from functools import partial
    ppl = PPLMetric(model, processor.tokenizer, ["c4"], args.max_seq_len, device=args.eval_device) 
    logger.log("PPL after pruning: {}".format(ppl))
    
    # eval_model(
    #     model=model,
    #     processor=processor,
    #     num_chunks=100
    #     # image_folder="./playground/data/eval/textvqa/train_images",
    #     # question_file="./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl",
    #     # annotation_file="./playground/data/eval/textvqa/TextVQA_0.5.1_val.json",
    #     # answers_file="answer.jsonl",
    #     # chunk_idx=0,
    #     # num_chunks=1,
    #     # max_new_tokens=500,
    #     # model_id=args.save_ckpt_log_name
    # )
    
    logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--base_model', type=str, default="meta-llama/Llama-2-7b-hf", help='base model name')
    parser.add_argument('--save_ckpt_log_name', type=str, default="llama_prune", help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='default pruning ratio (used when component-specific ratios are not set)')
    parser.add_argument('--text_attn_pruning_ratio', type=float, default=None, help='pruning ratio for text decoder attention heads (reduces KV cache)')
    parser.add_argument('--text_mlp_pruning_ratio', type=float, default=None, help='pruning ratio for text decoder MLP intermediate dim')
    parser.add_argument('--whisper_attn_pruning_ratio', type=float, default=None, help='pruning ratio for whisper encoder attention heads')
    parser.add_argument('--whisper_mlp_pruning_ratio', type=float, default=None, help='pruning ratio for whisper encoder MLP (fc1/fc2)')
    parser.add_argument('--pruner_type', type=str, default='l2', help='pruner type')

    # argument for generation
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p')
    parser.add_argument('--max_seq_len', type=int, default=512, help='max sequence length')

    # argument for layer-wise pruning/column-wise pruning
    parser.add_argument('--channel_wise', action='store_true', help='channel wise')
    parser.add_argument('--block_wise', action='store_true', help='block wise')
    parser.add_argument('--layer_wise', action='store_true', help='layer wise')
    parser.add_argument('--layer', type=int, default=12, help='remain the previous n layers')

    parser.add_argument('--whisper_block_layer_start', type=int, help='start layer of whisper layers', default=-1)
    parser.add_argument('--whisper_block_layer_end', type=int, help='end layer of whisper layers', default=-1)
    parser.add_argument('--block_attention_layer_start', type=int, help='start layer of block attention layers', default=-1)
    parser.add_argument('--block_attention_layer_end', type=int, help='end layer of block attention layers', default=-1)
    parser.add_argument('--block_mlp_layer_start', type=int, help='start layer of block mlp layers', default=-1)
    parser.add_argument('--block_mlp_layer_end', type=int, help='end layer of block mlp layers', default=-1)

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
    parser.add_argument('--save_model_path', type=str, default=None, help=' ')
    parser.add_argument('--post_prune_eval', action='store_true', help='run WER evaluation after pruning (before finetuning)')
    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
