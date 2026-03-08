'''
Refer to
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
'''

import os
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache_" + str(os.getpid())  # per-process temp cache
os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = "0"  # no limit for in-memory datasets
import sys
import argparse
from typing import List
from pathlib import Path

import torch
import transformers
from datasets import load_dataset, concatenate_datasets, Dataset, interleave_datasets
from datasets import Dataset as Dataset_datasets, load_from_disk
# from torchvision.datasets import CocoDetection
# CocoDetection(root="/home/kaixin/datasets/coco/train2017", annFile="/home/kaixin/datasets/coco/annotations/instances_train2017.json")
from trl import SFTTrainer, SFTConfig

from peft import LoraConfig, get_peft_model, TaskType, get_peft_model_state_dict, prepare_model_for_kbit_training

from transformers import AutoProcessor, AutoModelForVision2Seq, Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments

from LLMPruner.datasets.ppl_dataset import get_loaders
from transformers.image_utils import load_image
from PIL import Image
from io import BytesIO
import base64
import copy
from transformers import BitsAndBytesConfig

from transformers import TrainerCallback
from VLMEvalKit_utils import encode_image_to_base64
import csv
import json
import pdb

from audiobench.dataset import Dataset
from audiobench.model import Model
from audiobench.dataset_src.prompts.prompts import asr_instructions
import random
import numpy as np
import time
import copy
import tqdm
import wandb

import evaluate 
metric = evaluate.load("wer")

device = "cuda" if torch.cuda.is_available() else "cpu"

# system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
# Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
# The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
# Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

# def fetch_image(ele: dict[str, str | Image.Image]) -> Image.Image:
#     if "image" in ele:
#         image = ele["image"]
#     else:
#         image = ele["image_url"]
#     image_obj = None
#     if isinstance(image, Image.Image):
#         image_obj = image
#     elif image.startswith("http://") or image.startswith("https://"):
#         # fix memory leak issue while using BytesIO
#         image_obj = load_image(image)
#     elif image.startswith("file://"):
#         image_obj = Image.open(image[7:])
#     elif image.startswith("data:image"):
#         if "base64," in image:
#             _, base64_data = image.split("base64,", 1)
#             data = base64.b64decode(base64_data)
#             # fix memory leak issue while using BytesIO
#             with BytesIO(data) as bio:
#                 image_obj = copy.deepcopy(Image.open(bio))
#     else:
#         image_obj = Image.open(image)
#     if image_obj is None:
#         raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")

#     return image

def fetch_image(image) -> Image.Image:
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        # fix memory leak issue while using BytesIO
        image_obj = load_image(image)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            # fix memory leak issue while using BytesIO
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")

    return image_obj


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type","") in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def remove_image_from_conversations(
    conversations: list[dict] | list[list[dict]],
) -> list[dict] | list[list[dict]]:
    """
    Remove image or video from conversations.
    """
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == 'image':
                        if "image" in ele:
                            del ele["image"]
                        if "image_url" in ele:
                            del ele["image_url"]
                    elif ele["type"] == 'video':
                        if "video" in ele:
                            del ele["video"]
    return conversations

def process_vision_info(
    conversations: list[dict] | list[list[dict]],
) :

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        # elif "video" in vision_info:
        #     video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
        #     video_sample_fps_list.append(video_sample_fps)
        #     video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs #, video_inputs

# class DecodeAndPrintCallback(transformers.TrainerCallback):
#     def __init__(self, tokenizer, prompts, max_length=50):
#         self.tokenizer = tokenizer
#         self.prompts = prompts
#         self.max_length = max_length
#         self.prompt_ids = [tokenizer([
#             tokenizer.apply_chat_template(
#                 [
#                     {"role": "system", "content": "You are a helpful assistant."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 tokenize=False,
#                 add_generation_prompt=True
#             )], return_tensors="pt").to(device) for prompt in prompts]

#     def on_evaluate(self, args, state, control, **kwargs):
#         model = kwargs["model"]
        
#         for prompt_id in self.prompt_ids:
#             generated_ids = model.generate(
#                 input_ids=prompt_id.input_ids,
#                 max_new_tokens=self.max_length,
#                 do_sample=True,
#                 temperature=0.7,
#                 top_k=50,
#                 top_p=0.95
#             )
#             generated_ids = [
#                 output_ids[len(input_ids):] for input_ids, output_ids in zip(prompt_id.input_ids, generated_ids)
#             ]

#             response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#             print(f"\n=== Decoded output after epoch {state.epoch:.0f} ===")
#             print(response)
#             print("=" * 40)

from utils.callbacks import WEREvalCallback

def main(args):
    
    # Load Pruned Model
    model = Model(args.base_model)
    processor = model.processor

    print(model.model)

    # Check if whisper encoder was pruned (from saved config)
    whisper_pruned = (
        hasattr(model.model.config, 'speech_config') and
        getattr(model.model.config.speech_config, 'whisper_midblock_start', -1) >= 0
    )

    for param in model.model.speech_encoder.parameters():
        param.requires_grad = False
    for param in model.model.ln_speech.parameters():
        param.requires_grad = False
    for param in model.model.speech_audio_adapter.parameters():
        param.requires_grad = False

    model.model.text_decoder.gradient_checkpointing_enable()
    model.model.config.use_cache = False
    model.model.text_decoder.config.use_cache = False  # important for training
    # model.text_decoder = prepare_model_for_kbit_training(model.text_decoder)


    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size


    # processor.tokenizer.pad_token_id = 0
    # processor.tokenizer.padding_side = "left"


    # Prepare For LoRA
    # If whisper encoder was pruned, also add fc1/fc2 (whisper MLP) to LoRA targets
    lora_target_modules = args.lora_target_modules.split(",")
    if whisper_pruned:
        whisper_modules = ["fc1", "fc2", "out_proj"]
        for wm in whisper_modules:
            if wm not in lora_target_modules:
                lora_target_modules.append(wm)
        print(f"[INFO] Whisper encoder was pruned, LoRA targets extended: {lora_target_modules}")

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=args.lora_dropout,
        use_dora=True,
        bias="none",
        init_lora_weights="gaussian",
        task_type="CAUSAL_LM"
        # task_type=TaskType.ASR,
    )
    
    
    peft_model = get_peft_model(model.model, config)
    peft_model.print_trainable_parameters()

    

    DATASET_imda_part2 = "imda_part2_asr_test"
    DATASET_ID_TEST = "imda_part1_asr_test"

    DATASET_imda_part3_30s = "imda_part3_30s_asr_test"
    DATASET_imda_part4_30s = "imda_part4_30s_asr_test"

    DATASET_ID_TRAIN = "imda_part2_asr_test"
    DATASET_ID_EVAL = "imda_part1_asr_test"
    nsamples = -1
    MAX_SEQUENCE_LENGTH = 256
    # Load dataset and preprocess.
    # calib_ds = Dataset(DATASET_ID_CALIB, nsamples)
    
    # # ds_30 = []
    # # for d in calib_ds.input_data:
    # #     audio_array    = d["audio"]["array"]
    # #     sampling_rate  = d["audio"]["sampling_rate"]
    # #     instruction    = d["instruction"]
    # #     audio_duration = len(audio_array) / sampling_rate

    # #     # For ASR task, if audio duration is more than 30 seconds, we will chunk and infer separately
    # #     if audio_duration <= 30:
    # #         if audio_duration < 1:
    # #             print('Audio duration is less than 1 second. Padding the audio to 1 second.')
    # #             d["audio"]["array"] = np.pad(audio_array, (0, sampling_rate), 'constant')
    # #         ds_30.append(copy.deepcopy(d))
    # # train_dataset = ds_30[:2500]
    # # eval_dataset = ds_30[2500:]
    # # test_dataset = ds_30[2500:]
    
    # # calib_ds.raw_data = calib_ds.raw_data.shuffle(seed=42) #.select(range(NUM_CALIBRATION_SAMPLES))
    # ds_split = calib_ds.raw_data.train_test_split(test_size=0.2, seed=42)
    # train_dataset = ds_split["train"]
    # eval_dataset  = ds_split["test"]
    # test_dataset  = ds_split["test"]
    # # input(type(calib_ds.raw_data))
    # # train_dataset = calib_ds.raw_data[:2500]
    # # eval_dataset = calib_ds.raw_data[2500:]
    # # test_dataset = calib_ds.raw_data[2500:]
    
    # --- Cached dataset loading (avoids re-loading 2M+ samples every run) ---
    # Cache stores the RAW selected 20k train + 500 eval datasets (before .map()).
    # First process builds cache; other 7 wait via file lock, then load instantly.
    CACHE_DIR = "/home/jinchao/runtao/meralion_datasets/ASR/_cached"
    train_cache = os.path.join(CACHE_DIR, "train_20k_raw")
    eval_cache = os.path.join(CACHE_DIR, "eval_500_raw")
    cache_done_marker = os.path.join(CACHE_DIR, ".cache_done")
    lock_file = os.path.join(CACHE_DIR, ".cache_building.lock")

    def _build_raw_datasets():
        """Load full datasets, select subsets, save to cache."""
        print("[DATA] Loading IMDA_PART1 (2.2M samples)...")
        p1 = load_from_disk("/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR")
        print("[DATA] Loading IMDA_PART3...")
        p3 = load_from_disk("/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART3_conv_en_30_ASR")

        ds_list = [p1.shuffle(seed=42).select(range(10000)),
                   p3.shuffle(seed=42).select(range(10000))]
        _train = concatenate_datasets(ds_list).shuffle(seed=42)
        # Eval from PART1 (matches baseline evaluation for fair WER comparison)
        _eval = p1.shuffle(seed=42).select(range(10000, 10500))  # 500 samples after training split

        # Save cache
        os.makedirs(CACHE_DIR, exist_ok=True)
        print("[DATA] Saving cache to disk...")
        _train.save_to_disk(train_cache)
        _eval.save_to_disk(eval_cache)
        # Marker file signals cache is complete (atomic)
        with open(cache_done_marker, 'w') as f:
            f.write("done")
        print("[DATA] Cache saved. Subsequent runs will load instantly.")
        return _train, _eval

    if os.path.exists(cache_done_marker):
        print("[DATA] Loading cached datasets (instant)...")
        train_dataset = load_from_disk(train_cache)
        eval_dataset = load_from_disk(eval_cache)
    else:
        import fcntl
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(lock_file, 'w') as lf:
            try:
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Got lock — double-check cache (another process may have just finished)
                if os.path.exists(cache_done_marker):
                    print("[DATA] Cache appeared, loading...")
                    train_dataset = load_from_disk(train_cache)
                    eval_dataset = load_from_disk(eval_cache)
                else:
                    print("[DATA] Building cache from raw datasets (first run only)...")
                    train_dataset, eval_dataset = _build_raw_datasets()
                fcntl.flock(lf, fcntl.LOCK_UN)
            except BlockingIOError:
                # Another process is building cache — wait for it
                print("[DATA] Waiting for another process to build cache...")
                fcntl.flock(lf, fcntl.LOCK_EX)  # blocking wait
                fcntl.flock(lf, fcntl.LOCK_UN)
                print("[DATA] Cache ready, loading...")
                train_dataset = load_from_disk(train_cache)
                eval_dataset = load_from_disk(eval_cache)
    # eval_dataset_2 = Dataset("imda_part3_30s_asr_test", nsamples).raw_data.shuffle(seed=42).select(range(500))
    # eval_dataset_1 = Dataset("imda_part1_asr_test", nsamples).raw_data.shuffle(seed=42).select(range(500))
    # eval_dataset = concatenate_datasets([eval_dataset_1, eval_dataset_2]).shuffle(seed=42)

    # pdb.set_trace()
    
    def build_user_prompt(processor, instruction):
        prompt = "Instruction: {instruction} \nFollow the text instruction based on the following audio: <SpeechHere>"


        conversation = [
            {"role": "user", "content": prompt.format(instruction=instruction)}
        ]

        return processor.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True
        )

    def build_chat_prompt(processor, instruction, target_text):
        prompt = "Instruction: {instruction} \nFollow the text instruction based on the following audio: <SpeechHere>"


        conversation = [
            {"role": "user", "content": prompt.format(instruction=instruction)},
            {"role": "assistant", "content": target_text}
        ]

        return processor.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=False
        )
    
    def preprocess_keep_raw(sample):
        if "audio" in sample["context"].keys():
            
            audio_array = sample["context"]["audio"]["array"] # for official training dataset
            sampling_rate = sample["context"]["audio"]["sampling_rate"]
            instruction = sample["instruction"]["text"]
            # answer = sample["answer"]["text"]
            if sample["other_attributes"]["partition"] == "PART1":
                answer = "<Speaker1>: " + sample["other_attributes"]["Transcription"]
            elif sample["other_attributes"]["partition"] == "PART3":
                answer = sample["other_attributes"]["transcription"]
        elif "array" in sample["context"].keys():
            audio_array = sample["context"]["array"] 
            sampling_rate = sample["context"]["sampling_rate"]
            instruction = sample["instruction"]
            answer = sample["answer"]
        return {
            "audio_array": audio_array,
            "sampling_rate": sampling_rate,
            "instruction": instruction,
            "answer": answer,
        }

    # .map() in memory only — avoid writing huge Arrow cache files to disk
    train_ds = train_dataset.map(preprocess_keep_raw, remove_columns=train_dataset.column_names, keep_in_memory=True)
    eval_ds  = eval_dataset.map(preprocess_keep_raw,  remove_columns=eval_dataset.column_names, keep_in_memory=True)
    
    # Validation WER Monitoring
    wer_callback = WEREvalCallback(
        eval_dataset=eval_dataset, # use raw for original audio
        processor=processor,
        model_wrapper=model,
        log_file=os.path.join(args.output_dir, "validation_wer.jsonl"),
        eval_steps=100,
        num_samples=50
    )
    # test_ds  = test_dataset.map(preprocess_keep_raw,  remove_columns=test_dataset.column_names)

    def ensure_audio_np_1d(audio) -> np.ndarray:
        """
        Convert audio to np.ndarray float32 and ensure it's mono 1D.
        Accepts: list, np.ndarray, torch.Tensor
        """
        if isinstance(audio, list):
            audio = np.asarray(audio)
        elif hasattr(audio, "detach"):  # torch tensor
            audio = audio.detach().cpu().numpy()
        else:
            audio = np.asarray(audio)

        # handle (T,), (T, 1), (1, T), (T, C)
        if audio.ndim == 2:
            pass # input("dim 2 audio array")
            # squeeze singleton dim if present
            if 1 in audio.shape:
                audio = np.squeeze(audio)
            else:
                # average channels to mono
                # (T, C) -> mean over C; (C, T) -> mean over C
                if audio.shape[0] < audio.shape[1]:  # likely (C, T)
                    audio = audio.mean(axis=0)
                else:                                 # likely (T, C)
                    audio = audio.mean(axis=1)

        if audio.ndim != 1:
            raise ValueError(f"Audio must end up 1-D; got shape={audio.shape}")
        
        if len(audio) == 0:
            raise ValueError(f"Got empty audio array!")

        return audio.astype(np.float32, copy=False)


    class DataCollatorSpeechSeq2SeqWithPadding:
        def __init__(self, processor, ignore_index: int = -100):
            self.processor = processor
            self.tokenizer = processor.tokenizer
            self.ignore_index = ignore_index

        # def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        def __call__(self, features):
            # Build texts and audios per-sample
            chat_texts = []
            user_texts = []
            audios = []
            srs = []
            answers = []

            for f in features:
                instruction = f["instruction"]

                user_texts.append(build_user_prompt(self.processor, instruction))
                answers.append(f["answer"])

                audios.append(ensure_audio_np_1d(f["audio_array"]))
                srs.append(f["sampling_rate"])

            # If your dataset has consistent SR (usually 16k for whisper), enforce it here
            # Otherwise, you should resample beforehand.
            sampling_rate = srs[0]

            # Processor handles tokenization + audio feature extraction + padding
            prompt = self.processor(
                text=user_texts,
                audios=audios,
            )
            # keep any extra fields (e.g., audio features) to pass through to the model
            passthrough = {k: v for k, v in prompt.items()
                        if k not in ("input_ids", "attention_mask")}

            prompt_ids = prompt["input_ids"]           # [B, Lp]
            prompt_attn = prompt["attention_mask"]     # [B, Lp]
            B = prompt_ids.size(0)

            tok = self.processor.tokenizer
            # 2) Tokenize transcriptions WITHOUT padding; we'll pad after concatenation

            text_tok = tok(
                answers,
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=256,
                return_tensors=None,
            )
            text_ids_list = text_tok["input_ids"]

            # 3) Concatenate: input_ids = [PROMPT] + [TEXT]
            input_ids, attention_mask, labels = [], [], []
            for i in range(B):
                p_ids = prompt_ids[i].tolist()
                p_att = prompt_attn[i].tolist()
                t_ids = text_ids_list[i]


                ids  = p_ids + t_ids + [tok.eos_token_id]
                attn = p_att + [1] * (len(t_ids) + 1) 
                # labels: mask prompt tokens, learn only on text tokens
                lab  = [-100] * len(p_ids) + t_ids + [tok.eos_token_id]

                input_ids.append(ids)
                attention_mask.append(attn)
                labels.append(lab)

            # 4) Pad to max length in batch
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            max_len = max(len(x) for x in input_ids)

            def pad_to(seq, fill, L):
                return seq + [fill] * (L - len(seq))

            input_ids      = [pad_to(x, pad_id, max_len) for x in input_ids]
            attention_mask = [pad_to(x, 0,      max_len) for x in attention_mask]
            labels         = [pad_to(x, -100,   max_len) for x in labels]
            
            batch = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
            # 5) Include processor outputs needed by the model (e.g., audio features)
            for k, v in passthrough.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    batch[k] = v.to(torch.bfloat16)
                else:
                    batch[k] = v

            return batch
        
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
    


    
    # Select from ("seq2seq", "trainer", "SFT")
    # trainer_type = "seq2seq"
    trainer_type = "trainer"
    # trainer_type = "SFT"
    if trainer_type == "trainer":
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            label_names=["labels"],

            # Training length
            num_train_epochs=args.num_epochs,
            # warmup_ratio=0.05,
            # max_steps=800,
            # warmup_steps=20,
            # max_steps=5,
            # warmup_steps=2,

            # Batch sizes
            per_device_train_batch_size=args.micro_batch_size,
            per_device_eval_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,

            # Memory / efficiency
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},

            # Optimizer & LR schedule
            optim="adamw_torch_fused",
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            

            # Logging / evaluation / checkpointing
            # WER-based saving is handled by WEREvalCallback (best_model/ + latest_model/)
            # Trainer's own checkpointing is disabled to save disk space
            warmup_ratio=0.05,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="no",

            # Precision
            bf16=True,
            # tf32=True,

            # Stability
            max_grad_norm=0.3,

            # Reporting
            # report_to=["wandb"],

            # Multimodal SAFETY (critical)
            remove_unused_columns=False,
        )

        trainer = Trainer(
            # model=model.model,
            model=peft_model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,   # your Meralion2SFTCollator(processor)
            callbacks=[wer_callback]
        )
    elif trainer_type == "seq2seq":
        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            
            # Training length
            # num_train_epochs=args.num_epochs,
            # warmup_ratio=0.05,
            max_steps=300,
            warmup_steps=15,

            per_device_train_batch_size=args.micro_batch_size,
            per_device_eval_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,

            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},

            optim="adamw_torch_fused",          # if this errors, change to "adamw_torch"
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",

            logging_steps=10,

            eval_strategy="steps",
            eval_steps=50,

            save_strategy="no",
            # save_steps=50,

            # load_best_model_at_end=True,
            # metric_for_best_model="wer",
            # greater_is_better=False,

            bf16=True,
            # tf32=True,
            max_grad_norm=0.3,

            report_to=["wandb"],

            # CRITICAL: do not drop audio fields
            remove_unused_columns=False,

            # Seq2SeqTrainer-specific: enable generation during eval
            predict_with_generate=True,

            # For causal LM, this controls total generated length. You may prefer max_new_tokens via generation_config.
            # generation_max_length=256,

            # optional: speed up generation if supported
            generation_num_beams=1,
        )
        peft_model.generation_config.max_new_tokens = 256

        def compute_metrics(pred):
            pred_ids = pred.predictions
            label_ids = pred.label_ids
            # replace -100 with the original pad_token_id (since we had replaced padding in the DataCollator above) 
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            # convert the predictions and labels back into human readable text, ignoring special tokens 
            pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            wer = 100 * metric.compute(predictions=pred_str, references=label_str)
            return {"wer": wer}

        trainer = Seq2SeqTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            tokenizer=processor.tokenizer,   # helps decoding/logging
            compute_metrics=compute_metrics,
        )

    else:
        class DelayedEvalCallback(TrainerCallback):
            def __init__(self, start_step: int):
                self.start_step = start_step

            def on_step_end(self, args, state, control, **kwargs):
                # Disable evaluation until start_step
                if state.global_step < self.start_step:
                    control.should_evaluate = False
                return control
        
        training_args = SFTConfig(
            output_dir=args.output_dir,  # Directory to save the model
            num_train_epochs=args.num_epochs,  # Number of training epochs
            per_device_train_batch_size=args.micro_batch_size,  # Batch size for training
            per_device_eval_batch_size=args.micro_batch_size,  # Batch size for evaluation
            gradient_accumulation_steps=gradient_accumulation_steps,  # Steps to accumulate gradients
            gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
            # Optimizer and scheduler settings
            optim="adamw_torch_fused",  # Optimizer type
            learning_rate=args.learning_rate,  # Learning rate for training
            lr_scheduler_type="cosine",  # Type of learning rate scheduler
            # label_smoothing_factor = 0.1, # new: # helps avoid collapse
            # Logging and evaluation
            logging_steps=50,  # Steps interval for logging
            eval_steps=100,  # Steps interval for evaluation
            eval_strategy="steps",  # Strategy for evaluation
            save_strategy="no",  # Strategy for saving the model
            # save_steps=1000,  # Steps interval for saving
            # save_total_limit=2,  # Keep only the last 2 checkpoints
            # metric_for_best_model="eval_loss",  # Metric to evaluate the best model
            # greater_is_better=False,  # Whether higher metric values are better
            # load_best_model_at_end=True,  # Load the best model after training
            # Mixed precision and gradient settings
            bf16=True,  # Use bfloat16 precision
            tf32=True,  # Use TensorFloat-32 precision
            max_grad_norm=0.3,  # Maximum norm for gradient clipping
            warmup_ratio=0.05,  # Ratio of total steps for warmup (ori: 0.03)
            # Hub and reporting
            report_to="wandb",  #
            # Reporting tool for tracking metrics
            # Gradient checkpointing settings
            gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
            # Dataset configuration
            # dataset_text_field="",  # Text field in dataset
            # dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
            # max_seq_length=1024  # Maximum sequence length for input
        )

        trainer = SFTTrainer(
            model=model.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            peft_config=config,
            # processing_class=processor.tokenizer,
            processing_class=processor,
            dataset_kwargs={"skip_prepare_dataset": True}, # SFTTrainer does not tokenize audio
            callbacks=[DelayedEvalCallback(start_step=750), wer_callback] # do not eval before # steps
        )

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Load the best WER model (saved by WEREvalCallback) instead of using the final step
    best_model_dir = os.path.join(args.output_dir, "best_model")
    if os.path.exists(best_model_dir):
        print(f"\n[INFO] Loading best WER model from {best_model_dir}")
        from peft import set_peft_model_state_dict
        # Load best LoRA weights via state_dict (safer than load_adapter for existing "default")
        adapter_path = os.path.join(best_model_dir, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            import safetensors.torch
            adapter_state = safetensors.torch.load_file(adapter_path)
        else:
            adapter_path = os.path.join(best_model_dir, "adapter_model.bin")
            adapter_state = torch.load(adapter_path, map_location="cpu")
        set_peft_model_state_dict(peft_model, adapter_state)
        if os.path.exists(os.path.join(best_model_dir, "best_step.txt")):
            print(open(os.path.join(best_model_dir, "best_step.txt")).read())
    else:
        print("[WARN] No best WER model found, using final step model")

    # Save final model (best WER version)
    trainer.save_model(args.output_dir)

    # Final WER evaluation on PART1 test set (local, no internet needed)
    peft_model.eval()
    wer_metric = evaluate.load("wer")

    print("\n" + "="*50)
    print("Final WER Evaluation on IMDA PART1")
    print("="*50)

    test_data = load_from_disk("/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR")
    # Use samples 10500+ to avoid overlap with train (0-10000) and val (10000-10500)
    test_subset = test_data.shuffle(seed=42).select(range(10500, 11000))

    predictions = []
    references = []

    st = time.time()
    with torch.no_grad():
        for sample in tqdm.tqdm(test_subset, desc="Final WER eval"):
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
            device = next(peft_model.parameters()).device
            for k in inputs:
                if isinstance(inputs[k], torch.Tensor):
                    inputs[k] = inputs[k].to(device)
                    if inputs[k].dtype == torch.float32:
                        inputs[k] = inputs[k].to(torch.bfloat16)

            model_outputs = peft_model.generate(**inputs, max_new_tokens=256, do_sample=False, num_beams=1)
            input_len = inputs['input_ids'].shape[1]
            generated_ids = model_outputs[:, input_len:]
            pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            pred = pred.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()

            predictions.append(pred)
            references.append(ref)

    et = time.time()
    final_wer = wer_metric.compute(predictions=predictions, references=references)
    print(f"\n500 samples took: {et-st:.1f}s")
    print(f"Final Test WER (IMDA PART1): {final_wer:.4f}")
    print("="*50)

    # Save results
    results_dir = os.path.join(args.output_dir, "final_eval")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "final_wer.json"), 'w') as f:
        json.dump({
            "wer": final_wer,
            "num_samples": len(predictions),
            "dataset": "IMDA_PART1_mono_en_30_ASR",
            "base_model": args.base_model,
        }, f, indent=2)

    # Save per-sample details
    details = [{"prediction": p, "reference": r} for p, r in zip(predictions[:20], references[:20])]
    with open(os.path.join(results_dir, "final_wer_samples.json"), 'w') as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

    from LLMPruner.evaluator.ppl import PPLMetric
    model.half()
    # ppl = PPLMetric(model, processor.tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device="cuda")
    ppl = PPLMetric(model, processor.tokenizer, ["c4"], args.max_seq_len, device="cuda") 
    print("PPL after pruning: {}".format(ppl))
    print("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')   

    # Model Type&Path
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--prune_model', type=str, help='prune model name')
    parser.add_argument('--data_path', type=str, default='HuggingFaceM4/ChartQA', help='data path')
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

    # argument for layer-wise pruning/column-wise pruning
    parser.add_argument('--whisper_block_layer_start', type=int, help='start layer of whisper layers', default=-1)
    parser.add_argument('--whisper_block_layer_end', type=int, help='end layer of whisper layers', default=-1)

    # Lora Configuration
    parser.add_argument('--lora_r', type=int, default=8, help='lora r')
    parser.add_argument('--lora_alpha', type=int, default=16, help='lora alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    parser.add_argument('--lora_target_modules', type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj", help='lora target modules')
    # parser.add_argument('--lora_target_modules', type=str, default="gate_proj,down_proj,up_proj", help='lora target modules')

    # llm hyperparameters
    parser.add_argument('--train_on_inputs', default=False, action="store_true", help='Train on inputs. If False, masks out inputs in loss')
    parser.add_argument('--add_eos_token', default=False, action="store_true")
    parser.add_argument('--group_by_length', default=False, action="store_true", help="faster, but produces an odd training loss curve")
   
    # wandb params
    parser.add_argument('--wandb_project', type=str, default="")
    parser.add_argument('--wandb_run_name', type=str, default="")
    parser.add_argument('--resume_from_checkpoint', type=str, help="either training checkpoint or final adapter")

    #ddp
    parser.add_argument('--local_rank', type=int, default=-1)

    # PPL
    parser.add_argument('--max_seq_len', type=int, default=256)
   
    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version

    main(args)
