import os
import re

# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')
import logging
import numpy as np
import torch

from tqdm import tqdm

import soundfile as sf

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


import tempfile

from datasets import load_dataset, load_from_disk
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier, AWQ_MAPPING_REGISTRY
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from dataset import Dataset
from dataset_src.prompts.prompts import asr_instructions
import random

import pdb

from meralion2_bl_llmcompressor.modeling_meralion2 import MERaLiON2ForConditionalGeneration


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

# Select one
# QUANT_TYPE = "GPTQ"
QUANT_TYPE = "RTN"

repo_id = "/home/jinchao/runtao/LLM_base_model/MERaLiON-2-10B-ASR"
# repo_id = "/home/jinchao/runtao/LLM-Pruner/MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2-imda2-5e-5-merged"
# repo_id = "/home/jinchao/runtao/LLM-Pruner/MERaLiON-2-10B-ASR-0_25-7-35-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged"
repo_id = "/home/jinchao/runtao/LLM-Pruner/MERaLiON-2-10B-ASR-0_25-7-35-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-new"
# repo_id = "/home/jinchao/runtao/LLM-Pruner/MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged"


repo_id = "/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B"
repo_id = "/home/jinchao/runtao/LLM-Pruner/MERaLiON-2-3B-0_25-3-23-tuned-r32-full_gemma2-mix-1e-5-grad_accu_2-dropout01-merged"
repo_id = "/home/jinchao/runtao/LLM-Pruner/MERaLiON-2-3B-0_5-3-23-tuned-r16-full_gemma2-mix-1e-5-grad_accu_2-dropout01-merged"


repo_id = "/home/jinchao/runtao/LLM-Pruner/MERaLiON-2-3B-0_25-4-23-both-tuned-r16-a16-5e-6-bs8-imda1m3c-merged"

def meralion_2_model_loader(self):

    self.processor = AutoProcessor.from_pretrained(
        repo_id, 
        trust_remote_code=True,
    )
    self.model = MERaLiON2ForConditionalGeneration.from_pretrained(
        repo_id,
        use_safetensors=True,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",
        # attn_implementation="sdpa",
        # torch_dtype="auto",
        torch_dtype=torch.float16,
        # torch_dtype=torch.bfloat16
        low_cpu_mem_usage=False,   # ← CRITICAL
    )
    self.model.cuda()
    self.model.eval()
    
    # self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
    #     repo_id,
    #     use_safetensors=True,
    #     attn_implementation="flash_attention_2",
    #     trust_remote_code=True,
    #     load_in_8bit=True,          # ← 8-bit quantization
    #     device_map="auto",          # automatically place on GPU(s)
    #     torch_dtype=torch.float16,  # recommended for 8-bit
    # )
    
    
    # Select calibration dataset.
    NUM_CALIBRATION_SAMPLES = 256
    MAX_SEQUENCE_LENGTH = 256
    # DATASET_ID = "imda_part2_asr_test"
    # # DATASET_SPLIT = "validation"

    # # Select number of samples. 256 samples is a good place to start.
    # # Increasing the number of samples can improve accuracy.
    
    # old code: can be run, but performs worse than RTN
    # # Load dataset and preprocess.
    # ds = Dataset(DATASET_ID, NUM_CALIBRATION_SAMPLES)
    # # ds = ds.shuffle(seed=42)
    # # pdb.set_trace()
    # ds.raw_data = ds.raw_data.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

    # def build_chat_prompt(processor, instruction):
    #     prompt = "Instruction: {instruction} \nFollow the text instruction based on the following audio: <SpeechHere>"


    #     conversation = [
    #         {"role": "user", "content": prompt.format(instruction=instruction)}
    #     ]

    #     return processor.tokenizer.apply_chat_template(
    #         conversation=conversation,
    #         tokenize=False,
    #         add_generation_prompt=True
    #     )
    
    # # Process inputs.
    # def process(sample):
    #     audio_array = sample['context']["array"]
    #     sampling_rate = sample['context']["sampling_rate"]
    #     instruction = sample["instruction"]
    #     target_text = sample["answer"]
        
    #     chat_prompt = build_chat_prompt(self.processor, instruction)
        
    #     inputs = self.processor(
    #         text=chat_prompt,
    #         audios=audio_array,
    #         # sampling_rate=sample["sampling_rate"],
    #         # add_special_tokens=True,
    #         # return_tensors="pt",
    #     )
    #     # pdb.set_trace()

    #     # for key in inputs:
    #     #     # inputs[key] = inputs[key].to('cuda').to(dtype=self.model.dtype)
    #     #     inputs[key] = inputs[key].to('cuda').to(dtype=torch.float16)
    #     # inputs["input_features"] = inputs["input_features"].to(dtype=self.model.dtype)
    #     inputs["input_features"] = inputs["input_features"].to(dtype=self.model.dtype)
    #     # inputs["decoder_input_ids"] = inputs["labels"]
    #     # del inputs["labels"]

    #     return inputs

    # ds = ds.raw_data.map(process, remove_columns=ds.raw_data.column_names)

    # new code: still got errors
    IMDA_PART1_mono_en_30_ASR = load_from_disk("/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR") # 2258301 samples
    # IMDA_PART3_conv_en_30_ASR = load_from_disk("/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART3_conv_en_30_ASR")
    ds = IMDA_PART1_mono_en_30_ASR.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))
    
    if QUANT_TYPE == "GPTQ":
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
        
        # Process inputs.
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
            # audio_array = np.asarray(audio_array, dtype=np.float32)
            return {
                "audio_array": audio_array,
                "sampling_rate": sampling_rate,
                "instruction": instruction,
                "answer": answer,
                "text": "Instruction: {instruction} \nFollow the text instruction based on the following audio: <SpeechHere>"
                # "text": build_user_prompt(self.processor, instruction)
            }

        ds = ds.map(preprocess_keep_raw, remove_columns=ds.column_names)
        
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
                input("dim 2 audio array")
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
                    batch[k] = v

                return batch
            
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(self.processor)

        def llmcompressor_preprocess(example):
            # Build the text prompt exactly like your collator does (or just use instruction).
            # IMPORTANT: return keys that match MERaLiON2Processor.__call__ kwargs.
            return {
                "text": build_user_prompt(self.processor, example["instruction"]),          # or your templated prompt string
                "audios": example["audio_array"],        # your float array
                # "sampling_rate": example["sampling_rate"]
            }


        # Configure the quantization algorithm to run.
        # recipe = [
        #     AWQModifier(ignore=["speech_encoder","ln_speech","speech_audio_adapter"], \
        #         scheme="W4A16_ASYM", targets=["Linear"], \
        #         mappings=AWQ_MAPPING_REGISTRY["Gemma3ForConditionalGeneration"]),
        # ]


        recipe = GPTQModifier(targets="Linear", scheme="W4A16", 
                            ignore=[
                                    # r"re:^speech_encoder\.",
                                    # "ln_speech",
                                    # r"re:^speech_audio_adapter\.",
                                    "text_decoder.lm_head",
                                    # skip attention projections
                                    # "text_decoder.model.layers.*.self_attn.q_proj",
                                    # "text_decoder.model.layers.*.self_attn.k_proj",
                                    # "text_decoder.model.layers.*.self_attn.v_proj",
                                    # "text_decoder.model.layers.*.self_attn.o_proj",
                                    ],
                            dampening_frac = 0.05, # default (e.g., 0.01)
                            #   per_token=True,
                            #   flatten_inputs=True,
                            #   reshape_inputs=True,
                            )

        oneshot(
            model=self.model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=MAX_SEQUENCE_LENGTH,
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
            preprocessing_func=llmcompressor_preprocess,
            data_collator=data_collator,
            trust_remote_code_model=True
        )
    
    elif QUANT_TYPE == "RTN":
        recipe = QuantizationModifier(
            targets="Linear",
            scheme="W4A16",
            ignore=[
                    r"re:^speech_encoder\.",
                    "ln_speech",
                    r"re:^speech_audio_adapter\.",
                    "text_decoder.lm_head",
                    ],
        )

        oneshot(model=self.model, recipe=recipe, trust_remote_code_model=True)

    # SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A16-G128"
    SAVE_DIR = "./saved_meralion/MERaLiON-2-10B-ASR-W4A16-G256-damp05-textonly"
    SAVE_DIR = "./saved_meralion/MERaLiON-2-10B-ASR-0_25-7-35-W8A16-RTN-textonly"
    SAVE_DIR = "./saved_meralion/MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2-imda2-5e-5-merged-W8A16-RTN-textonly"
    SAVE_DIR = "./saved_meralion/MERaLiON-2-10B-ASR-0_25-7-35-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-W4A16-RTN-textonly"
    SAVE_DIR = "./saved_meralion/MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-W8A16-RTN-textonly"
    SAVE_DIR = "./saved_meralion/MERaLiON-2-10B-ASR-W4A16-RTN-textonly"
    SAVE_DIR = "./saved_meralion/MERaLiON-2-10B-ASR-0_25-7-35-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-new-W8A16-RTN-textonly"
    SAVE_DIR = "./saved_meralion/MERaLiON-2-3B-W8A16-RTN-textonly"
    SAVE_DIR = "./saved_meralion/MERaLiON-2-3B-W4A16-RTN-textonly"
    SAVE_DIR = "./saved_meralion/MERaLiON-2-3B-0_25-3-23-tuned-r32-full_gemma2-mix-1e-5-grad_accu_2-dropout01-merged-W8A16-RTN-textonly"
    SAVE_DIR = "./saved_meralion/MERaLiON-2-3B-0_25-3-23-tuned-r32-full_gemma2-mix-1e-5-grad_accu_2-dropout01-merged-W4A16-RTN-textonly"
    SAVE_DIR = "./saved_meralion/MERaLiON-2-3B-0_5-3-23-tuned-r16-full_gemma2-mix-1e-5-grad_accu_2-dropout01-merged-W8A16-RTN-textonly"
    # SAVE_DIR = "./saved_meralion/MERaLiON-2-3B-0_5-3-23-tuned-r16-full_gemma2-mix-1e-5-grad_accu_2-dropout01-merged-W4A16-RTN-textonly"
    
    SAVE_DIR = "./saved_meralion/MERaLiON-2-3B-0_25-4-23-both-tuned-r16-a16-5e-6-bs8-imda1m3c-merged-W4A16-RTN-textonly"

    SAVE_DIR = "./saved_meralion/MERaLiON-2-3B-0_25-4-23-both-tuned-r16-a16-5e-6-bs8-imda1m3c-merged-W4A16-GPTQ"
    self.model.save_pretrained(SAVE_DIR, save_compressed=True)
    self.processor.save_pretrained(SAVE_DIR)
    
    input("Quantization completed")
    self.model.to("cuda")

    logger.info("Model loaded: {}".format(repo_id))



def do_sample_inference(self, audio_array, instruction):

    # prompt = "Given the following audio context: <SpeechHere>\n\nText instruction: {instruction}"
    prompt = "Instruction: {instruction} \nFollow the text instruction based on the following audio: <SpeechHere>"
    conversation = [
            {"role": "user", "content": prompt.format(instruction=instruction)}
        ]

    chat_prompt = self.processor.tokenizer.apply_chat_template(
                conversation          = conversation,
                tokenize              = False,
                add_generation_prompt = True
            )

    inputs = self.processor(text=chat_prompt, audios=audio_array)

    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to('cuda')
        if inputs[key].dtype is torch.float32:
            inputs[key] = inputs[key].to(torch.bfloat16)

    model_outputs = self.model.generate(**inputs, max_new_tokens=256)
    generated_ids = model_outputs[:, inputs['input_ids'].size(1):]
    response      = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = response.removeprefix("<Speaker1>: ").removesuffix("\n")

    return response


def meralion_2_model_generation(self, input):

    audio_array    = input["audio"]["array"]
    sampling_rate  = input["audio"]["sampling_rate"]
    instruction    = input["instruction"]
    audio_duration = len(audio_array) / sampling_rate

    # For ASR task, if audio duration is more than 30 seconds, we will chunk and infer separately
    if audio_duration > 30 and input['task_type'] == 'ASR':
        logger.info('Audio duration is more than 30 seconds. Chunking and inferring separately.')
        audio_chunks = []
        for i in range(0, len(audio_array), 30 * sampling_rate):
            audio_chunks.append(audio_array[i:i + 30 * sampling_rate])
        
        model_predictions = [do_sample_inference(self, chunk_array, instruction) for chunk_array in tqdm(audio_chunks)]
        output = ' '.join(model_predictions)


    elif audio_duration > 30:
        logger.info('Audio duration is more than 30 seconds. Taking first 30 seconds.')

        audio_array = audio_array[:30 * sampling_rate]
        output = do_sample_inference(self, audio_array, instruction)
    
    else: 
        if audio_duration < 1:
            logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
            audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

        output = do_sample_inference(self, audio_array, instruction)

    return output

