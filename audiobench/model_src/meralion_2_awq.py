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

from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier, AWQ_MAPPING_REGISTRY
from dataset import Dataset
from dataset_src.prompts.prompts import asr_instructions
import random

import pdb


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

repo_id = "/home/kaixin/programs/LLM_base_model/MERaLiON-2-10B-ASR"

def meralion_2_model_loader(self):

    self.processor = AutoProcessor.from_pretrained(
    repo_id, 
    trust_remote_code=True,
    )
    # self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
    #     repo_id,
    #     use_safetensors=True,
    #     trust_remote_code=True,
    #     attn_implementation="flash_attention_2",
    #     # attn_implementation="sdpa",
    #     torch_dtype=torch.bfloat16
    # )
    self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
        repo_id,
        use_safetensors=True,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        load_in_8bit=True,          # ← 8-bit quantization
        device_map="auto",          # automatically place on GPU(s)
        torch_dtype=torch.float16,  # recommended for 8-bit
    )
    
    # Perform AWQ
    
    # Select calibration dataset.
    DATASET_ID = "imda_part2_asr_test"
    # DATASET_SPLIT = "validation"

    # Select number of samples. 256 samples is a good place to start.
    # Increasing the number of samples can improve accuracy.
    NUM_CALIBRATION_SAMPLES = 128
    MAX_SEQUENCE_LENGTH = 256

    # Load dataset and preprocess.
    ds = Dataset(DATASET_ID, NUM_CALIBRATION_SAMPLES)
    # ds = ds.shuffle(seed=42)
    
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


    # def preprocess(example):
    #     return {
    #         "text": tokenizer.apply_chat_template(
    #             [{"role": "user", "content": example["text"]}],
    #             tokenize=False,
    #         )
    #     }
    
    def preprocess(example):
        audio       = example['context']
        reference   = example['answer']
        instruction = random.choice(asr_instructions)
        return {
                "audio"      : audio,
                "instruction": instruction,
                "reference"  : reference,
                "task_type"  : "ASR"
                }


    # ds = ds.input_data.map(preprocess)
    
    
    ds = ds.raw_data.map(preprocess)
    pdb.set_trace()
    
    # print(ds)
    # print(AWQ_MAPPING_REGISTRY.keys())


    # Configure the quantization algorithm to run.
    recipe = [
        AWQModifier(ignore=["speech_encoder","ln_speech","speech_audio_adapter"], \
            scheme="W4A16_ASYM", targets=["Linear"], \
            mappings=AWQ_MAPPING_REGISTRY["Gemma3ForConditionalGeneration"]),
    ]

    # Apply algorithms.
    oneshot(
        model=self.model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        trust_remote_code_model=True
    )
    
    # input(self.model)
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

