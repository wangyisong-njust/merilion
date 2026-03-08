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

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, BitsAndBytesConfig


import tempfile

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True


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
    if '8bit' in self.model_name:
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            repo_id,
            use_safetensors=True,
            # attn_implementation="flash_attention_2",
            trust_remote_code=True,
            load_in_8bit=True,          # ← 8-bit quantization
            device_map="auto",          # automatically place on GPU(s)
            torch_dtype=torch.float16,  # recommended for 8-bit
        )
    elif '4bit' in self.model_name:
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # )
        # self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
        #     repo_id,
        #     use_safetensors=True,
        #     # attn_implementation="flash_attention_2",
        #     trust_remote_code=True,
        #     device_map="auto",
        #     torch_dtype=torch.bfloat16,
        #     quantization_config=bnb_config
        # )
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            repo_id,
            use_safetensors=True,
            # attn_implementation="flash_attention_2",
            trust_remote_code=True,
            load_in_4bit=True,          # ← 8-bit quantization
            device_map="auto",          # automatically place on GPU(s)
            torch_dtype=torch.bfloat16,  # recommended for 8-bit
        )
        
        # vllm quantize
        
    else:
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            repo_id,
            use_safetensors=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            # attn_implementation="sdpa",
            device_map="cuda",
            torch_dtype=torch.bfloat16
        )
    
    # input(self.model)
    # self.model.to("cuda")

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
            if "8bit" in self.model_name:
                inputs[key] = inputs[key].to(torch.float16)
            else:
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

