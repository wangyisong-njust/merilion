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

from vllm import LLM, SamplingParams
from vllm_plugin_meralion2 import NoRepeatNGramLogitsProcessor

import tempfile


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

    self.model = LLM(model=repo_id,
                tokenizer=repo_id,
                limit_mm_per_prompt={"audio": 1},
                trust_remote_code=True,
                dtype=torch.bfloat16
                )
    # self.model.to("cuda")

    logger.info("Model loaded: {}".format(repo_id))



def do_sample_inference(self, audio_array, sampling_rate, instruction):

    prompt = (
        "<start_of_turn>user\n"
        f"Instruction: {instruction} \nFollow the text instruction based on the following audio: <SpeechHere><end_of_turn>\n"
        "<start_of_turn>model\n")
    
    sampling_params = SamplingParams(
                        temperature=0.0,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.0,
                        seed=42,
                        max_tokens=1024,
                        stop_token_ids=None,
                        logits_processors=[NoRepeatNGramLogitsProcessor(6)]
                        )

    mm_data = {"audio": [(audio_array, sampling_rate)]}
    inputs = {"prompt": prompt, "multi_modal_data": mm_data}

    # batch inference
    inputs = [inputs] * 2

    outputs = self.model.generate(inputs, sampling_params=sampling_params)

    for o in outputs:
        response = o.outputs[0].text
        response = response.removeprefix("<Speaker1>: ").removesuffix("\n")
        print(response)
        break
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
        
        model_predictions = [do_sample_inference(self, chunk_array, sampling_rate, instruction) for chunk_array in tqdm(audio_chunks)]
        output = ' '.join(model_predictions)


    elif audio_duration > 30:
        logger.info('Audio duration is more than 30 seconds. Taking first 30 seconds.')

        audio_array = audio_array[:30 * sampling_rate]
        output = do_sample_inference(self, audio_array, sampling_rate, instruction)
    
    else: 
        if audio_duration < 1:
            logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
            audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

        output = do_sample_inference(self, audio_array, sampling_rate, instruction)

    return output

