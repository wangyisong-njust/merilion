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

repo_id = "/home/jinchao/runtao/LLM_base_model/MERaLiON-2-10B-ASR"
# "wer": 0.043449637919684
# Inference took: 1535.3745684623718 s
# model weights take 18.83GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 12.89GiB.
# Inference took: 1545.7122983932495 s

repo_id = "/home/jinchao/runtao/meralion_test/saved_meralion/MERaLiON-2-10B-ASR-W8A16-G256-damp05-full"
# model weights take 11.12GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 20.60GiB.
# Inference took: 667.2631313800812 s

# repo_id = "/home/jinchao/runtao/meralion_test/saved_meralion/MERaLiON-2-10B-ASR-W4A16-G256-damp05-full"
# model weights take 7.37GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 24.35GiB.
# Inference took: 951.8644194602966 s

repo_id = "/home/jinchao/runtao/meralion_test/saved_meralion/MERaLiON-2-10B-ASR-W8A16-RTN-textonly"
# "wer": 0.04332772535537512
# Inference took: 1152.133632183075 s

# FP8 KV cache
# "wer": 0.04347402043254578
# Inference took: 1200.5482964515686 s

# nvidia-smi: 33909MiB
# the current vLLM instance can use total_gpu_memory (39.49GiB) x gpu_memory_utilization (0.90) = 35.55GiB
# model weights take 11.12GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 20.60GiB.


repo_id = "/home/jinchao/runtao/meralion_test/saved_meralion/MERaLiON-2-10B-ASR-W8A16-G256-damp05-textonly"
#  "wer": 0.043547167971131104
# Inference took: 1154.338224887848 s

# FP8 KV cache
# "wer": 0.043449637919684
# Inference took: 1194.9065828323364 s

repo_id = "/home/jinchao/runtao/meralion_test/saved_meralion/MERaLiON-2-10B-ASR-W4A16-G256-damp05-textonly"
# "wer": 0.06441859898081097
# Inference took: 984.3337061405182 s

# FP8 KV cache
# "wer": 0.06424792139077853
# Inference took: 986.0367386341095 s


repo_id = "/home/jinchao/runtao/meralion_test/saved_meralion/MERaLiON-2-10B-ASR-W4A16-RTN-textonly"
# Inference took: 975.2079074382782 s
# "wer": 0.06198034769463341

# FP8 KV cache
# Inference took: 974.6922211647034 s
# "wer": 0.06083436959012996
# the current vLLM instance can use total_gpu_memory (39.49GiB) x gpu_memory_utilization (0.90) = 35.55GiB
# model weights take 7.37GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 24.35GiB.

# ---------- Pruned ------------
repo_id = "/home/jinchao/runtao/LLM_base_model/MERaLiON-2-10B-ASR-0_5-5-40-merged"

# model weights take 13.84GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 17.88GiB.


repo_id = "/home/jinchao/runtao/LLM_base_model/MERaLiON-2-10B-ASR-0_25-7-35-merged"

# model weights take 16.86GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 14.86GiB.

# output would use up max num of tokens and is very slow


repo_id = "/home/jinchao/runtao/LLM_base_model/MERaLiON-2-10B-ASR-0_5-5-40"
# Inference took: 1173.5332446098328 s
# "wer": 0.07163582278789651
# model weights take 13.84GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 17.88GiB.


repo_id = "/home/jinchao/runtao/LLM_base_model/MERaLiON-2-10B-ASR-0_25-7-35"
# Inference took: 1433.9808933734894 s
# "wer": 0.04679004218174725
# model weights take 16.86GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 14.86GiB


repo_id = "/home/jinchao/runtao/meralion_test/saved_meralion/MERaLiON-2-10B-ASR-0_25-7-35-W4A16-RTN-textonly"
# Inference took: 922.2395508289337 s
# "wer": 0.1438080608587521
# model weights take 6.83GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 24.89GiB.

repo_id = "/home/jinchao/runtao/meralion_test/saved_meralion/MERaLiON-2-10B-ASR-0_25-7-35-W8A16-RTN-textonly"
# Inference took: 1077.1056673526764 s
# "wer": 0.05130080706117573
# model weights take 10.10GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 21.62GiB.


repo_id = "/home/jinchao/runtao/LLM-Pruner/MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2-merged"
# finetune for 5 steps
# Inference took: 1052.6886668205261 s
# "wer": 0.14514909906614976
# finetune for 100 steps：overfit on 30s dataset, may recognize multiple speakers and use up max gen token
# model weights take 13.84GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 17.88GiB.


repo_id = "/home/jinchao/runtao/LLM-Pruner/MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2-imda2-merged"
# finetune for 200 steps
# repeat the sentence for multiple times

repo_id = "/home/jinchao/runtao/LLM-Pruner/MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2-imda2-5e-5-merged"
# data collator from github project replacing audio encoder
# finetune for 100 steps
# Inference took: 968.4088923931122 s
# "wer": 0.06032233682003267

repo_id = "./saved_meralion/MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2-imda2-5e-5-merged-W8A16-RTN-textonly"

# repo_id = "/home/jinchao/runtao/LLM-Pruner/MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2-imda2-4e-4-merged"
# finetune for 100 steps
# Inference took: 783.5632629394531 s
# "wer": 0.06046863189720333
# model weights take 8.61GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 23.11GiB.


repo_id = "/home/jinchao/runtao/LLM-Pruner/MERaLiON-2-10B-ASR-0_25-7-35-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged"
# finetune for 300 steps
# Inference took: 1071.8222706317902 s
# "wer": 0.04791163777338892

repo_id = "./saved_meralion/MERaLiON-2-10B-ASR-0_25-7-35-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-W8A16-RTN-textonly"
# Inference took: 837.2072851657867 s
# "wer": 0.047740960183356496
# Inference took: 834.0932602882385 s
# "wer": 0.04805793285055958

repo_id = "./saved_meralion/MERaLiON-2-10B-ASR-0_25-7-35-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-W4A16-RTN-textonly"
# Inference took: 731.2834343910217 s
# "wer": 0.050154828956672275

repo_id = "/home/jinchao/runtao/LLM-Pruner/MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2"
# Inference took: 960.1988291740417 s
# "wer": 0.05768902543096092


repo_id = "./saved_meralion/MERaLiON-2-10B-ASR-0_5-5-40-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-W8A16-RTN-textonly"
# Inference took: 774.3419942855835 s
# "wer": 0.057518347840928484

repo_id = "/home/jinchao/runtao/LLM-Pruner/MERaLiON-2-10B-ASR-0_25-7-35-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-new"
# Inference took: 1067.4837272167206 s
# "wer": 0.048228610440592005

repo_id = "./saved_meralion/MERaLiON-2-10B-ASR-0_25-7-35-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-new-W8A16-RTN-textonly"
# Inference took: 822.7822391986847 s
# "wer": 0.04852120059493331

repo_id = "./saved_meralion/MERaLiON-2-10B-ASR-0_25-7-35-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-new-W4A16-RTN-textonly"
# Inference took: 717.8974680900574 s
# "wer": 0.0502279764952576


# 3B MODEL
repo_id = "/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B"
# Inference took: 743.5066623687744 s
# "wer": 0.04888693828785995
# "wer": 0.04881379074927462
# model weights take 6.50GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 25.22GiB.


# repo_id = "./saved_meralion/MERaLiON-2-3B-W8A16-RTN-textonly"
# Inference took: 645.8711411952972 s
# "wer": 0.049057615877892376
# "wer": 0.04908199839075415
# model weights take 4.59GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 27.13GiB.


# repo_id = "./saved_meralion/MERaLiON-2-3B-W4A16-RTN-textonly"
# Inference took: 
# "wer": 0.05078877429107844
# model weights take 3.69GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 28.03GiB.


# repo_id = "/home/jinchao/runtao/LLM-Pruner/MERaLiON-2-3B-0_25-3-23-tuned-r32-full_gemma2-mix-1e-5-grad_accu_2-dropout01-merged"
# Inference took: 553.35094165802 s
# "wer": 0.05283690537146758
# "wer": 0.05269061029429693
# model weights take 5.88GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 25.84GiB.


# repo_id = "./saved_meralion/MERaLiON-2-3B-0_25-3-23-tuned-r32-full_gemma2-mix-1e-5-grad_accu_2-dropout01-merged-W8A16-RTN-textonly"
# Inference took: 506.1559851169586 s
# "wer": 0.05317826055153244
# "wer": 0.05332455562870309
# model weights take 4.30GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 27.42GiB.


# repo_id = "./saved_meralion/MERaLiON-2-3B-0_25-3-23-tuned-r32-full_gemma2-mix-1e-5-grad_accu_2-dropout01-merged-W4A16-RTN-textonly"
# Inference took: 496.08927035331726 s
# "wer": 0.058688708458293715
# "wer": 0.07743886084899909 ？
# model weights take 3.54GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 28.18GiB.


# repo_id = "/home/jinchao/runtao/LLM-Pruner/MERaLiON-2-3B-0_5-3-23-tuned-r16-full_gemma2-mix-1e-5-grad_accu_2-dropout01-merged"
# Inference took: 533.3116009235382 s
# "wer": 0.06193158266890986
# model weights take 5.29GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 26.43GiB.

# repo_id = "./saved_meralion/MERaLiON-2-3B-0_5-3-23-tuned-r16-full_gemma2-mix-1e-5-grad_accu_2-dropout01-merged-W8A16-RTN-textonly"
# Inference took: 488.4539806842804 s
# "wer": 0.061834052617462755
# model weights take 4.01GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 27.71GiB.

repo_id = "/home/jinchao/runtao/LLM-Pruner/meralion_checkpoints/MERaLiON-2-3B-0_25-4-23-both"
# model weights take 5.80GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 25.92GiB.
# Inference took: 617.4014122486115 s
# "wer": 0.1600711969375564

repo_id = "/home/jinchao/runtao/LLM-Pruner/MERaLiON-2-3B-0_25-4-23-both-tuned-r16-a16-5e-6-bs8-imda1m3c-merged"
# Inference took: 638.2531740665436 s
# "wer": 0.054592446297515425

repo_id = "./saved_meralion/MERaLiON-2-3B-0_25-4-23-both-tuned-r16-a16-5e-6-bs8-imda1m3c-merged-W8A16-RTN-textonly"
# model weights take 4.25GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 27.47GiB.
# Inference took: 574.1968142986298 s
# "wer": 0.05476312388754785

repo_id = "./saved_meralion/MERaLiON-2-3B-0_25-4-23-both-tuned-r16-a16-5e-6-bs8-imda1m3c-merged-W4A16-RTN-textonly"
# model weights take 3.51GiB; non_torch_memory takes 0.09GiB; 
# PyTorch activation peak memory takes 3.73GiB; the rest of the memory reserved for KV Cache is 28.21GiB.
# Inference took: 554.1442537307739 s
# "wer": 0.05803038061102577

def meralion_2_model_loader(self):
    model_path = self.model_name
    if not os.path.exists(model_path):
        # Fallback to check in specific directories if not an absolute path
        possible_paths = [
            os.path.join("/home/jinchao/runtao/LLM-Pruner/meralion_checkpoints", self.model_name),
            os.path.join("/home/jinchao/runtao/LLM_base_model", self.model_name),
            os.path.join("/home/jinchao/runtao/meralion_test/saved_meralion", self.model_name),
            repo_id # fallback to the last hardcoded repo_id
        ]
        for p in possible_paths:
            if os.path.exists(p):
                model_path = p
                break
    
    logger.info(f"Loading model from: {model_path}")
    self.model = LLM(model=model_path,
                tokenizer=model_path,
                limit_mm_per_prompt={"audio": 1},
                trust_remote_code=True,
                )
    logger.info("Model loaded: {}".format(model_path))



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
        # response = re.sub(r'^(?:<[^>]*>\s*)+(?:(?::|\r?\n)\s*)?', '', response).removesuffix("\n") # remove sth like "<><>: " or "<><>\n"
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

