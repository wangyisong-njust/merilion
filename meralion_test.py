import os
import sys
import argparse
from typing import List
from pathlib import Path

import torch
import transformers
from datasets import load_dataset

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration

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

model_dir = "MERaLiON-2-10B-ASR-0_25-7-35-merged"

processor = AutoProcessor.from_pretrained(
    model_dir, 
    trust_remote_code=True,
)
model = MERaLiON2ForConditionalGeneration.from_pretrained(
    model_dir,
    use_safetensors=True,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    # attn_implementation="sdpa",
    torch_dtype=torch.bfloat16
).to("cuda")
model.eval()


DATASET_ID_CALIB = "imda_part2_asr_test"
DATASET_ID_TEST = "imda_part1_asr_test"
MAX_SEQUENCE_LENGTH = 256
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
input("inference ended")

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
print('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =')
print('Dataset name: {}'.format(dataset_name.upper()))
print('Model name: {}'.format(model_name.upper()))
print(json.dumps({metrics: results[metrics]}, indent=4, ensure_ascii=False))
print('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =')

# Save the scores
with open(f'{file_save_folder}/{model_name}/{dataset_name}_{metrics}_score.json', 'w') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
input("Eval complete")