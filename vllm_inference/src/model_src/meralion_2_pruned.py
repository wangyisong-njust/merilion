"""
HuggingFace-based inference for pruned MERaLiON-2 models.
vLLM cannot load pruned models with non-uniform layer dimensions (midblock system).
This loader uses HuggingFace model.generate() which handles resize_to_match() correctly.
"""

import os
import sys
sys.path.append('.')
sys.path.append('../')
import logging
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def meralion_2_pruned_model_loader(self):
    # Strip "-pruned" suffix to get actual model directory name
    actual_name = self.model_name.replace("-pruned", "")
    model_path = actual_name
    if not os.path.exists(model_path):
        possible_paths = [
            os.path.join("/home/jinchao/runtao/LLM-Pruner/meralion_checkpoints", actual_name),
            os.path.join("/home/jinchao/runtao/LLM_base_model", actual_name),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                model_path = p
                break

    logger.info(f"Loading pruned model (HF) from: {model_path}")

    self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        use_safetensors=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )

    logger.info(f"Pruned model loaded: {model_path}")


def do_sample_inference(self, audio_array, instruction):
    prompt = "Instruction: {instruction} \nFollow the text instruction based on the following audio: <SpeechHere>"
    conversation = [
        {"role": "user", "content": prompt.format(instruction=instruction)}
    ]

    chat_prompt = self.processor.tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = self.processor(text=chat_prompt, audios=audio_array, return_tensors="pt")

    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to('cuda')
            if inputs[key].dtype == torch.float32:
                inputs[key] = inputs[key].to(torch.bfloat16)

    model_outputs = self.model.generate(**inputs, max_new_tokens=256, do_sample=False, num_beams=1)
    generated_ids = model_outputs[:, inputs['input_ids'].size(1):]
    response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = response.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()

    return response


def meralion_2_pruned_model_generation(self, input):
    audio_array = input["audio"]["array"]
    sampling_rate = input["audio"]["sampling_rate"]
    instruction = input["instruction"]
    audio_duration = len(audio_array) / sampling_rate

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
