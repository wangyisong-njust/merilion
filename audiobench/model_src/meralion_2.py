import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor
from typing import List, Optional, Tuple, Union
import logging
import os

logger = logging.getLogger(__name__)

def meralion_2_model_loader(self):
    from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration
    from meralion2_bl.configuration_meralion2 import MERaLiON2Config
    
    logger.info(f"Loading model from: {self.model_name}")
    
    # No device_map="auto": incompatible with DDP (torchrun assigns each
    # process its own GPU via LOCAL_RANK; device_map would override that).
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    self.model = MERaLiON2ForConditionalGeneration.from_pretrained(
        self.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(f"cuda:{local_rank}")
    self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
    logger.info(f"Model loaded: {self.model_name}")

def do_sample_get_loss(self, audio_array, instruction, reference):
    prompt = "Instruction: {instruction} \nFollow the text instruction based on the following audio: <SpeechHere>"
    conversation = [{"role": "user", "content": prompt.format(instruction=instruction)}]
    chat_prompt = self.processor.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    
    inputs = self.processor(text=chat_prompt, audios=audio_array, return_tensors="pt").to("cuda")
    labels = self.processor(text=reference, return_tensors="pt").input_ids.to("cuda")
    
    outputs = self.model(**inputs, labels=labels)
    return outputs.loss

def meralion_2_model_get_loss(self, input):
    audio_array = input["audio"]["array"]
    sampling_rate = input["audio"]["sampling_rate"]
    instruction = input["instruction"]
    reference = input['reference']
    audio_duration = len(audio_array) / sampling_rate

    if audio_duration > 30:
        return do_sample_get_loss(self, audio_array[:30 * sampling_rate], instruction, reference)
    else:
        if audio_duration < 1:
            audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')
        return do_sample_get_loss(self, audio_array, instruction, reference)

def do_sample_get_inputs(self, audio_array, instruction):
    prompt = "Instruction: {instruction} \nFollow the text instruction based on the following audio: <SpeechHere>"
    conversation = [{"role": "user", "content": prompt.format(instruction=instruction)}]
    chat_prompt = self.processor.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    
    inputs = self.processor(text=chat_prompt, audios=audio_array, return_tensors="pt")
    for k in inputs:
        if isinstance(inputs[k], torch.Tensor):
            inputs[k] = inputs[k].to("cuda")
            if inputs[k].dtype == torch.float32:
                inputs[k] = inputs[k].to(torch.bfloat16)
    return inputs

def meralion_2_model_get_inputs(self, input):
    audio_array = input["audio"]["array"]
    sampling_rate = input["audio"]["sampling_rate"]
    instruction = input["instruction"]
    audio_duration = len(audio_array) / sampling_rate

    if audio_duration > 30:
        return do_sample_get_inputs(self, audio_array[:30 * sampling_rate], instruction)
    else:
        if audio_duration < 1:
            audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')
        return do_sample_get_inputs(self, audio_array, instruction)

def meralion_2_model_generation(self, input):
    inputs = meralion_2_model_get_inputs(self, input)
    with torch.no_grad():
        model_outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
        )
    input_len = inputs['input_ids'].shape[1]
    generated_ids = model_outputs[:, input_len:]
    response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = response.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()
    return response
