"""
vLLM-based inference for pruned MERaLiON-2 models with non-uniform layer dimensions.

Uses the custom PrunedGemma2Model that handles per-layer attention/MLP sizes
from the midblock pruning config. This replaces the HF-based meralion_2_pruned.py
with vLLM for much faster inference.

Requirements:
  - vLLM v0.6.5+
  - vllm_plugin_meralion2 installed (for NoRepeatNGramLogitsProcessor)
  - pruned_gemma2_vllm.py and meralion2_vllm_pruned.py in Python path
"""
import os
import sys
import logging
import numpy as np

from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def meralion_2_vllm_pruned_model_loader(self):
    """Load a pruned MERaLiON-2 model using vLLM with PrunedGemma2Model."""
    import torch
    from vllm import LLM, SamplingParams, ModelRegistry

    # Add the pruned model code to path
    vllm_inference_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if vllm_inference_dir not in sys.path:
        sys.path.insert(0, vllm_inference_dir)

    # Register the pruned model class BEFORE creating LLM
    from meralion2_vllm_pruned import MERaLiON2PrunedForConditionalGeneration
    if "MERaLiON2ForConditionalGeneration" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "MERaLiON2ForConditionalGeneration",
            MERaLiON2PrunedForConditionalGeneration,
        )
    else:
        logger.info("MERaLiON2ForConditionalGeneration already registered, "
                     "replacing with pruned-aware version")
        ModelRegistry.register_model(
            "MERaLiON2ForConditionalGeneration",
            MERaLiON2PrunedForConditionalGeneration,
        )

    # Resolve model path
    model_path = self.model_name
    if not os.path.exists(model_path):
        possible_paths = [
            os.path.join("/home/jinchao/runtao/LLM-Pruner/meralion_checkpoints", self.model_name),
            os.path.join("/home/jinchao/runtao/LLM_base_model", self.model_name),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                model_path = p
                break

    logger.info(f"Loading pruned model via vLLM from: {model_path}")
    self.model = LLM(
        model=model_path,
        tokenizer=model_path,
        limit_mm_per_prompt={"audio": 1},
        trust_remote_code=True,
    )
    logger.info(f"Pruned model loaded via vLLM: {model_path}")


def do_sample_inference(self, audio_array, sampling_rate, instruction):
    from vllm import SamplingParams
    try:
        from vllm_plugin_meralion2 import NoRepeatNGramLogitsProcessor
        logits_processors = [NoRepeatNGramLogitsProcessor(6)]
    except ImportError:
        logits_processors = []

    prompt = (
        "<start_of_turn>user\n"
        f"Instruction: {instruction} \nFollow the text instruction based on the following audio: <SpeechHere><end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.0,
        seed=42,
        max_tokens=1024,
        stop_token_ids=None,
        logits_processors=logits_processors,
    )

    mm_data = {"audio": [(audio_array, sampling_rate)]}
    inputs = [{"prompt": prompt, "multi_modal_data": mm_data}]
    outputs = self.model.generate(inputs, sampling_params=sampling_params)

    response = outputs[0].outputs[0].text
    response = response.removeprefix("<Speaker1>: ").removesuffix("\n")
    return response


def meralion_2_vllm_pruned_model_generation(self, input):
    audio_array = input["audio"]["array"]
    sampling_rate = input["audio"]["sampling_rate"]
    instruction = input["instruction"]
    audio_duration = len(audio_array) / sampling_rate

    if audio_duration > 30 and input.get('task_type') == 'ASR':
        logger.info('Audio > 30s. Chunking and inferring separately.')
        audio_chunks = [
            audio_array[i:i + 30 * sampling_rate]
            for i in range(0, len(audio_array), 30 * sampling_rate)
        ]
        predictions = [
            do_sample_inference(self, chunk, sampling_rate, instruction)
            for chunk in tqdm(audio_chunks)
        ]
        return ' '.join(predictions)

    elif audio_duration > 30:
        logger.info('Audio > 30s. Taking first 30 seconds.')
        audio_array = audio_array[:30 * sampling_rate]

    elif audio_duration < 1:
        logger.info('Audio < 1s. Padding to 1 second.')
        audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

    return do_sample_inference(self, audio_array, sampling_rate, instruction)
