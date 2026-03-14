"""Quantize a pruned MERaLiON-2 checkpoint to INT8 (W8A16) or INT4 (W4A16) using
llm-compressor RTN (no calibration data required).

The quantized model is saved in compressed-tensors format and can be loaded
directly by vLLM — it auto-detects the quantization from config.json.

Usage:
    # INT8 weight-only (recommended, minimal WER loss):
    python quantize_pruned.py \
        --model meralion_tune_log/MERaLiON-2-3B-v3-td50-mid6-20-tune \
        --scheme W8A16

    # INT4 weight-only (more aggressive, check WER after):
    python quantize_pruned.py \
        --model meralion_tune_log/MERaLiON-2-3B-v3-td50-mid6-20-tune \
        --scheme W4A16

Then benchmark / eval the quantized model:
    python vllm_benchmark_pruned.py --pruned <save_dir> --original ...
    python vllm_eval_wer.py --model <save_dir> ...
"""
import os
import sys
import argparse
import logging

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

script_dir = os.path.dirname(os.path.abspath(__file__))
quant_src = os.path.join(script_dir, "vllm_inference")
if quant_src not in sys.path:
    sys.path.insert(0, quant_src)

from meralion_2_quant import align_mlp_dims  # reuse existing helper

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def quantize_pruned(model_path: str, scheme: str = "W8A16", save_dir: str = None):
    """Quantize a pruned MERaLiON-2 checkpoint with RTN (no calibration).

    Uses AutoModelForCausalLM so the pruned checkpoint's own custom modeling
    code is used (not the llmcompressor version), correctly handling non-uniform
    midblock layer dimensions.
    """
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    if save_dir is None:
        model_name = os.path.basename(model_path.rstrip("/"))
        save_dir = os.path.join(os.path.dirname(model_path), f"{model_name}-{scheme}-RTN")

    logger.info(f"Model:   {model_path}")
    logger.info(f"Scheme:  {scheme} (RTN, no calibration)")
    logger.info(f"Save to: {save_dir}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Use AutoModelForCausalLM so the checkpoint's own custom code is loaded —
    # this correctly handles the non-uniform midblock dimensions.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
    )
    model.cuda()
    model.eval()

    # Pad MLP intermediate_size to multiples of 128 for quantized kernel
    # alignment (Marlin / AllSpark require weight dims divisible by 128).
    # This is a no-op when intermediate_size is already aligned (e.g. 7168).
    align_mlp_dims(model, alignment=128)

    recipe = QuantizationModifier(
        targets="Linear",
        scheme=scheme,
        ignore=[
            # Keep speech encoder in FP16 — quantizing Whisper offers little
            # speedup (runs once per prefill) and risks audio quality degradation.
            r"re:^speech_encoder\.",
            "ln_speech",
            r"re:^speech_audio_adapter\.",
            # lm_head ties weights with embed_tokens; skip to avoid issues.
            "text_decoder.lm_head",
        ],
    )

    oneshot(model=model, recipe=recipe, trust_remote_code_model=True)

    model.save_pretrained(save_dir, save_compressed=True)
    processor.save_pretrained(save_dir)
    logger.info(f"Saved to: {save_dir}")
    logger.info("Load in vLLM with: LLM(model=save_dir, trust_remote_code=True)")
    logger.info("(vLLM auto-detects compressed-tensors format from config.json)")
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize pruned MERaLiON-2 (RTN, no calibration)")
    parser.add_argument("--model", required=True, help="Path to pruned/tuned checkpoint")
    parser.add_argument("--scheme", default="W8A16", choices=["W8A16", "W4A16"],
                        help="W8A16 = INT8 weights (recommended); W4A16 = INT4 weights")
    parser.add_argument("--save_dir", default=None,
                        help="Output dir (default: <model>-<scheme>-RTN)")
    args = parser.parse_args()
    quantize_pruned(args.model, args.scheme, args.save_dir)
