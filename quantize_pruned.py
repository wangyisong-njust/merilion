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
import argparse
import logging

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def align_mlp_dims(model, alignment=128):
    """Pad MLP intermediate_size to a multiple of `alignment`.

    vLLM's quantized kernels (Marlin, AllSpark) require weight dimensions
    divisible by 128. Pruned models may have non-aligned intermediate_size.
    Zero-padding is safe: extra gate/up dims produce zeros after activation,
    multiplied by zero columns in down_proj → zero contribution.
    """
    actual_intermediate = None
    for name, module in model.named_modules():
        if 'text_decoder' in name and 'gate_proj' in name and isinstance(module, torch.nn.Linear):
            actual_intermediate = module.out_features
            break

    if actual_intermediate is None:
        logger.info("No text_decoder MLP found, skipping alignment")
        return

    aligned = ((actual_intermediate + alignment - 1) // alignment) * alignment
    if aligned == actual_intermediate:
        logger.info(f"MLP intermediate_size {actual_intermediate} already aligned to {alignment}")
        return

    pad = aligned - actual_intermediate
    logger.info(f"Padding MLP intermediate_size: {actual_intermediate} → {aligned} (+{pad})")

    for name, module in model.named_modules():
        if 'text_decoder' not in name or '.mlp.' not in name:
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        w = module.weight.data
        if 'gate_proj' in name or 'up_proj' in name:
            module.weight = torch.nn.Parameter(
                torch.cat([w, torch.zeros(pad, w.shape[1], dtype=w.dtype, device=w.device)], dim=0))
            module.out_features = aligned
        elif 'down_proj' in name:
            module.weight = torch.nn.Parameter(
                torch.cat([w, torch.zeros(w.shape[0], pad, dtype=w.dtype, device=w.device)], dim=1))
            module.in_features = aligned

    if hasattr(model.config, 'text_config'):
        model.config.text_config.intermediate_size = aligned


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
        suffix = "FP8" if scheme == "FP8_DYNAMIC" else f"{scheme}-RTN"
        save_dir = os.path.join(os.path.dirname(model_path), f"{model_name}-{suffix}")

    logger.info(f"Model:   {model_path}")
    mode = "dynamic FP8, no calibration" if scheme == "FP8_DYNAMIC" else "RTN, no calibration"
    logger.info(f"Scheme:  {scheme} ({mode})")
    logger.info(f"Save to: {save_dir}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Use AutoModelForSpeechSeq2Seq — the model's auto_map registers this class,
    # not AutoModelForCausalLM. trust_remote_code handles relative imports correctly.
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
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

    # Fix conflicting generation_config: use_cache=False + cache_implementation=hybrid
    # causes a strict validation error in save_pretrained. Clear cache_implementation.
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.cache_implementation = None

    model.save_pretrained(save_dir, save_compressed=True)
    processor.save_pretrained(save_dir)
    logger.info(f"Saved to: {save_dir}")
    logger.info("Load in vLLM with: LLM(model=save_dir, trust_remote_code=True)")
    logger.info("(vLLM auto-detects compressed-tensors format from config.json)")
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize pruned MERaLiON-2 with llm-compressor")
    parser.add_argument("--model", required=True, help="Path to pruned/tuned checkpoint")
    parser.add_argument("--scheme", default="W8A16",
                        choices=["W8A16", "W4A16", "FP8_DYNAMIC"],
                        help="W8A16/W4A16: INT weight-only RTN; FP8_DYNAMIC: FP8 weights+activations")
    parser.add_argument("--save_dir", default=None,
                        help="Output dir (default: <model>-<scheme>-RTN or <model>-FP8)")
    args = parser.parse_args()
    quantize_pruned(args.model, args.scheme, args.save_dir)
