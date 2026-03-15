"""Quantize a pruned MERaLiON-2 checkpoint to INT4 using AutoAWQ.

AutoAWQ uses activation-aware calibration (64-128 samples) to compute per-channel
scales, producing significantly better INT4 quality than RTN. The quantized model
is saved in AWQ format and can be loaded directly by vLLM.

Calibration uses text-only samples (no audio) since AWQ calibration forward passes
cannot inject audio embeddings. The text decoder is calibrated on ASR transcription
text in the model's chat template format; speech encoder remains in FP16.

Usage:
    python quantize_pruned_awq.py \
        --model meralion_checkpoints/MERaLiON-2-3B-v3-td50-mid3-22 \
        --dataset /path/to/IMDA_PART1_mono_en_30_ASR

Then benchmark:
    python vllm_benchmark_pruned.py --pruned <save_dir> ...
    python vllm_eval_wer.py --model <save_dir> ...
"""
import json
import os
import shutil
import sys
import argparse
import logging

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForCausalLM

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def align_mlp_dims(model, alignment=128):
    """Pad MLP intermediate_size to a multiple of `alignment`.

    AWQ GEMM kernel requires weight dimensions divisible by the group size (128).
    Zero-padding is safe: extra gate/up dims produce zeros after activation.
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


def get_calib_texts(dataset_path, num_calib=64, seed=42):
    """Extract text-only calibration samples from the ASR dataset.

    Uses transcription text formatted in the model's chat template.
    Audio is omitted — the text decoder is calibrated on output-side token sequences.
    """
    from datasets import load_from_disk
    dataset = load_from_disk(dataset_path)
    dataset = dataset.shuffle(seed=seed).select(range(min(num_calib * 2, len(dataset))))

    texts = []
    for sample in dataset:
        instruction = sample.get("instruction", {})
        if isinstance(instruction, dict):
            instruction = instruction.get("text", "")
        transcription = (
            sample.get("other_attributes", {}).get("Transcription", "")
            or sample.get("answer", {}).get("text", "")
            if isinstance(sample.get("answer"), dict)
            else sample.get("answer", "")
        )
        if not transcription:
            continue
        text = (
            "<start_of_turn>user\n"
            f"Instruction: {instruction}\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
            f"{transcription}<end_of_turn>"
        )
        texts.append(text)
        if len(texts) >= num_calib:
            break

    logger.info(f"Built {len(texts)} calibration texts")
    return texts


def quantize_awq(model_path: str, dataset_path: str, save_dir: str = None,
                  num_calib: int = 64, q_group_size: int = 128):
    """Quantize a pruned MERaLiON-2 checkpoint to W4A16 with AutoAWQ.

    Strategy for custom model loading:
      AutoAWQForCausalLM.from_pretrained internally calls AutoModelForCausalLM, but
      our model's auto_map only registers AutoModelForSpeechSeq2Seq. We work around
      this by pre-loading the model with AutoModelForSpeechSeq2Seq and registering
      its class with AutoModelForCausalLM so AutoAWQ finds it on the second load.
    """
    from awq import AutoAWQForCausalLM

    if save_dir is None:
        model_name = os.path.basename(model_path.rstrip("/"))
        save_dir = os.path.join(os.path.dirname(model_path), f"{model_name}-W4A16-AWQ")

    model_path = os.path.abspath(model_path)
    if dataset_path:
        dataset_path = os.path.abspath(dataset_path)

    logger.info(f"Model:   {model_path}")
    logger.info(f"Scheme:  W4A16 (AWQ, {num_calib} calibration samples, group={q_group_size})")
    logger.info(f"Save to: {save_dir}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    # --- Register meralion2 with AutoAWQ ---
    # AutoAWQ checks config.model_type against AWQ_CAUSAL_LM_MODEL_MAP *before*
    # loading the model. We register a custom subclass of the Gemma2 AWQ class
    # that routes get_model_layers() to text_decoder.model.layers (where our
    # Gemma2 decoder layers live). The Gemma2 AWQ class's get_layers_for_scaling
    # and get_act_order_weights work unchanged since the layer structure is identical.
    #
    # We also register with AutoModelForCausalLM so the AWQ from_pretrained call
    # (which uses AutoModelForCausalLM internally) can load our custom model class.
    logger.info("Pre-loading model to register with AutoModelForCausalLM...")
    reference_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )
    model_class = type(reference_model)
    config_class = type(reference_model.config)
    del reference_model

    AutoModelForCausalLM.register(config_class, model_class)

    from awq.models.auto import AWQ_CAUSAL_LM_MODEL_MAP
    from awq.models.gemma2 import Gemma2AWQForCausalLM
    import awq.models.base as _awq_base

    class MERaLiONAWQForCausalLM(Gemma2AWQForCausalLM):
        """AWQ wrapper for MERaLiON-2: text decoder lives at model.text_decoder.model.layers."""

        @staticmethod
        def get_model_layers(model):
            return model.text_decoder.model.layers

        @staticmethod
        def move_embed(model, device):
            model.text_decoder.model.embed_tokens = (
                model.text_decoder.model.embed_tokens.to(device)
            )

    AWQ_CAUSAL_LM_MODEL_MAP["meralion2"] = MERaLiONAWQForCausalLM
    # base.py also looks up model_type in TRANSFORMERS_AUTO_MAPPING_DICT to select
    # the HF auto class for loading. Point meralion2 to AutoModelForCausalLM, which
    # we've already registered with our model class above.
    if hasattr(_awq_base, "TRANSFORMERS_AUTO_MAPPING_DICT"):
        _awq_base.TRANSFORMERS_AUTO_MAPPING_DICT["meralion2"] = "AutoModelForCausalLM"

    logger.info("Loading with AutoAWQForCausalLM...")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Align MLP dims to 128 for AWQ GEMM kernel compatibility
    align_mlp_dims(model.model, alignment=128)

    # AWQ internals access self.model.model.* (rotary_emb, embed_tokens, layers…).
    # MERaLiON-2 has no .model attribute; the Gemma2Model lives at text_decoder.model.
    # Add .model as a plain alias in __dict__ (bypasses nn.Module.__setattr__ so
    # Gemma2Model is not double-registered as a submodule or iterated twice).
    model.model.__dict__["model"] = model.model.text_decoder.model

    # AWQ calibration calls model(**inp) where inp tensors may be on CPU after
    # AWQ moves embed_tokens to GPU. Patch Gemma2Model.forward on this instance
    # to auto-move all input tensors to the embed_tokens device.
    _td_model = model.model.text_decoder.model
    _orig_td_fwd = _td_model.forward

    def _auto_device_forward(input_ids=None, **kwargs):
        dev = _td_model.embed_tokens.weight.device
        if input_ids is not None:
            input_ids = input_ids.to(dev)
        kwargs = {k: v.to(dev) if isinstance(v, torch.Tensor) else v
                  for k, v in kwargs.items()}
        return _orig_td_fwd(input_ids=input_ids, **kwargs)

    _td_model.forward = _auto_device_forward

    # Build calibration texts
    calib_texts = get_calib_texts(dataset_path, num_calib)

    quant_config = {
        "zero_point": True,
        "q_group_size": q_group_size,
        "w_bit": 4,
        "version": "GEMM",
    }

    logger.info("Quantizing...")
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_texts,
        max_calib_samples=num_calib,
        max_calib_seq_len=512,
    )

    # Fix conflicting generation_config before saving
    inner = getattr(model, "model", model)
    if hasattr(inner, "generation_config") and inner.generation_config is not None:
        inner.generation_config.cache_implementation = None

    model.save_quantized(save_dir)
    processor.save_pretrained(save_dir)

    # save_quantized / save_pretrained may corrupt auto_map (writes empty string)
    # and architectures.  Restore both from source config.json.
    src_cfg_path = os.path.join(model_path, "config.json")
    dst_cfg_path = os.path.join(save_dir, "config.json")
    if os.path.exists(src_cfg_path) and os.path.exists(dst_cfg_path):
        with open(src_cfg_path) as f:
            src_cfg = json.load(f)
        with open(dst_cfg_path) as f:
            dst_cfg = json.load(f)
        changed = {}
        for key in ("architectures", "auto_map"):
            if key in src_cfg:
                dst_cfg[key] = src_cfg[key]
                changed[key] = src_cfg[key]
        if changed:
            with open(dst_cfg_path, "w") as f:
                json.dump(dst_cfg, f, indent=2)
            for k, v in changed.items():
                logger.info(f"  restored {k}: {v}")

    # Copy custom .py files needed for trust_remote_code
    for fname in os.listdir(model_path):
        if fname.endswith(".py"):
            shutil.copy2(os.path.join(model_path, fname), os.path.join(save_dir, fname))
            logger.info(f"  copied {fname}")

    logger.info(f"Saved to: {save_dir}")
    logger.info("Load in vLLM: LLM(model=save_dir, trust_remote_code=True)")
    logger.info("  vLLM auto-detects AWQ from quantization_config in config.json")
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize pruned MERaLiON-2 with AutoAWQ (W4A16)")
    parser.add_argument("--model", required=True, help="Path to pruned/tuned checkpoint")
    parser.add_argument("--dataset", required=True, help="Path to IMDA_PART1_mono_en_30_ASR dataset")
    parser.add_argument("--save_dir", default=None, help="Output dir (default: <model>-W4A16-AWQ)")
    parser.add_argument("--num_calib", type=int, default=64,
                        help="Calibration samples (64-128 recommended)")
    parser.add_argument("--q_group_size", type=int, default=128,
                        help="AWQ group size (default: 128)")
    args = parser.parse_args()
    quantize_awq(args.model, args.dataset, args.save_dir, args.num_calib, args.q_group_size)
