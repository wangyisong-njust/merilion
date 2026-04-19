"""Batch WER evaluation for MERaLiON-2 models.

Runs model.generate() on batches of audio samples simultaneously,
giving much higher GPU utilization than per-sample inference.

Supports the same quant modes as infer_gpu.py (bf16, fp16, int8, int4, mlx4).

Usage:
    python eval_wer_batch.py \\
        --model /path/to/model \\
        --dataset /path/to/IMDA_PART1_mono_en_30_ASR \\
        --quant bf16 --batch_size 8 --output wer_full_batch.json
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


SAMPLE_RATE = 16000
CHUNK_SIZE = SAMPLE_RATE * 30
SPEECH_TOKENS_PER_CHUNK = 100
MAX_CHUNKS = 8


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


# AudioBench normalization (matches AudioLLMs/AudioBench preprocess_text_asr)
def _normalize_text_audiobench(text: str) -> str:
    import jiwer
    from whisper.normalizers import EnglishTextNormalizer
    _jiwer_pipeline = jiwer.Compose([
        jiwer.RemoveMultipleSpaces(),
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemoveKaldiNonWords(),
        jiwer.RemovePunctuation(),
    ])
    _whisper_norm = EnglishTextNormalizer()
    text = text.lower()
    text = _whisper_norm(text)
    # digits to words (0-9 only; Whisper handles larger numbers)
    for digit, word in [("0","zero"),("1","one"),("2","two"),("3","three"),
                        ("4","four"),("5","five"),("6","six"),("7","seven"),
                        ("8","eight"),("9","nine")]:
        text = re.sub(r'\b' + digit + r'\b', word, text)
    # remove bracket content [] () {} <>
    text = re.sub(r'[\(\[\{\<][^\n\(\)\[\]\{\}\<\>]*[\)\]\}\>]', "", text)
    text = _jiwer_pipeline(text)
    text = re.sub(r'\b(uh|umm|um|er|ah)\b', '', text)
    return text.strip()


def _model_is_pruned(model) -> bool:
    try:
        cfg = model.text_decoder.model.config
        return (getattr(cfg, "midblock_start", -1) >= 0
                and getattr(cfg, "midblock_ratio", 1.0) < 1.0)
    except Exception:
        return False


def prepare_audio(audio_array: np.ndarray, sample_rate: int, processor):
    import librosa
    fe = processor.feature_extractor
    target_sr = fe.sampling_rate
    if sample_rate != target_sr:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate,
                                       target_sr=target_sr)
    chunks = []
    for i in range(0, len(audio_array), CHUNK_SIZE):
        chunk = audio_array[i:i + CHUNK_SIZE]
        if len(chunk) < target_sr:
            chunk = np.pad(chunk, (0, target_sr - len(chunk)), "constant")
        chunks.append(chunk)
    chunks = chunks[:MAX_CHUNKS]
    out = fe(chunks, sampling_rate=target_sr, return_attention_mask=True,
             padding="max_length", return_tensors="pt", do_normalize=True)
    return out.input_features, out.attention_mask, len(chunks) * SPEECH_TOKENS_PER_CHUNK


# ── copy of MLX4 quant from infer_gpu.py ────────────────────────────────────

class _MLX4Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, has_bias, group_size=64):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.group_size   = group_size
        self._pad   = (-in_features) % group_size
        I_pad       = in_features + self._pad
        n_groups    = I_pad // group_size
        self.register_buffer("weight_q",
            torch.zeros(out_features, I_pad // 2, dtype=torch.uint8))
        self.register_buffer("scales",
            torch.zeros(out_features, n_groups, dtype=torch.float16))
        self.register_buffer("zeros",
            torch.zeros(out_features, n_groups, dtype=torch.float16))
        if has_bias:
            self.register_buffer("linear_bias",
                torch.zeros(out_features, dtype=torch.float16))
        else:
            self.linear_bias = None

    @staticmethod
    def from_linear(module, group_size=64):
        O, I  = module.weight.shape
        new   = _MLX4Linear(I, O, module.bias is not None, group_size)
        w     = module.weight.detach().float()
        I_pad = I + new._pad
        n_groups = I_pad // group_size
        if new._pad:
            w = torch.nn.functional.pad(w, (0, new._pad))
        wg    = w.view(O, n_groups, group_size)
        w_min = wg.min(dim=-1).values
        w_max = wg.max(dim=-1).values
        scale = (w_max - w_min).clamp(min=1e-8) / 15.0
        zero  = w_min
        q     = ((wg - zero.unsqueeze(-1)) / scale.unsqueeze(-1)).round().clamp(0, 15).to(torch.uint8)
        q_lo  = q[:, :, 0::2]
        q_hi  = q[:, :, 1::2]
        packed = (q_hi << 4) | q_lo
        new.weight_q.copy_(packed.view(O, I_pad // 2))
        new.scales.copy_(scale.to(torch.float16))
        new.zeros.copy_(zero.to(torch.float16))
        if module.bias is not None:
            new.linear_bias.copy_(module.bias.data.to(torch.float16))
        return new

    def _dequantize(self):
        O, _ = self.weight_q.shape
        I_pad = (self.in_features + self._pad)
        n_groups = I_pad // self.group_size
        packed = self.weight_q.view(O, n_groups, self.group_size // 2)
        q_lo = (packed & 0x0F).to(torch.float32)
        q_hi = ((packed >> 4) & 0x0F).to(torch.float32)
        q = torch.stack([q_lo, q_hi], dim=-1).view(O, n_groups, self.group_size)
        scale = self.scales.float().unsqueeze(-1)
        zero  = self.zeros.float().unsqueeze(-1)
        w = scale * q + zero
        w = w.view(O, I_pad)[:, :self.in_features]
        return w

    def forward(self, x):
        w    = self._dequantize().to(x.dtype)
        bias = self.linear_bias.to(x.dtype) if self.linear_bias is not None else None
        return torch.nn.functional.linear(x, w, bias)


def _apply_mlx4_quant(model, group_size=64):
    SKIP = {"speech_encoder", "speech_audio_adapter", "lm_head"}
    n = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(s in name for s in SKIP) or "text_decoder" not in name:
            continue
        parts  = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], _MLX4Linear.from_linear(module, group_size))
        n += 1
    print(f"  MLX4 quant: replaced {n} Linear layers (group_size={group_size})")


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(model_path, quant="bf16", flash_attn=True, device="cuda"):
    from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration
    from transformers import AutoProcessor

    print("Loading processor …")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    common = dict(use_safetensors=True)

    if quant == "mlx4":
        print("Loading FP16 → MLX int4 (group=64) …")
        t0 = time.time()
        model = MERaLiON2ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, **common)
        _apply_mlx4_quant(model, group_size=64)
        model = model.to(device)

    elif quant in ("int8", "int4"):
        import bitsandbytes as bnb
        from torch import nn as _nn
        print(f"Loading FP16 → BnB {quant.upper()} …")
        t0 = time.time()
        model = MERaLiON2ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, **common)
        BNB_SKIP = ["speech_encoder", "speech_audio_adapter", "lm_head"]
        for mod_name, module in list(model.named_modules()):
            if not isinstance(module, _nn.Linear):
                continue
            if any(s in mod_name for s in BNB_SKIP):
                continue
            parts  = mod_name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            w = module.weight.data.cpu()
            has_bias = module.bias is not None
            if quant == "int8":
                new_layer = bnb.nn.Linear8bitLt(
                    module.in_features, module.out_features,
                    bias=has_bias, has_fp16_weights=False, threshold=6.0)
                new_layer.weight = bnb.nn.Int8Params(
                    w, requires_grad=False, has_fp16_weights=False)
            else:
                new_layer = bnb.nn.Linear4bit(
                    module.in_features, module.out_features,
                    bias=has_bias, quant_type="nf4",
                    compute_dtype=torch.bfloat16)
                new_layer.weight = bnb.nn.Params4bit(
                    w, requires_grad=False, quant_type="nf4")
            if has_bias:
                new_layer.bias = _nn.Parameter(module.bias.data)
            setattr(parent, parts[-1], new_layer)
        model = model.to(device)

    else:
        dtype = torch.bfloat16 if quant == "bf16" else torch.float16
        attn  = "flash_attention_2" if flash_attn else "sdpa"
        print(f"Loading {quant.upper()} (attn={attn}) …")
        t0 = time.time()
        try:
            model = MERaLiON2ForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=dtype, attn_implementation=attn, **common)
        except Exception as e:
            if flash_attn and "flash" in str(e).lower():
                print(f"  FlashAttn2 unavailable ({e}), falling back to sdpa …")
                model = MERaLiON2ForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=dtype, attn_implementation="sdpa", **common)
            else:
                raise
        model = model.to(device)

    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model, processor


# ── batch inference ───────────────────────────────────────────────────────────

def infer_batch(model, processor, audio_list, sr_list, instruction,
                max_new_tokens=128, device="cuda"):
    """Run model.generate() on a batch of audio samples.

    All samples in the batch must have the same n_speech (i.e. same number of
    30-second audio chunks).  For IMDA_PART1 30s clips this is always 100.
    Returns list of (text, n_tokens, total_batch_s).
    """
    features_list, attn_list, n_speech_list = [], [], []
    for audio, sr in zip(audio_list, sr_list):
        feats, attn, n_sp = prepare_audio(audio, sr, processor)
        features_list.append(feats)
        attn_list.append(attn)
        n_speech_list.append(n_sp)

    if len(set(n_speech_list)) > 1:
        raise ValueError(f"Batch contains mixed n_speech lengths: {set(n_speech_list)}")

    B = len(audio_list)
    n_speech = n_speech_list[0]
    tokenizer = processor.tokenizer
    speech_token_id = model.config.speech_token_index

    conversation = [{"role": "user",
                     "content": (f"Instruction: {instruction} \n"
                                 "Follow the text instruction based on the "
                                 "following audio: <SpeechHere>")}]
    prompt  = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True)
    raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
    pos     = raw_ids.index(speech_token_id)
    ids     = raw_ids[:pos] + [speech_token_id] * n_speech + raw_ids[pos + 1:]

    input_ids              = torch.tensor([ids] * B, dtype=torch.long)
    attention_mask         = torch.ones_like(input_ids)
    input_features         = torch.cat(features_list, dim=0)   # (B, 128, 3000)
    feature_attention_mask = torch.cat(attn_list, dim=0)       # (B, 3000)

    try:
        _dev = next(p.device for p in model.parameters() if p.device.type != "cpu")
    except StopIteration:
        _dev = torch.device(device)
    try:
        _dtype = next(p.dtype for p in model.parameters()
                      if p.dtype in (torch.float16, torch.bfloat16))
    except StopIteration:
        _dtype = torch.bfloat16

    input_ids              = input_ids.to(_dev)
    attention_mask         = attention_mask.to(_dev)
    input_features         = input_features.to(_dev).to(_dtype)
    feature_attention_mask = feature_attention_mask.to(_dev)

    _gen_cfg = getattr(model, "generation_config", None)
    if _gen_cfg is not None:
        _gen_cfg.cache_implementation = None

    seq_len = input_ids.shape[1]
    eos_ids = {tokenizer.eos_token_id,
               tokenizer.convert_tokens_to_ids("<end_of_turn>")}

    if _model_is_pruned(model):
        from transformers import DynamicCache
        past_kv = DynamicCache()
    else:
        from transformers.cache_utils import HybridCache
        past_kv = HybridCache(
            model.text_decoder.model.config,
            max_batch_size=B,
            max_cache_len=seq_len + max_new_tokens,
            dtype=_dtype,
            device=_dev,
        )

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            past_key_values=past_kv,
            eos_token_id=list(eos_ids),
        )
    torch.cuda.synchronize()
    total_s = time.time() - t0

    results = []
    for i in range(B):
        generated = output_ids[i][seq_len:]
        n_tokens  = max(len(generated), 1)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        text = text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()
        results.append((text, n_tokens, total_s))
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch WER evaluation for MERaLiON-2")
    parser.add_argument("--model",      required=True)
    parser.add_argument("--dataset",    required=True)
    parser.add_argument("--quant",      default="bf16",
                        choices=["bf16", "fp16", "int8", "int4", "mlx4"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=999999,
                        help="Max samples to evaluate (default: full dataset)")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--output",     default="wer_batch.json")
    parser.add_argument("--no_flash_attn", action="store_true")
    parser.add_argument("--audiobench", action="store_true",
                        help="Use AudioBench field names (context.array / answer) "
                             "and AudioBench WER normalization")
    parser.add_argument("--resume", default=None,
                        help="Path to existing output JSON to resume from; "
                             "already-evaluated samples are loaded, eval continues from next idx")
    args = parser.parse_args()
    args.model = os.path.abspath(args.model)

    if not torch.cuda.is_available():
        print("ERROR: no CUDA device found.")
        sys.exit(1)

    torch.cuda.reset_peak_memory_stats(args.device)
    model, processor = load_model(
        args.model, quant=args.quant,
        flash_attn=not args.no_flash_attn, device=args.device)
    gpu_mem_load_gb = torch.cuda.max_memory_allocated(args.device) / 1e9
    print(f"  GPU VRAM after load: {gpu_mem_load_gb:.2f} GB")

    from datasets import load_from_disk

    data    = load_from_disk(os.path.abspath(args.dataset))
    if args.audiobench:
        # AudioBench: use all samples, no offset
        end    = min(args.num_samples, len(data))
        subset = data.select(range(end))
    else:
        shuffled = data.shuffle(seed=42)
        start    = min(10500, len(shuffled))
        end      = len(shuffled)  # full dataset after offset
        subset   = shuffled.select(range(start, end))
    n_actual = len(subset)
    print(f"  Dataset: {n_actual} samples (batch_size={args.batch_size})")

    _AUDIOBENCH_INSTR = "Please help me transcribe the speech into text."

    def _get_audio_sr_ref(s):
        if args.audiobench:
            a   = np.asarray(s["context"]["array"], dtype=np.float32)
            sr  = s["context"]["sampling_rate"]
            ref = s["answer"]
            ins = _AUDIOBENCH_INSTR
        else:
            a   = np.asarray(s["context"]["audio"]["array"], dtype=np.float32)
            sr  = s["context"]["audio"]["sampling_rate"]
            ref = s["other_attributes"]["Transcription"]
            ins = s["instruction"]["text"] if isinstance(s["instruction"], dict) else s["instruction"]
        if a.ndim == 2:
            a = a.mean(axis=-1)
        return a, sr, ref, ins

    # Warm up with a single sample
    print("Warming up GPU …")
    s0 = subset[0]
    a0, sr0, _, inst0 = _get_audio_sr_ref(s0)
    infer_batch(model, processor, [a0], [sr0], inst0,
                max_new_tokens=args.max_new_tokens, device=args.device)
    torch.cuda.reset_peak_memory_stats(args.device)

    predictions, references, latencies, n_tokens_list = [], [], [], []
    samples_out = []
    batch_size  = args.batch_size

    # Resume: load previously evaluated samples and skip them
    resume_from = 0
    if args.resume and os.path.exists(args.resume):
        with open(args.resume) as f:
            prev = json.load(f)
        for entry in prev.get("samples", []):
            predictions.append(entry["prediction"])
            references.append(entry["reference"])
            latencies.append(entry["latency_s"])
            n_tokens_list.append(entry["n_tokens"])
            samples_out.append(entry)
        resume_from = len(samples_out)
        print(f"  Resumed {resume_from} samples from {args.resume}, continuing from idx {resume_from}")

    i = resume_from
    while i < n_actual:
        batch_end = min(i + batch_size, n_actual)
        batch     = [subset[j] for j in range(i, batch_end)]
        B         = len(batch)

        audio_list = []
        sr_list    = []
        ref_list   = []
        instr_list = []
        for s in batch:
            a, sr, ref, ins = _get_audio_sr_ref(s)
            audio_list.append(a)
            sr_list.append(sr)
            ref_list.append(ref)
            instr_list.append(ins)

        instr = instr_list[0]  # same instruction for all WER samples

        try:
            results = infer_batch(model, processor, audio_list, sr_list, instr,
                                  max_new_tokens=args.max_new_tokens, device=args.device)
            total_s = results[0][2]
            lat_per = total_s / B
            for j, (pred, n_tok, _) in enumerate(results):
                ref = ref_list[j]
                tps = n_tok / lat_per
                predictions.append(pred)
                references.append(ref)
                latencies.append(lat_per)
                n_tokens_list.append(n_tok)
                idx = i + j
                print(f"  [{idx+1:4d}/{n_actual}] {lat_per:5.2f}s  {tps:6.1f} tok/s | {pred[:60]}")
                samples_out.append({"idx": idx, "reference": ref, "prediction": pred,
                                    "latency_s": lat_per, "n_tokens": n_tok, "decode_tps": tps})
        except ValueError as e:
            # Mixed n_speech — fall back to one by one
            print(f"  [warn] {e} — falling back to per-sample for this batch")
            for j in range(B):
                res = infer_batch(model, processor, [audio_list[j]], [sr_list[j]], instr_list[j],
                                  max_new_tokens=args.max_new_tokens, device=args.device)
                pred, n_tok, total_s = res[0]
                tps = n_tok / total_s
                ref = ref_list[j]
                predictions.append(pred)
                references.append(ref)
                latencies.append(total_s)
                n_tokens_list.append(n_tok)
                idx = i + j
                print(f"  [{idx+1:4d}/{n_actual}] {total_s:5.2f}s  {tps:6.1f} tok/s | {pred[:60]}")
                samples_out.append({"idx": idx, "reference": ref, "prediction": pred,
                                    "latency_s": total_s, "n_tokens": n_tok, "decode_tps": tps})
        i = batch_end

    _norm_fn = _normalize_text_audiobench if args.audiobench else _normalize_text
    norm_preds = [_norm_fn(p) or "empty" for p in predictions]
    norm_refs  = [_norm_fn(r) or "empty" for r in references]
    if args.audiobench:
        from jiwer import compute_measures
        incorrect, total = 0, 0
        for p, r in zip(norm_preds, norm_refs):
            m = compute_measures(r, p)
            incorrect += m["substitutions"] + m["deletions"] + m["insertions"]
            total     += m["substitutions"] + m["deletions"] + m["hits"]
        wer = incorrect / total if total > 0 else 0.0
    else:
        import evaluate as _ev
        wer = _ev.load("wer").compute(predictions=norm_preds, references=norm_refs)
    avg_lat     = float(np.mean(latencies))
    avg_tps     = float(np.mean([n / l for n, l in zip(n_tokens_list, latencies) if l > 0]))
    gpu_peak_gb = torch.cuda.max_memory_allocated(args.device) / 1e9

    print(f"\n{'='*60}")
    print(f"WER:           {wer:.4f}  ({wer*100:.2f}%)  [normalized]")
    print(f"Avg latency:   {avg_lat:.2f} s/sample")
    print(f"Avg decode:    {avg_tps:.2f} tok/s")
    print(f"GPU VRAM peak: {gpu_peak_gb:.2f} GB")
    print(f"quant:         {args.quant}")
    print(f"batch_size:    {batch_size}")
    print(f"{'='*60}")

    result = {
        "model":          args.model,
        "quant_method":   args.quant,
        "device":         args.device,
        "batch_size":     batch_size,
        "num_samples":    n_actual,
        "wer":            wer,
        "avg_latency_s":  avg_lat,
        "avg_decode_tps": avg_tps,
        "gpu_mem_load_gb": gpu_mem_load_gb,
        "gpu_mem_peak_gb": gpu_peak_gb,
        "latencies":      latencies,
        "samples":        samples_out,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
