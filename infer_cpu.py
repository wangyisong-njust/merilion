"""CPU inference for pruned MERaLiON-2 with torchao quantization.

Supports multiple quantization schemes:
  - W8A8  (--w8a8):   INT8 weights + INT8 dynamic activations → real INT8 GEMM
                       via oneDNN.  Fastest on CPUs with VNNI/AMX.
  - W8A16 (--int8ao):  INT8 weight-only, FP32 activations (torchao, compile-ok)
  - W4A16 (--int4):    INT4 weight-only, FP32 activations (torchao, experimental)
  - W8A16 (default):   INT8 dynamic quantization (legacy torch.quantization)

Also supports loading from a packed .mera checkpoint (produced by pack_model.py).
Pass a .mera file path to --model and quantization/compile flags apply as usual.

torchao stores weights as INT4/INT8 and dequantizes on-the-fly during each
forward pass using optimized SIMD kernels (AVX-VNNI on x86, NEON on ARM).
Combined with torch.compile, this achieves real quantized GEMM speedup.

The pruned model's non-uniform layer dimensions are fully supported since
torchao quantizes each nn.Linear independently.

Start point: the merged pruned BF16 model from merge_lora.py.
The AWQ model (GPU-only kernels) cannot be used for CPU inference.

Install:  pip install torchao

Usage:
    # Single audio file:
    python infer_cpu.py \
        --model meralion_tune_log/MERaLiON-2-3B-v3-td50-mid3-22-tune \
        --audio sample.wav

    # From packed .mera checkpoint (no HuggingFace dir needed):
    python infer_cpu.py --model model.mera --w8a8 --audio sample.wav

    # WER + latency benchmark on dataset:
    python infer_cpu.py \
        --model meralion_tune_log/MERaLiON-2-3B-v3-td50-mid3-22-tune \
        --dataset /path/to/IMDA_PART1_mono_en_30_ASR \
        --num_samples 50 --output cpu_results.json

    # FP32 baseline (no quantization):
    python infer_cpu.py --model ... --dataset ... --no_quant
"""
import argparse
import json
import os
import re
import sys
import time

# Hide all CUDA devices before importing torch so that torchao INT4 kernels
# always dispatch to the CPU backend (old torchao versions create CUDA tensors
# internally even when the model is on CPU, if a GPU is visible).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

def _normalize_text(text: str) -> str:
    """Lowercase + strip punctuation for fair WER comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def _model_is_pruned(model) -> bool:
    """Return True if the model has non-uniform KV heads (was pruned via midblock).

    Gemma2Model.forward() auto-selects the cache when past_key_values is None:
      - pruned  (midblock_start>=0, midblock_ratio<1) → DynamicCache
      - original                                      → HybridCache
    We must NOT override this for the original model; forcing DynamicCache on it
    produces degenerate <Speaker> outputs because HybridCache-specific attention
    masking in prepare_inputs_for_generation is then skipped.
    """
    try:
        cfg = model.text_decoder.model.config
        return (getattr(cfg, "midblock_start", -1) >= 0
                and getattr(cfg, "midblock_ratio", 1.0) < 1.0)
    except Exception:
        return False


SAMPLE_RATE = 16000
CHUNK_SIZE = SAMPLE_RATE * 30
SPEECH_TOKENS_PER_CHUNK = 100
MAX_CHUNKS = 8


def _apply_int8_dynamic(model):
    """INT8 dynamic quantization applied only to text decoder transformer blocks.

    Three sub-modules are kept in FP32:
      speech_encoder    — Whisper audio features; INT8 corrupts them completely,
                          causing the text decoder to hallucinate web-scraped text
      speech_audio_adapter — audio-to-text projection; same sensitivity as encoder
      lm_head           — tied to embed_tokens; DynamicQuantizedLinear breaks the
                          tie and produces degenerate logits (< < < < ...)

    Only text_decoder.model (Gemma2 transformer blocks: QKV/O projections + FFN)
    is quantised.  These dominate model size and compute, so INT8 still gives
    meaningful memory and latency reduction.
    Typical CPU speedup: 1.5–2×.  WER degradation: <0.3%.
    """
    text_decoder = getattr(model, 'text_decoder', None)
    if text_decoder is None:
        print("  WARNING: text_decoder not found — skipping INT8 (unknown model structure)")
        return

    transformer = getattr(text_decoder, 'model', None)
    if transformer is None:
        print("  WARNING: text_decoder.model not found — skipping INT8")
        return

    torch.quantization.quantize_dynamic(
        transformer, {nn.Linear}, dtype=torch.qint8, inplace=True)
    print("  (quantized: text_decoder.model | FP32: speech_encoder, audio_adapter, lm_head)")


def _apply_torchao_int8(model):
    """INT8 weight-only quantization via torchao — compatible with torch.compile.

    Unlike torch.quantization.quantize_dynamic (legacy QEngine, Dynamo-opaque),
    torchao int8_weight_only() uses AffineQuantizedTensor which Dynamo can trace.
    Typical speedup with compile: 1.3–1.8× vs FP32.
    """
    try:
        from torchao.quantization import quantize_, int8_weight_only
        quantize_(model, int8_weight_only())
        return
    except ImportError:
        pass
    raise RuntimeError(
        "torchao not found or too old for int8_weight_only. "
        "Upgrade with: pip install torchao --upgrade")


def _apply_torchao_w8a8(model):
    """W8A8 (INT8 weights + INT8 dynamic activations) via torchao.

    Both weights and activations are quantized to INT8, enabling real INT8 GEMM
    on CPU via oneDNN when combined with torch.compile.  Unlike weight-only
    schemes (which dequantize to FP32 for GEMM), this keeps the entire matmul
    in INT8 → significant speedup on CPUs with VNNI/AMX support.

    Only text_decoder.model (Gemma2 transformer blocks) is quantized — same
    scope as _apply_int8_dynamic — to avoid corrupting speech encoder features,
    the audio adapter projection, and the tied lm_head weights.
    """
    text_decoder = getattr(model, 'text_decoder', None)
    if text_decoder is None:
        print("  WARNING: text_decoder not found — skipping W8A8")
        return
    transformer = getattr(text_decoder, 'model', None)
    if transformer is None:
        print("  WARNING: text_decoder.model not found — skipping W8A8")
        return

    try:
        from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
        quantize_(transformer, Int8DynamicActivationInt8WeightConfig())
        print("  (quantized: text_decoder.model W8A8 | FP32: speech_encoder, audio_adapter, lm_head)")
        return
    except ImportError:
        pass

    try:
        from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
        quantize_(transformer, int8_dynamic_activation_int8_weight())
        print("  (quantized: text_decoder.model W8A8 | FP32: speech_encoder, audio_adapter, lm_head)")
        return
    except ImportError:
        pass

    raise RuntimeError(
        "torchao W8A8 API not found (need Int8DynamicActivationInt8WeightConfig "
        "or int8_dynamic_activation_int8_weight). "
        "Upgrade with: pip install torchao --upgrade")


def _apply_torchao_int4(model):
    """INT4 weight-only quantization via torchao (experimental for pruned models).

    WARNING: old torchao Int4WeightOnlyQuantizer replaces the weight *tensor*
    with a packed uint8 blob.  On tied-weight models (lm_head ↔ embed_tokens)
    this corrupts the LM head forward pass → WER > 100%.
    Only use if the installed torchao supports per-layer filtering.

    torchao >= 0.3: quantize_() + int4_weight_only()
    """
    try:
        from torchao.quantization import quantize_, int4_weight_only
        quantize_(model, int4_weight_only())
        return
    except ImportError:
        pass

    try:
        from torchao.quantization.quant_api import Int4WeightOnlyQuantizer
        Int4WeightOnlyQuantizer(device="cpu").quantize(model)
        return
    except ImportError:
        pass

    raise RuntimeError(
        "No compatible torchao INT4 API found. "
        "Upgrade with: pip install torchao --upgrade")


class NGramDraft:
    """N-gram draft predictor for speculative decoding (no auxiliary model needed).

    Scans the full token context (prompt + generated so far) for a matching
    n-gram prefix and returns the historically-following token as the next
    draft candidate.  Tries 4-gram (prefix = last 3 tokens) before 3-gram
    (prefix = last 2 tokens) so longer context takes priority.

    For each speculative step, up to *gamma* draft tokens are proposed by
    chaining lookups: after predicting draft[0], it appends draft[0] to the
    working context and looks up draft[1], etc.

    Works entirely on Python lists — no tensors, no learned weights.
    """

    def __init__(self, ngram_sizes: tuple = (3, 4)):
        self.ngram_sizes = sorted(ngram_sizes, reverse=True)  # largest first

    def propose(self, ctx: list, gamma: int) -> list:
        """Return up to *gamma* draft token ids by chained n-gram lookup."""
        draft: list = []
        cur = list(ctx)
        for _ in range(gamma):
            tok = self._next(cur)
            if tok is None:
                break
            draft.append(tok)
            cur.append(tok)
        return draft

    def _next(self, ctx: list):
        """Return the token that historically followed the current n-gram suffix."""
        n = len(ctx)
        for ng in self.ngram_sizes:
            plen = ng - 1           # prefix length: 3 for 4-gram, 2 for 3-gram
            if n < ng:
                continue
            prefix = ctx[-plen:]    # list — compared with list slices below
            # Search positions 0..n-ng-1 so the following token is at most ctx[-2],
            # never the current last token itself (avoids trivial self-match).
            for i in range(n - ng):
                if ctx[i: i + plen] == prefix:
                    return ctx[i + plen]
        return None


def load_model_cpu_native(model_path: str):
    """Load original (non-pruned) model for default FP32 inference.

    Uses meralion2_bl for loading (the same path that works for pruned models)
    but with no quantization.  transcribe_native() then uses the native
    processor(text=..., audios=...) interface so audio preprocessing and
    speech-token expansion are handled by the model's own processor code.
    """
    from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration
    from transformers import AutoProcessor

    print(f"Loading processor …")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model in FP32 on CPU …")
    t0 = time.time()
    model = MERaLiON2ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    )
    model = model.cpu()
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model, processor


def transcribe_native(model, processor, audio_array: np.ndarray, sample_rate: int,
                      instruction: str = "Transcribe the speech",
                      max_new_tokens: int = 128,
                      speculative: bool = False,
                      gamma: int = 5) -> str:
    """Run ASR inference using the native processor + manual prefill/decode loop.

    Uses apply_chat_template (prepends <bos>) and a pre-created HybridCache
    sized for prefill + max_new_tokens to avoid generate() overflow issues.

    When *speculative* is True, each decode step proposes up to *gamma* draft
    tokens via 4-gram/3-gram lookup over the full context (prompt + generated),
    then verifies them in a single batched forward pass.  Accepted tokens cost
    zero extra forward passes; rejected tokens fall back to the model's greedy
    prediction.  Typical speedup on repetitive ASR output: 1.3–2×.
    """
    import librosa
    from transformers.cache_utils import HybridCache

    fe = processor.feature_extractor
    target_sr = fe.sampling_rate
    if sample_rate != target_sr:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=target_sr)
    if len(audio_array) > target_sr * 30:
        audio_array = audio_array[:target_sr * 30]

    conversation = [{"role": "user",
                     "content": (f"Instruction: {instruction} \n"
                                 "Follow the text instruction based on the "
                                 "following audio: <SpeechHere>")}]
    prompt = processor.tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=prompt, audios=audio_array,
                       sampling_rate=target_sr, return_tensors="pt")
    inputs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    tokenizer      = processor.tokenizer
    input_ids      = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_features = inputs.get("input_features")
    feat_attn_mask = inputs.get("feature_attention_mask")
    seq_len        = input_ids.shape[1]

    eos_ids = {tokenizer.eos_token_id,
               tokenizer.convert_tokens_to_ids("<end_of_turn>")}

    model_dtype = next(model.parameters()).dtype
    cache = HybridCache(
        model.text_decoder.model.config,
        max_batch_size=1,
        max_cache_len=seq_len + max_new_tokens,
        dtype=model_dtype,
        device=torch.device("cpu"),
    )

    generated_ids = []
    with torch.inference_mode():
        # ── Prefill ──────────────────────────────────────────────────────────
        t0 = time.time()
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feat_attn_mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(0, seq_len),
            return_dict=True,
        )
        next_tok = int(out.logits[0, -1].argmax())
        generated_ids.append(next_tok)
        t1 = time.time()

        # ── Decode ───────────────────────────────────────────────────────────
        # all_ctx: full token sequence seen so far (prompt + generated).
        # cur_pos: next write position in the KV cache.
        all_ctx = list(input_ids[0].tolist()) + generated_ids
        cur_pos = seq_len          # next_tok will be written here
        ngram   = NGramDraft() if speculative else None

        # Speculative-decoding statistics (accumulated across all steps).
        n_fwd       = 0   # verifier forward passes
        n_spec_acc  = 0   # draft tokens accepted
        n_spec_tot  = 0   # draft tokens proposed

        while len(generated_ids) < max_new_tokens:
            if next_tok in eos_ids:
                break

            draft = ngram.propose(all_ctx, gamma) if ngram is not None else []

            if draft:
                # ── Speculative verify ────────────────────────────────────────
                K         = len(draft)
                spec_ids  = torch.tensor([[next_tok] + draft])            # (1, K+1)
                spec_attn = torch.ones(1, cur_pos + K + 1, dtype=torch.long)
                spec_cpos = torch.arange(cur_pos, cur_pos + K + 1)

                out = model(
                    input_ids=spec_ids,
                    attention_mask=spec_attn,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=spec_cpos,
                    return_dict=True,
                )
                n_fwd      += 1
                n_spec_tot += K

                # Accept greedily: logits[i] predicts the token at cur_pos+i+1.
                n_acc   = 0
                stopped = False
                for i in range(K):
                    if len(generated_ids) >= max_new_tokens:
                        stopped = True
                        break
                    pred = int(out.logits[0, i].argmax())
                    if pred == draft[i]:
                        generated_ids.append(draft[i])
                        all_ctx.append(draft[i])
                        n_acc      += 1
                        n_spec_acc += 1
                        if draft[i] in eos_ids:
                            next_tok = draft[i]
                            stopped  = True
                            break
                    else:
                        # First mismatch: use model's prediction, discard rest.
                        generated_ids.append(pred)
                        all_ctx.append(pred)
                        next_tok = pred
                        n_acc   += 1
                        stopped  = True
                        break

                if not stopped:
                    # All K drafts accepted — claim bonus token from last logit.
                    if len(generated_ids) < max_new_tokens:
                        bonus = int(out.logits[0, K].argmax())
                        generated_ids.append(bonus)
                        all_ctx.append(bonus)
                        next_tok = bonus
                        n_acc   += 1

                # Advance cache pointer by the number of tokens actually committed
                # (next_tok itself was written at cur_pos; n_acc covers the rest).
                cur_pos += n_acc

            else:
                # ── Regular greedy step ───────────────────────────────────────
                cur_attn = torch.ones(1, cur_pos + 1, dtype=torch.long)
                out = model(
                    input_ids=torch.tensor([[next_tok]]),
                    attention_mask=cur_attn,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=torch.tensor([cur_pos]),
                    return_dict=True,
                )
                next_tok = int(out.logits[0, -1].argmax())
                generated_ids.append(next_tok)
                all_ctx.append(next_tok)
                cur_pos += 1
                n_fwd   += 1

        t2 = time.time()

    prefill_s  = t1 - t0
    decode_s   = t2 - t1
    n_out      = max(len(generated_ids) - 1, 1)
    decode_tps = n_out / decode_s if decode_s > 0 else 0.0
    stats = {"n_tokens": len(generated_ids), "prefill_s": prefill_s,
             "decode_s": decode_s, "decode_tps": decode_tps}
    if speculative and n_spec_tot > 0:
        stats["spec_accept_rate"] = n_spec_acc / n_spec_tot
        stats["spec_fwd_passes"]  = n_fwd

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip(), stats


def load_model_cpu(model_path: str, int4: bool = False, int8: bool = True,
                   int8ao: bool = False, w8a8: bool = False,
                   compile: bool = True, compile_mode: str = "max-autotune"):
    """Load pruned model on CPU with optional quantization and torch.compile.

    Args:
        int8:    apply PyTorch INT8 dynamic quantization (default; NOT compile-compatible)
        int8ao:  apply torchao INT8 weight-only (compile-compatible)
        w8a8:    apply torchao W8A8 dynamic activation + weight INT8 (compile-compatible;
                 real INT8 GEMM via oneDNN — fastest on VNNI/AMX CPUs)
        int4:    apply torchao INT4 weight-only (compile-compatible; experimental)
        compile: apply torch.compile for kernel fusion (skipped automatically for int8)
    """
    from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration
    from transformers import AutoProcessor

    print(f"Loading processor …")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model in FP32 on CPU …")
    t0 = time.time()
    model = MERaLiON2ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    )
    model = model.cpu()
    model.eval()
    # cache_implementation is reset to None before every generate() call in
    # transcribe() so we never trigger "Passing both cache_implementation and
    # past_key_values".  No need to touch it here.
    print(f"  Loaded in {time.time()-t0:.1f}s")

    if int8 and not int4 and not int8ao and not w8a8:
        print("Applying INT8 dynamic quantization (torch.quantization.quantize_dynamic) …")
        t0 = time.time()
        _apply_int8_dynamic(model)
        print(f"  Done in {time.time()-t0:.1f}s")
        # torch.compile is incompatible with DynamicQuantizedLinear (legacy QEngine ops
        # are not traceable by Dynamo → garbled outputs).  INT8 BLAS gives speedup
        # directly without compile, so skip it silently.
        if compile:
            print("  (torch.compile skipped: not compatible with INT8 dynamic quant)")
        compile = False

    if int8ao and not int4 and not w8a8:
        model = model.to(torch.device("cpu"))
        print("Applying torchao INT8 weight-only quantization (compile-compatible) …")
        t0 = time.time()
        _apply_torchao_int8(model)
        print(f"  Done in {time.time()-t0:.1f}s")

    if w8a8:
        model = model.to(torch.device("cpu"))
        print("Applying torchao W8A8 quantization (INT8 weights + INT8 dynamic activations) …")
        t0 = time.time()
        _apply_torchao_w8a8(model)
        print(f"  Done in {time.time()-t0:.1f}s")

    if int4:
        # Verify all tensors are on CPU before packing
        cuda_params = [(n, p.device) for n, p in model.named_parameters() if p.device.type != "cpu"]
        cuda_bufs   = [(n, b.device) for n, b in model.named_buffers()    if b.device.type != "cpu"]
        if cuda_params or cuda_bufs:
            print(f"  WARNING: {len(cuda_params)} params and {len(cuda_bufs)} buffers still on CUDA — moving them")
            for n, _ in cuda_params + cuda_bufs:
                print(f"    {n}")
        model = model.to(torch.device("cpu"))
        print("Applying torchao INT4 weight-only quantization (experimental) …")
        t0 = time.time()
        _apply_torchao_int4(model)
        print(f"  Done in {time.time()-t0:.1f}s")

    if compile:
        print(f"Compiling with torch.compile (mode={compile_mode!r}, first inference will be slow) …")
        model = torch.compile(model, mode=compile_mode)

    return model, processor


# ── .mera packed checkpoint loader ──────────────────────────────────────────

_MERA_MAGIC   = b"MERA"
_MERA_VERSION = 1


def _read_mera_header(path: str) -> tuple:
    """Parse the binary header of a .mera file.

    Returns (header_dict, tensor_data_offset) where tensor_data_offset is the
    byte position in the file where the raw tensor data begins.
    """
    import struct
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic != _MERA_MAGIC:
            raise ValueError(f"{path}: not a valid .mera file (bad magic: {magic!r})")
        version = struct.unpack("<I", fh.read(4))[0]
        if version != _MERA_VERSION:
            raise ValueError(f"{path}: unsupported .mera version {version}")
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header_bytes = fh.read(header_len)
        tensor_data_offset = fh.tell()

    header = json.loads(header_bytes.rstrip(b"\x00"))
    return header, tensor_data_offset


def _reconstruct_processor(header: dict):
    """Reconstruct the MERaLiON2Processor from configs bundled in the header.

    Writes tokenizer files to a temp directory, loads them with AutoTokenizer,
    builds the WhisperFeatureExtractor from the bundled preprocessor config,
    then instantiates MERaLiON2Processor directly — no AutoProcessor needed.
    """
    import base64
    import importlib.util
    import tempfile
    from transformers import AutoTokenizer, WhisperFeatureExtractor

    configs      = header.get("configs", {})
    source_files = header.get("source_files", {})
    model_config = header["model_config"]

    # ── tokenizer ─────────────────────────────────────────────────────────
    tok_dir = tempfile.mkdtemp(prefix="mera_tok_")
    try:
        for fname in ("tokenizer.json", "tokenizer_config.json",
                      "special_tokens_map.json", "chat_template.json"):
            content = configs.get(fname)
            if content is None:
                continue
            fpath = os.path.join(tok_dir, fname)
            if isinstance(content, dict):
                with open(fpath, "w") as fh:
                    json.dump(content, fh)
            else:
                with open(fpath, "w") as fh:
                    fh.write(content)

        sp = configs.get("tokenizer.model")
        if isinstance(sp, dict) and "_base64" in sp:
            with open(os.path.join(tok_dir, "tokenizer.model"), "wb") as fh:
                fh.write(base64.b64decode(sp["_base64"]))

        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    finally:
        import shutil
        shutil.rmtree(tok_dir, ignore_errors=True)

    # ── feature extractor ─────────────────────────────────────────────────
    fe_cfg = configs.get("preprocessor_config.json", {})
    # Remove non-constructor keys that WhisperFeatureExtractor doesn't accept
    fe_cfg = {k: v for k, v in fe_cfg.items()
              if not k.startswith("_") and k != "processor_class"}
    feature_extractor = WhisperFeatureExtractor(**fe_cfg)

    # ── processor class ───────────────────────────────────────────────────
    proc_src = source_files.get("processing_meralion2.py", "")
    if proc_src:
        with tempfile.NamedTemporaryFile(
                suffix=".py", prefix="mera_proc_", delete=False, mode="w") as tf:
            tf.write(proc_src)
            tmp_py = tf.name
        try:
            spec = importlib.util.spec_from_file_location("_mera_proc", tmp_py)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            MERaLiON2Processor = mod.MERaLiON2Processor
        finally:
            os.unlink(tmp_py)
    else:
        # Fallback: processor source wasn't bundled, use local copy
        script_dir = os.path.dirname(os.path.abspath(__file__))
        proc_path  = os.path.join(script_dir, "meralion2_bl", "processing_meralion2.py")
        if not os.path.exists(proc_path):
            raise RuntimeError(
                "processing_meralion2.py not bundled in .mera file and not found "
                "in meralion2_bl/. Re-pack with a model dir that contains it.")
        spec = importlib.util.spec_from_file_location("_mera_proc", proc_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        MERaLiON2Processor = mod.MERaLiON2Processor

    speech_token_index = model_config.get("speech_token_index", 255999)
    return MERaLiON2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        speech_token_index=speech_token_index,
    )


def _load_tensors(header: dict, packed_path: str) -> dict:
    """Read tensor data from the .mera file, dequantize INT8 → FP32.

    Uses numpy memmap so only the requested byte ranges are faulted in —
    the full tensor section is never loaded into RAM at once.

    INT8 weight tensors are stored alongside a "<name>_scale" FP32 tensor.
    Dequantization: w_fp32 = w_int8.float() * scale (per output channel).
    All other tensors (FP16) are cast to FP32 before returning.
    """
    tensor_index = header["tensors"]

    # Byte offset where tensor data starts inside the file
    _, tensor_data_start = _read_mera_header(packed_path)

    mm = np.memmap(packed_path, dtype="uint8", mode="r", offset=tensor_data_start)

    # Collect scale tensors first so we can dequantize in one pass
    scales: dict = {}
    for name, info in tensor_index.items():
        if name.endswith("_scale") and info["dtype"] == "float32":
            raw   = mm[info["offset"]: info["offset"] + info["nbytes"]]
            scale = torch.from_numpy(
                np.frombuffer(raw, dtype=np.float32).copy()
            ).reshape(info["shape"])
            scales[name] = scale

    state_dict: dict = {}
    n = len(tensor_index)
    skipped_scale = 0
    for i, (name, info) in enumerate(tensor_index.items(), 1):
        if name.endswith("_scale"):
            skipped_scale += 1
            continue                     # already loaded above

        raw  = mm[info["offset"]: info["offset"] + info["nbytes"]]
        np_dtype = {"int8": np.int8, "float16": np.float16,
                    "float32": np.float32, "bfloat16": np.uint16}.get(info["dtype"])
        if np_dtype is None:
            raise ValueError(f"Unknown dtype {info['dtype']!r} for tensor {name!r}")

        arr    = np.frombuffer(raw, dtype=np_dtype).copy().reshape(info["shape"])
        tensor = torch.from_numpy(arr)

        if info["dtype"] == "bfloat16":
            tensor = tensor.view(torch.bfloat16)

        if info["dtype"] == "int8":
            scale_name = name + "_scale"
            if scale_name not in scales:
                raise RuntimeError(f"Missing scale tensor {scale_name!r} for INT8 weight {name!r}")
            scale    = scales[scale_name]
            scale_bc = scale.float().reshape(-1, *([1] * (tensor.ndim - 1)))
            tensor   = tensor.float() * scale_bc
        else:
            tensor = tensor.float()

        state_dict[name] = tensor

        if i % 100 == 0 or (i - skipped_scale) == (n - len(scales)):
            print(f"  [{i:4d}/{n}] tensors loaded")

    return state_dict


def load_model_packed(packed_path: str, int4: bool = False, int8: bool = True,
                      int8ao: bool = False, w8a8: bool = False,
                      compile: bool = True, compile_mode: str = "max-autotune"):
    """Load a .mera packed checkpoint and return (model, processor).

    Pipeline:
        1. Parse binary header  → model config + bundled processor files
        2. Reconstruct processor (tokenizer + feature extractor)
        3. Instantiate model architecture from config (empty weights)
        4. Read + dequantize tensors from file  → FP32 state_dict
        5. load_state_dict + tie_weights
        6. Optionally apply torchao quantization (w8a8 / int8ao / int4)
        7. Optionally torch.compile

    Quantization flags have the same semantics as load_model_cpu().
    """
    from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration
    from meralion2_bl.configuration_meralion2 import MERaLiON2Config

    packed_path = os.path.abspath(packed_path)
    print(f"Loading packed checkpoint: {packed_path}")

    # ── 1. parse header ────────────────────────────────────────────────────
    t0 = time.time()
    header, _ = _read_mera_header(packed_path)
    print(f"  Header parsed  (format v{header['format_version']}, "
          f"storage={header.get('storage','?')})  "
          f"{len(header['tensors'])} tensor entries")

    # ── 2. processor ───────────────────────────────────────────────────────
    print("Reconstructing processor …")
    processor = _reconstruct_processor(header)
    print(f"  OK  ({time.time()-t0:.1f}s)")

    # ── 3. model architecture from config ──────────────────────────────────
    print("Instantiating model from config …")
    config = MERaLiON2Config.from_dict(header["model_config"])
    model  = MERaLiON2ForConditionalGeneration(config)
    model.eval()

    # ── 4. load + dequantize tensors ───────────────────────────────────────
    print("Reading tensors …")
    t1 = time.time()
    state_dict = _load_tensors(header, packed_path)
    print(f"  {len(state_dict)} tensors dequantized  ({time.time()-t1:.1f}s)")

    # ── 5. populate model weights ──────────────────────────────────────────
    print("Loading state dict …")
    if hasattr(model, "speech_encoder"):
        model.speech_encoder.resize_to_match(state_dict, "speech_encoder")
    if hasattr(model, "text_decoder"):
        model.text_decoder.resize_to_match(state_dict, "text_decoder")

    msg = model.load_state_dict(state_dict, strict=False)
    if msg.missing_keys:
        print(f"  Missing keys  : {len(msg.missing_keys)}")
    if msg.unexpected_keys:
        print(f"  Unexpected    : {len(msg.unexpected_keys)}")

    if "text_decoder.lm_head.weight" not in state_dict:
        model.tie_weights()

    model = model.cpu().to(torch.float32)
    del state_dict
    print(f"  Loaded  ({time.time()-t0:.1f}s total)")

    # ── 6 & 7. quantization + compile  (same as load_model_cpu) ────────────
    if int8 and not int4 and not int8ao and not w8a8:
        print("Applying INT8 dynamic quantization …")
        t1 = time.time()
        _apply_int8_dynamic(model)
        print(f"  Done in {time.time()-t1:.1f}s")
        if compile:
            print("  (torch.compile skipped: not compatible with INT8 dynamic quant)")
        compile = False

    if int8ao and not int4 and not w8a8:
        print("Applying torchao INT8 weight-only quantization …")
        t1 = time.time()
        _apply_torchao_int8(model)
        print(f"  Done in {time.time()-t1:.1f}s")

    if w8a8:
        print("Applying torchao W8A8 quantization …")
        t1 = time.time()
        _apply_torchao_w8a8(model)
        print(f"  Done in {time.time()-t1:.1f}s")

    if int4:
        model = model.to(torch.device("cpu"))
        print("Applying torchao INT4 weight-only quantization …")
        t1 = time.time()
        _apply_torchao_int4(model)
        print(f"  Done in {time.time()-t1:.1f}s")

    if compile:
        print(f"Compiling with torch.compile (mode={compile_mode!r}, first inference will be slow) …")
        model = torch.compile(model, mode=compile_mode)

    return model, processor


def prepare_audio(audio_array: np.ndarray, sample_rate: int, processor):
    """Resample, chunk, extract mel features. Returns (input_features, mask, n_speech_tokens)."""
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
            chunk = np.pad(chunk, (0, target_sr - len(chunk)), 'constant')
        chunks.append(chunk)
    chunks = chunks[:MAX_CHUNKS]

    out = fe(chunks, sampling_rate=target_sr, return_attention_mask=True,
             padding="max_length", return_tensors="pt", do_normalize=True)
    return out.input_features, out.attention_mask, len(chunks) * SPEECH_TOKENS_PER_CHUNK


def transcribe(model, processor, audio_array: np.ndarray, sample_rate: int,
               instruction: str = "Transcribe the speech",
               max_new_tokens: int = 128,
               speculative: bool = False,
               gamma: int = 5) -> str:
    """Run ASR inference for a single audio sample.

    When *speculative* is True, passes *gamma* as prompt_lookup_num_tokens to
    model.generate(), enabling HuggingFace's built-in prompt-lookup speculative
    decoding (n-gram search over the full input context).
    """
    input_features, feature_attention_mask, n_speech = prepare_audio(
        audio_array, sample_rate, processor)

    tokenizer = processor.tokenizer
    speech_token_id = model.config.speech_token_index

    conversation = [{"role": "user",
                     "content": (f"Instruction: {instruction} \n"
                                 "Follow the text instruction based on the "
                                 "following audio: <SpeechHere>")}]
    prompt = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True)
    raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        pos = raw_ids.index(speech_token_id)
    except ValueError:
        raise RuntimeError(
            f"speech_token_id={speech_token_id} not in tokenized prompt. "
            "Verify processor matches model.")

    # Expand the single <SpeechHere> placeholder to n_speech copies
    input_ids = torch.tensor(
        [raw_ids[:pos] + [speech_token_id] * n_speech + raw_ids[pos + 1:]],
        dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    # Always clear cache_implementation to prevent the "Passing both
    # cache_implementation and past_key_values" error in generate().
    _gen_cfg = getattr(model, "generation_config", None)
    if _gen_cfg is not None:
        _gen_cfg.cache_implementation = None

    # Create the right cache type and size before calling generate():
    #   Original model  → HybridCache sized for prefill + generation
    #   Pruned model    → DynamicCache (handles non-uniform KV heads)
    #
    # WHY pre-create instead of letting Gemma2Model.forward() create it:
    #   forward() creates HybridCache with max_cache_len=seq_len (prefill only).
    #   The first generated token overflows that cache → garbage output.
    model_dtype = next(model.parameters()).dtype
    max_cache = input_ids.shape[1] + max_new_tokens
    if _model_is_pruned(model):
        from transformers import DynamicCache
        past_kv = DynamicCache()
    else:
        from transformers.cache_utils import HybridCache
        past_kv = HybridCache(
            model.text_decoder.model.config,
            max_batch_size=1,
            max_cache_len=max_cache,
            dtype=model_dtype,
            device=torch.device("cpu"),
        )

    t0 = time.time()
    with torch.inference_mode():
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            past_key_values=past_kv,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<end_of_turn>"),
            ],
        )
        if speculative:
            gen_kwargs["prompt_lookup_num_tokens"] = gamma
        output_ids = model.generate(**gen_kwargs)
    total_s = time.time() - t0

    generated  = output_ids[0][input_ids.shape[1]:]
    n_tokens   = max(len(generated), 1)
    # generate() fuses prefill+decode; tps is slightly conservative (lower bound)
    decode_tps = n_tokens / total_s if total_s > 0 else 0.0
    stats = {"n_tokens": n_tokens, "decode_tps": decode_tps}

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip(), stats


def main():
    parser = argparse.ArgumentParser(
        description="CPU ASR inference — pruned MERaLiON-2 with torchao quantization")
    parser.add_argument("--model", required=True,
                        help="Model directory OR packed .mera checkpoint file")
    parser.add_argument("--audio", default=None,
                        help="Single audio file (.wav/.flac/.mp3)")
    parser.add_argument("--instruction", default="Transcribe the speech")
    parser.add_argument("--dataset", default=None,
                        help="IMDA_PART1_mono_en_30_ASR dataset path")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--no_quant", action="store_true",
                        help="FP32 baseline, no quantization")
    parser.add_argument("--int4", action="store_true",
                        help="Use torchao INT4 (experimental; may break tied-weight models)")
    parser.add_argument("--int8ao", action="store_true",
                        help="Use torchao INT8 weight-only (compile-compatible, vs dynamic INT8)")
    parser.add_argument("--w8a8", action="store_true",
                        help="Use torchao W8A8 (INT8 weights + INT8 dynamic activations). "
                             "Real INT8 GEMM via oneDNN — fastest on VNNI/AMX CPUs.")
    parser.add_argument("--no_compile", action="store_true",
                        help="Skip torch.compile (faster startup, slower inference)")
    parser.add_argument("--compile_mode", default="max-autotune",
                        choices=["max-autotune", "reduce-overhead", "default"],
                        help="torch.compile mode (default: max-autotune). "
                             "reduce-overhead: faster startup, less tuning. "
                             "default: minimal compilation, lowest RAM.")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Use model's native code (trust_remote_code=True). "
                             "For the original un-pruned model only. "
                             "Bypasses meralion2_bl and all cache workarounds.")
    parser.add_argument("--speculative", action="store_true",
                        help="Enable n-gram speculative decoding (3-gram + 4-gram). "
                             "Proposes draft tokens from the generation context with "
                             "zero extra model overhead; typical speedup 1.3–2×.")
    parser.add_argument("--gamma", type=int, default=5,
                        help="Max draft tokens per speculative step (default: 5). "
                             "Higher values increase potential speedup but also the "
                             "cost of a rejected step.")
    parser.add_argument("--output", default="cpu_results.json")
    parser.add_argument("--audio_dir", default=None,
                        help="Save each sample's audio as WAV here (for HTML demo page)")
    parser.add_argument("--save_samples", action="store_true",
                        help="Include per-sample predictions + references in JSON output")
    args = parser.parse_args()
    args.model = os.path.abspath(args.model)

    use_int8ao = not args.no_quant and args.int8ao
    use_w8a8   = not args.no_quant and args.w8a8
    use_int4   = not args.no_quant and args.int4
    use_int8   = not args.no_quant and not args.int4 and not args.int8ao and not args.w8a8

    # Measure RSS before loading
    try:
        import psutil, os as _os
        _proc = psutil.Process(_os.getpid())
        _ram_before_mb = _proc.memory_info().rss / 1e6
    except ImportError:
        _proc = None
        _ram_before_mb = 0.0

    if args.model.endswith(".mera"):
        model, processor = load_model_packed(
            args.model,
            int8=use_int8,
            int8ao=use_int8ao,
            w8a8=use_w8a8,
            int4=use_int4,
            compile=not args.no_compile,
            compile_mode=args.compile_mode,
        )
    else:
        model, processor = load_model_cpu(
            args.model,
            int8=use_int8,
            int8ao=use_int8ao,
            w8a8=use_w8a8,
            int4=use_int4,
            compile=not args.no_compile,
            compile_mode=args.compile_mode,
        )

    if _proc is not None:
        ram_after_mb = _proc.memory_info().rss / 1e6
        print(f"  RAM after load+quant: {ram_after_mb:.0f} MB  (delta: {ram_after_mb - _ram_before_mb:+.0f} MB)")
    else:
        ram_after_mb = 0.0

    _infer = transcribe_native if args.trust_remote_code else transcribe

    def _run_infer(model, processor, audio, sr, instruction, max_new_tokens):
        return _infer(model, processor, audio, sr,
                      instruction=instruction,
                      max_new_tokens=max_new_tokens,
                      speculative=args.speculative,
                      gamma=args.gamma)

    if args.speculative:
        print(f"Speculative decoding: enabled  (gamma={args.gamma}, "
              f"ngram={'3+4-gram' if args.trust_remote_code else 'prompt-lookup'})")

    # ── single audio file ──────────────────────────────────────────────────
    if args.audio:
        import soundfile as sf
        audio, sr = sf.read(args.audio)
        if audio.ndim == 2:
            audio = audio.mean(axis=-1)
        audio = audio.astype(np.float32)
        t0 = time.time()
        text, stats = _run_infer(model, processor, audio, sr,
                                 args.instruction, args.max_new_tokens)
        accept_str = (f"  accept={stats['spec_accept_rate']:.0%}"
                      if "spec_accept_rate" in stats else "")
        print(f"\nTranscription ({time.time()-t0:.2f}s, "
              f"{stats['decode_tps']:.1f} tok/s{accept_str}):\n  {text}")
        return

    # ── dataset benchmark + WER ────────────────────────────────────────────
    if args.dataset:
        from datasets import load_from_disk
        import evaluate

        data = load_from_disk(os.path.abspath(args.dataset))
        subset = data.shuffle(seed=42).select(
            range(10500, 10500 + args.num_samples))

        predictions, references, latencies = [], [], []
        samples_out = []
        if args.audio_dir:
            os.makedirs(args.audio_dir, exist_ok=True)
        for i in range(args.num_samples):
            sample = subset[i]
            audio = np.asarray(sample["context"]["audio"]["array"],
                               dtype=np.float32)
            sr    = sample["context"]["audio"]["sampling_rate"]
            if audio.ndim == 2:
                audio = audio.mean(axis=-1)
            instr = (sample["instruction"]["text"]
                     if isinstance(sample["instruction"], dict)
                     else sample["instruction"])
            ref = sample["other_attributes"]["Transcription"]

            audio_file = None
            if args.audio_dir:
                import soundfile as _sf
                audio_file = os.path.join(args.audio_dir, f"sample_{i:03d}.wav")
                if not os.path.exists(audio_file):
                    _sf.write(audio_file, audio, sr)

            t0 = time.time()
            pred, stats = _run_infer(model, processor, audio, sr, instr,
                                     args.max_new_tokens)
            elapsed = time.time() - t0
            predictions.append(pred)
            references.append(ref)
            latencies.append(elapsed)
            tps_str = f"{stats['decode_tps']:.1f} tok/s"
            acc_str = (f"  acc={stats['spec_accept_rate']:.0%}"
                       if "spec_accept_rate" in stats else "")
            print(f"  [{i+1:3d}/{args.num_samples}] {elapsed:5.1f}s  "
                  f"{tps_str:>12}{acc_str} | {pred[:60]}")
            entry = {"idx": i, "reference": ref, "prediction": pred,
                     "latency_s": elapsed, **stats}
            if audio_file:
                entry["audio_file"] = audio_file
            samples_out.append(entry)

        wer_metric = evaluate.load("wer")
        norm_preds = [_normalize_text(p) for p in predictions]
        norm_refs  = [_normalize_text(r) for r in references]
        wer         = wer_metric.compute(predictions=norm_preds, references=norm_refs)
        avg_lat     = float(np.mean(latencies))
        avg_tps     = float(np.mean([s["decode_tps"] for s in
                                     [e for e in samples_out if "decode_tps" in e]]))
        suffix = "_native" if args.trust_remote_code else ""
        spec_sfx = f"_spec{args.gamma}" if args.speculative else ""
        quant_method = ("int4"   + suffix + spec_sfx if use_int4
                        else "w8a8"  + suffix + spec_sfx if use_w8a8
                        else "int8ao" + suffix + spec_sfx if use_int8ao
                        else "int8"  + suffix + spec_sfx if use_int8
                        else "fp32"  + suffix + spec_sfx)

        avg_accept = float(np.mean([s["spec_accept_rate"] for s in samples_out
                                    if "spec_accept_rate" in s])) if args.speculative else None

        print(f"\n{'='*60}")
        print(f"WER:           {wer:.4f}  ({wer*100:.2f}%)  [normalized]")
        print(f"Avg latency:   {avg_lat:.2f} s/sample")
        print(f"Avg decode:    {avg_tps:.2f} tok/s")
        if avg_accept is not None:
            print(f"Spec accept:   {avg_accept:.1%}  (gamma={args.gamma})")
        print(f"quant:         {quant_method}")
        print(f"compiled:      {not args.no_compile}  (mode={args.compile_mode})")
        print(f"{'='*60}")
        result = {
            "model": args.model,
            "quant_method": quant_method,
            "compiled": not args.no_compile,
            "compile_mode": args.compile_mode if not args.no_compile else None,
            "speculative": args.speculative,
            "gamma": args.gamma if args.speculative else None,
            "num_samples": args.num_samples,
            "wer": wer,
            "avg_latency_s": avg_lat,
            "avg_decode_tps": avg_tps,
            "avg_spec_accept_rate": avg_accept,
            "ram_mb": ram_after_mb,
            "latencies": latencies,
        }
        if args.save_samples:
            result["samples"] = samples_out
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.output}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
