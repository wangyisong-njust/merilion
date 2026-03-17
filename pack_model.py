#!/usr/bin/env python3
"""Pack a MERaLiON-2 model directory into a single .mera checkpoint file.

Binary format (MERA v1)
-----------------------
Bytes  0-3:   magic  b"MERA"
Bytes  4-7:   version uint32 LE = 1
Bytes  8-15:  header length uint64 LE  (includes trailing ALIGN padding)
Bytes 16-N:   header JSON (UTF-8), zero-padded to ALIGN-byte boundary
Bytes  N+:    tensor data blocks, each ALIGN-padded

Header JSON keys:
    format_version    int
    model_config      dict  (config.json verbatim)
    configs           dict  fname -> JSON-decoded content or base64 string (binary)
    source_files      dict  fname -> Python source string
    storage           str   "int8" or "float16"
    tensors           dict  name -> {dtype, shape, offset, nbytes}
                             INT8 linear weights also have a "<name>_scale" entry

Storage strategy:
    text_decoder.model.*  nn.Linear weights  → INT8 per-output-channel + FP32 scale
    everything else                          → FP16

On the target device:  load .mera → reconstruct model → dequantize INT8 → FP32
  → optionally re-quantize with torchao (w8a8/int8ao/int4) → torch.compile.

Usage:
    python pack_model.py path/to/model_dir --output model.mera
    python pack_model.py path/to/model_dir --no_quant  # FP16 only, larger file
"""

import argparse
import base64
import glob
import json
import os
import struct
import sys
import time

import numpy as np
import torch

MAGIC = b"MERA"
VERSION = 1
ALIGN = 64  # bytes; all headers and tensors are padded to this boundary

# JSON config files to bundle (present in the model directory)
_CONFIG_FILES = [
    "config.json",          # already loaded separately but also stored verbatim
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "processor_config.json",
    "generation_config.json",
    "chat_template.json",
    "tokenizer.model",      # SentencePiece binary → base64-encoded
]

# Python source files to bundle (custom processor / model code)
_SOURCE_FILES = [
    "processing_meralion2.py",
]


# ── quantization helpers ────────────────────────────────────────────────────

def _quantize_int8_per_channel(tensor: torch.Tensor):
    """Symmetric per-output-channel INT8 quantization.

    Returns (int8_tensor, float32_scale).
    scale shape = (out_features,) = first dim of tensor.
    """
    w = tensor.float()
    reduce_dims = tuple(range(1, w.ndim))
    scale = w.abs().amax(dim=reduce_dims).clamp(min=1e-8) / 127.0
    scale_bc = scale.reshape(-1, *([1] * (w.ndim - 1)))
    w_int8 = (w / scale_bc).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale.float()


def _should_quantize(name: str, tensor: torch.Tensor) -> bool:
    """True for nn.Linear weight tensors in the text decoder transformer."""
    if not name.endswith(".weight"):
        return False
    if tensor.ndim < 2:
        return False
    # Only the Gemma-2 transformer blocks dominate size — quantize those.
    # Speech encoder, audio adapter, embeddings stay in FP16.
    if not name.startswith("text_decoder.model.layers."):
        return False
    return True


# ── writer helpers ──────────────────────────────────────────────────────────

def _pad(data: bytes) -> bytes:
    """Append zero bytes so that total length is a multiple of ALIGN."""
    rem = (-len(data)) % ALIGN
    return data + bytes(rem)


def _append_tensor(name: str, tensor: torch.Tensor, tensor_index: dict,
                   blobs: list, offset_ref: list) -> None:
    """Encode tensor, update index, append to blob list, advance offset."""
    data = tensor.contiguous().numpy().tobytes()
    padded = _pad(data)
    tensor_index[name] = {
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": list(tensor.shape),
        "offset": offset_ref[0],
        "nbytes": len(data),
    }
    blobs.append(padded)
    offset_ref[0] += len(padded)


# ── main packer ─────────────────────────────────────────────────────────────

def pack_model(model_dir: str, output_path: str, quantize: bool = True) -> None:
    """Load model from *model_dir* and write a packed .mera checkpoint."""
    from safetensors.torch import load_file

    model_dir = os.path.abspath(model_dir)
    print(f"Packing  {model_dir}")
    print(f"      →  {output_path}")

    # ── 1. model config ──────────────────────────────────────────────────────
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        sys.exit(f"ERROR: config.json not found in {model_dir}")
    with open(config_path) as fh:
        model_config = json.load(fh)

    # ── 2. auxiliary config files ────────────────────────────────────────────
    configs: dict = {}
    for fname in _CONFIG_FILES:
        fpath = os.path.join(model_dir, fname)
        if not os.path.exists(fpath):
            continue
        if fname.endswith(".model"):            # SentencePiece binary → base64
            with open(fpath, "rb") as fh:
                configs[fname] = {"_base64": base64.b64encode(fh.read()).decode("ascii")}
        else:
            with open(fpath) as fh:
                try:
                    configs[fname] = json.load(fh)
                except json.JSONDecodeError:
                    configs[fname] = fh.read()    # store as plain string

    # ── 3. Python source files (processor class, etc.) ───────────────────────
    source_files: dict = {}
    for fname in _SOURCE_FILES:
        fpath = os.path.join(model_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as fh:
                source_files[fname] = fh.read()

    print(f"  Bundled {len(configs)} config file(s), "
          f"{len(source_files)} source file(s)")

    # ── 4. load state dict ───────────────────────────────────────────────────
    print("Loading weights …")
    state_dict: dict = {}
    sf_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if sf_files:
        for sf in sf_files:
            state_dict.update(load_file(sf, device="cpu"))
    else:
        bin_files = sorted(glob.glob(os.path.join(model_dir, "*.bin")))
        if not bin_files:
            sys.exit(f"ERROR: no .safetensors or .bin files found in {model_dir}")
        for bf in bin_files:
            state_dict.update(torch.load(bf, map_location="cpu", weights_only=True))

    n_total = len(state_dict)
    print(f"  {n_total} tensors loaded")

    # ── 5. encode tensors ─────────────────────────────────────────────────────
    tensor_index: dict = {}
    blobs: list = []
    offset = [0]  # mutable so _append_tensor can advance it

    n_int8 = n_fp16 = 0
    for i, (name, tensor) in enumerate(state_dict.items(), 1):
        tensor = tensor.cpu()

        if quantize and _should_quantize(name, tensor):
            w_int8, scale = _quantize_int8_per_channel(tensor)
            _append_tensor(name,               w_int8,             tensor_index, blobs, offset)
            _append_tensor(name + "_scale",    scale,              tensor_index, blobs, offset)
            n_int8 += 1
        else:
            t_f16 = tensor.to(torch.float16)
            _append_tensor(name, t_f16, tensor_index, blobs, offset)
            n_fp16 += 1

        if i % 100 == 0 or i == n_total:
            print(f"  [{i:4d}/{n_total}]  {offset[0] / 1e9:.2f} GB encoded")

    print(f"  {n_int8} INT8 linear weights  +  {n_fp16} FP16 tensors")

    # ── 6. build & write header ──────────────────────────────────────────────
    header = {
        "format_version": VERSION,
        "model_config":   model_config,
        "configs":        configs,
        "source_files":   source_files,
        "storage":        "int8" if quantize else "float16",
        "tensors":        tensor_index,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_padded = _pad(header_bytes)

    print(f"Writing {output_path} …")
    with open(output_path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<I", VERSION))
        fh.write(struct.pack("<Q", len(header_padded)))
        fh.write(header_padded)
        for blob in blobs:
            fh.write(blob)

    file_size = os.path.getsize(output_path)
    print(f"\nDone.")
    print(f"  File size : {file_size / 1e9:.2f} GB")
    print(f"  Header    : {len(header_padded) / 1e6:.1f} MB")
    print(f"  Tensors   : {offset[0] / 1e9:.2f} GB")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack a MERaLiON-2 model directory into a single .mera file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model_dir",
                        help="Source model directory (must contain config.json + weights)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output .mera path (default: <model_dir>.mera)")
    parser.add_argument("--no_quant", action="store_true",
                        help="Store all weights as FP16 — larger file, no INT8 compression")
    args = parser.parse_args()

    model_dir = args.model_dir.rstrip("/\\")
    output    = args.output or (model_dir + ".mera")

    t0 = time.time()
    pack_model(model_dir, output, quantize=not args.no_quant)
    print(f"  Total     : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
