"""Extract the text_decoder of a MERaLiON-2-3B (W4A16 or bf16) checkpoint
into a standalone Gemma2 directory that vLLM can load natively (no plugin).

Use case: smoke-test vLLM's Marlin kernel on the actual W4A16 weights
without needing vllm_plugin_meralion2.

Output layout:
    <out_dir>/
      config.json                 # Gemma2Config (+ quantization_config copied)
      tokenizer.* / generation_config.json
      model-*.safetensors         # text_decoder weights, prefix stripped

Usage:
  python extract_gemma2_from_meralion.py \\
      --src   quant_checkpoints/MERaLiON-2-3B-W4A16-RTN \\
      --out   quant_checkpoints/Gemma2-2B-W4A16-RTN
"""
import argparse
import glob
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="MERaLiON-2-3B (or W4A16) dir")
    ap.add_argument("--out", required=True, help="Output Gemma2 dir")
    ap.add_argument("--prefix", default="text_decoder.",
                    help="Prefix to strip from each weight key")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- 1) Build Gemma2 config from MERaLiON's `text_config` ---
    src_cfg_path = os.path.join(args.src, "config.json")
    with open(src_cfg_path) as f:
        src_cfg = json.load(f)

    text_cfg = src_cfg.get("text_config")
    if text_cfg is None:
        raise SystemExit("config.json has no `text_config` field — is this MERaLiON-2?")

    # vLLM picks the model architecture from `architectures`.  Force Gemma2.
    text_cfg = dict(text_cfg)
    text_cfg["architectures"] = ["Gemma2ForCausalLM"]
    text_cfg.setdefault("model_type", "gemma2")
    text_cfg["torch_dtype"] = src_cfg.get("torch_dtype", "bfloat16")

    # Carry over the quantization_config if present (compressed-tensors).
    if src_cfg.get("quantization_config") is not None:
        text_cfg["quantization_config"] = src_cfg["quantization_config"]

    out_cfg_path = os.path.join(args.out, "config.json")
    with open(out_cfg_path, "w") as f:
        json.dump(text_cfg, f, indent=2)
    print(f"  wrote {out_cfg_path}")

    # --- 2) Copy tokenizer + generation_config + chat_template ---
    for fname in ("tokenizer.json", "tokenizer_config.json",
                  "tokenizer.model", "special_tokens_map.json",
                  "generation_config.json", "chat_template.jinja"):
        src_p = os.path.join(args.src, fname)
        if os.path.exists(src_p):
            shutil.copy2(src_p, os.path.join(args.out, fname))
            print(f"  copied {fname}")

    # --- 3) Read each safetensors shard, keep only text_decoder.* keys,
    #         strip the prefix, and write to a new shard ---
    sf_files = sorted(glob.glob(os.path.join(args.src, "*.safetensors")))
    if not sf_files:
        raise SystemExit(f"no *.safetensors in {args.src}")

    n_kept, n_dropped = 0, 0
    out_sd = {}                  # {new_key: tensor}
    skip_lm_head = False
    for sf in sf_files:
        with safe_open(sf, framework="pt") as f:
            for k in f.keys():
                if not k.startswith(args.prefix):
                    n_dropped += 1
                    continue
                new_k = k[len(args.prefix):]
                # MERaLiON keeps lm_head outside the gemma2 model (tied to
                # embed_tokens).  Gemma2ForCausalLM expects `lm_head.weight`
                # at top level; that mapping already matches if we just strip
                # the prefix.
                t = f.get_tensor(k)
                out_sd[new_k] = t
                n_kept += 1

    # Save as a single shard (3B-class, ~6 GB bf16 / ~2 GB W4A16; fits one shard).
    out_sf = os.path.join(args.out, "model.safetensors")
    save_file(out_sd, out_sf, metadata={"format": "pt"})
    print(f"  kept {n_kept} text_decoder keys, dropped {n_dropped} non-text keys")
    print(f"  wrote {out_sf}  ({os.path.getsize(out_sf)/1e9:.2f} GB)")

    print("\nDone. Now run vLLM directly on this dir:")
    print(f"  python bench_marlin_vllm.py --bf16 ... --w4a16 {args.out}")


if __name__ == "__main__":
    main()
