#!/usr/bin/env bash
# Build the HF distribution dir by copying model weights into the package.
#
# Inputs (override via env vars):
#   GPTQ_MARLIN_DIR  — quant_checkpoints/MERaLiON-2-3B-W4A16-GPTQ-Marlin
#                      (output of quantize_gptq_marlin.py)
#   EAGLE_CKPT       — eagle_best.pt produced by run_eagle_train.sh
#   OUT_DIR          — destination (default: ./hf_eagle_pkg_built)
#
# Usage:
#   GPTQ_MARLIN_DIR=quant_checkpoints/MERaLiON-2-3B-W4A16-GPTQ-Marlin \
#   EAGLE_CKPT=eagle_best.pt \
#   bash build.sh
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_REPO="$(cd "$WORKDIR/.." && pwd)"            # main merilion repo

GPTQ_MARLIN_DIR=${GPTQ_MARLIN_DIR:?set GPTQ_MARLIN_DIR=quant_checkpoints/...}
EAGLE_CKPT=${EAGLE_CKPT:?set EAGLE_CKPT=path/to/eagle_best.pt}
OUT_DIR=${OUT_DIR:-$WORKDIR/_built}

PYTHON_PATH=${PYTHON_PATH:-python}

echo "Building HF distribution dir at $OUT_DIR"
mkdir -p "$OUT_DIR"

# 1) Copy code files (Python modules + READMEs + requirements)
for f in README.md requirements.txt modeling_eagle.py example_inference.py; do
    cp -v "$WORKDIR/$f" "$OUT_DIR/"
done

# 2) Copy the EAGLE module + custom MERaLiON modeling code, vendored
mkdir -p "$OUT_DIR/meralion2_bl"
cp -v "$SRC_REPO/eagle_model.py" "$OUT_DIR/eagle_model.py"
for f in "$SRC_REPO"/meralion2_bl/*.py; do
    cp -v "$f" "$OUT_DIR/meralion2_bl/"
done

# 3) Convert eagle_best.pt → eagle.safetensors + eagle_config.json
"$PYTHON_PATH" - <<PYEOF
import json, os, torch
from safetensors.torch import save_file

ckpt_path = "$EAGLE_CKPT"
out_dir   = "$OUT_DIR"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt["eagle_state"]
# safetensors requires contiguous + no shared mem
sd = {k: v.detach().contiguous() for k, v in sd.items()}
save_file(sd, os.path.join(out_dir, "eagle.safetensors"),
          metadata={"format": "pt"})

meta = {
    "num_layers":  ckpt.get("num_layers", 1),
    "step":        ckpt.get("step", -1),
    "val_acc":     ckpt.get("val_acc"),
    "val_ms_acc":  ckpt.get("val_ms_acc"),
    "K_default":   4,
}
with open(os.path.join(out_dir, "eagle_config.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"  wrote eagle.safetensors ({sum(v.numel() for v in sd.values())/1e6:.1f}M params)")
PYEOF

# 4) Copy the Gemma2-only W4A16 sub-dir.  The quantize_gptq_marlin.py output
#    has a `_gemma2_only` cache built by load_gptq_marlin.py; if it doesn't
#    exist yet, build it first.
GEMMA2_DIR="${GPTQ_MARLIN_DIR%/}_gemma2_only"
if [ ! -d "$GEMMA2_DIR" ] || [ ! -s "$GEMMA2_DIR/model.safetensors" ]; then
    echo "Building Gemma2-only cache at $GEMMA2_DIR …"
    "$PYTHON_PATH" - <<PYEOF
import sys
sys.path.insert(0, "$SRC_REPO")
from load_gptq_marlin import _extract_gemma2_dir
_extract_gemma2_dir("$GPTQ_MARLIN_DIR", "$GEMMA2_DIR")
PYEOF
fi

mkdir -p "$OUT_DIR/text_decoder_w4a16"
for f in config.json quantize_config.json model.safetensors \
         tokenizer.json tokenizer_config.json tokenizer.model \
         special_tokens_map.json generation_config.json; do
    if [ -e "$GEMMA2_DIR/$f" ]; then
        cp -v "$GEMMA2_DIR/$f" "$OUT_DIR/text_decoder_w4a16/"
    fi
done

# 5) Copy the patch script so users can rebuild auto-gptq
cp -v "$SRC_REPO/patch_autogptq_marlin_only.py" "$OUT_DIR/"
cp -v "$SRC_REPO/setup_cuda_includes.sh" "$OUT_DIR/" 2>/dev/null || true

echo
echo "============================================================"
echo "Built distribution at:  $OUT_DIR"
echo "Total size:             $(du -sh "$OUT_DIR" | cut -f1)"
echo "Files:"
( cd "$OUT_DIR" && find . -maxdepth 2 -type f | sort )
echo
echo "Next steps:"
echo "  1. cd $OUT_DIR"
echo "  2. Test locally:"
echo "       python example_inference.py <some.wav> --repo $OUT_DIR"
echo "  3. Upload to HF:"
echo "       huggingface-cli upload <user>/MERaLiON-2-3B-EAGLE-W4A16 $OUT_DIR ."
