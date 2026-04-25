#!/usr/bin/env bash
# Marlin kernel smoke test: bf16 vs W4A16-RTN, decode tok/s only.
# Pure text path on text_decoder; no audio, no spec.
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKDIR"

PYTHON_PATH=${PYTHON_PATH:-/home/jinchao/miniconda3/envs/audiobench_quant/bin/python}
[ -x "$PYTHON_PATH" ] || PYTHON_PATH=/home/kaixin/anaconda3/envs/audiobench_quant/bin/python

# Models — override via env var if your paths differ.
BF16_MODEL=${BF16_MODEL:-/home/kaixin/programs/LLM_base_model/MERaLiON-2-3B}
[ -d "$BF16_MODEL" ] || BF16_MODEL=/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B
W4A16_MODEL=${W4A16_MODEL:-$WORKDIR/quant_checkpoints/MERaLiON-2-3B-W4A16-RTN}

STEPS=${STEPS:-256}

# Pick the freest GPU.
if [ -z "${GPU:-}" ]; then
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
          | sort -t, -k2 -n -r | head -n1 | cut -d, -f1)
fi
echo "Using GPU $GPU"
echo "  bf16 :  $BF16_MODEL"
echo "  w4a16:  $W4A16_MODEL"
[ -d "$W4A16_MODEL" ] || { echo "ERROR: $W4A16_MODEL not found.  Run quantize_w4a16.py first."; exit 1; }

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_PATH" -u "$WORKDIR/bench_marlin_smoke.py" \
    --bf16  "$BF16_MODEL" \
    --w4a16 "$W4A16_MODEL" \
    --steps "$STEPS"
