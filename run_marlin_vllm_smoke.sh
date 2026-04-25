#!/usr/bin/env bash
# Marlin vLLM smoke test on A100. Run inside the marlin_test conda env.
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKDIR"

# Default to the env we just built; user can override.
PYTHON_PATH=${PYTHON_PATH:-python}

BF16_MODEL=${BF16_MODEL:-/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B}
[ -d "$BF16_MODEL" ] || BF16_MODEL=/home/kaixin/programs/LLM_base_model/MERaLiON-2-3B
W4A16_MODEL=${W4A16_MODEL:-$WORKDIR/quant_checkpoints/MERaLiON-2-3B-W4A16-RTN}
[ -d "$W4A16_MODEL" ] || W4A16_MODEL=/home/jinchao/runtao/LLM-Pruner/quant_checkpoints/MERaLiON-2-3B-W4A16-RTN

MAX_TOKENS=${MAX_TOKENS:-256}

if [ -z "${GPU:-}" ]; then
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
          | sort -t, -k2 -n -r | head -n1 | cut -d, -f1)
fi
echo "Using GPU $GPU"
echo "  bf16 :  $BF16_MODEL"
echo "  w4a16:  $W4A16_MODEL"
[ -d "$BF16_MODEL"  ] || { echo "ERROR: $BF16_MODEL not found";  exit 1; }
[ -d "$W4A16_MODEL" ] || { echo "ERROR: $W4A16_MODEL not found"; exit 1; }

# vLLM env hygiene: avoid runtime warnings, keep things deterministic.
export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

"$PYTHON_PATH" -u "$WORKDIR/bench_marlin_vllm.py" \
    --bf16  "$BF16_MODEL" \
    --w4a16 "$W4A16_MODEL" \
    --max_tokens "$MAX_TOKENS"
