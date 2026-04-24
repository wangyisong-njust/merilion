#!/bin/bash
# ============================================================
# Quick bench: Medusa-accelerated decode vs BF16 no-spec baseline.
# Runs both on the same samples and prints a side-by-side summary.
# ============================================================
set -e
export PYTHONUNBUFFERED=1

PYTHON_PATH=${PYTHON_PATH:-/home/jinchao/miniconda3/envs/audiobench_quant/bin/python}
WORKDIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
MODEL=${MODEL:-/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B}
DATASET=${DATASET:-/home/jinchao/runtao/meralion_datasets/ASR/imda_part1_asr_test}
HEADS=${HEADS:-$WORKDIR/medusa_heads_best.pt}
NUM_SAMPLES=${NUM_SAMPLES:-20}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}

# GPU auto-pick
if [ -z "$GPU" ] || [ "$GPU" = "auto" ]; then
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
          | sort -t, -k2 -n -r | head -1 | cut -d, -f1)
fi
export CUDA_VISIBLE_DEVICES=$GPU
cd "$WORKDIR"

BASELINE_OUT=${BASELINE_OUT:-gpu_bf16_nospec.json}
MEDUSA_OUT=${MEDUSA_OUT:-gpu_medusa_new.json}
FORCE=${FORCE:-0}

echo "========================================"
echo "  Config"
echo "========================================"
echo "  MODEL         : $MODEL"
echo "  HEADS         : $HEADS"
echo "  DATASET       : $DATASET"
echo "  NUM_SAMPLES   : $NUM_SAMPLES"
echo "  GPU           : $GPU"
echo "  baseline out  : $BASELINE_OUT"
echo "  medusa out    : $MEDUSA_OUT"

if [ ! -f "$HEADS" ]; then
    echo "ERROR: heads checkpoint not found: $HEADS"; exit 1
fi

# ── Baseline ──────────────────────────────────────────────────────────────────
echo
echo "========================================"
echo "  [1/2] BF16 no-spec baseline"
echo "========================================"
if [ "$FORCE" != "1" ] && [ -f "$BASELINE_OUT" ]; then
    echo "  [skip] $BASELINE_OUT already exists"
else
    "$PYTHON_PATH" -u infer_gpu.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --num_samples "$NUM_SAMPLES" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --quant bf16 \
        --output "$BASELINE_OUT" \
        --save_samples \
        | tee "${BASELINE_OUT%.json}.log"
fi

# ── Medusa ────────────────────────────────────────────────────────────────────
echo
echo "========================================"
echo "  [2/2] Medusa (heads=$(basename "$HEADS"))"
echo "========================================"
if [ "$FORCE" != "1" ] && [ -f "$MEDUSA_OUT" ]; then
    echo "  [skip] $MEDUSA_OUT already exists"
else
    "$PYTHON_PATH" -u infer_gpu_medusa.py \
        --model "$MODEL" \
        --heads "$HEADS" \
        --dataset "$DATASET" \
        --num_samples "$NUM_SAMPLES" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --output "$MEDUSA_OUT" \
        | tee "${MEDUSA_OUT%.json}.log"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo
echo "========================================"
echo "  Summary"
echo "========================================"
"$PYTHON_PATH" - "$BASELINE_OUT" "$MEDUSA_OUT" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f: b = json.load(f)
with open(sys.argv[2]) as f: m = json.load(f)

lat_b = b.get('avg_latency_s', 0); tps_b = b.get('avg_decode_tps', 0); wer_b = b.get('wer', 0)
lat_m = m.get('avg_latency_s', 0); tps_m = m.get('avg_decode_tps', 0); wer_m = m.get('wer', 0)
acc   = m.get('avg_spec_accept_rate', 0)
vb    = b.get('gpu_mem_peak_gb')
vm    = m.get('gpu_mem_peak_gb')

hdr = f"  {'':<22} {'Lat(s)':>8} {'TPS':>7} {'WER%':>6} {'VRAM(GB)':>10}"
print(hdr); print('  ' + '-' * (len(hdr)-2))
print(f"  {'bf16 baseline':<22} {lat_b:>8.3f} {tps_b:>7.2f} {wer_b*100:>5.2f}% "
      f"{(vb if vb is not None else 0):>10.2f}")
print(f"  {'bf16 + Medusa':<22} {lat_m:>8.3f} {tps_m:>7.2f} {wer_m*100:>5.2f}% "
      f"{(vm if vm is not None else 0):>10.2f}")
print()
print(f"  latency speedup     : {lat_b / max(lat_m, 1e-6):.2f}x")
print(f"  throughput speedup  : {tps_m / max(tps_b, 1e-6):.2f}x")
print(f"  accept rate (Medusa): {acc:.1%}")
PYEOF
