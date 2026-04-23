#!/bin/bash
# ============================================================
# Quantize + 3-way benchmark: bf16 / W4A16 / bf16+Medusa
#
# 1. Produce W4A16 compressed-tensors checkpoints from the bf16
#    base model (RTN by default, optional GPTQ/AWQ).
# 2. Run bench_3way_a100.py on IMDA_PART1 to compare all three
#    configs side-by-side.
#
# Defaults are set for the kaixin machine; override any variable
# inline (e.g. `GPU=0 METHOD=GPTQ bash run_w4a16_medusa_bench.sh`).
# ============================================================
export PYTHONUNBUFFERED=1

# ── Paths / env ────────────────────────────────────────────────────────────────
PYTHON_PATH=${PYTHON_PATH:-/home/kaixin/anaconda3/envs/llm_pruner_meralion/bin/python}
WORKDIR=${WORKDIR:-/home/kaixin/yisong/merilion}
ORIGINAL=${ORIGINAL:-/home/kaixin/programs/LLM_base_model/MERaLiON-2-3B}
DATASET=${DATASET:-/home/kaixin/ssd/data/ASR/IMDA_PART1_mono_en_30_ASR}
CALIB_DS=${CALIB_DS:-/home/kaixin/meralion_datasets/train/ASR/IMDA_PART1_mono_en_30_ASR}
MEDUSA_SRC=${MEDUSA_SRC:-/home/kaixin/yisong/merilion/hf_medusa_pkg}
QUANT_ROOT=${QUANT_ROOT:-/home/kaixin/yisong/merilion/quant_checkpoints}

# ── Knobs ──────────────────────────────────────────────────────────────────────
METHOD=${METHOD:-RTN}            # RTN | GPTQ | AWQ
NUM_CALIB=${NUM_CALIB:-512}
CALIB_SEQ_LEN=${CALIB_SEQ_LEN:-512}

NUM_SAMPLES=${NUM_SAMPLES:-20}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
GPU=${GPU:-2}
FORCE=${FORCE:-0}               # 1 = re-quantize + re-bench even if outputs exist
SEQUENTIAL=${SEQUENTIAL:-0}     # 1 = pass --sequential to the 3-way bench (VRAM-constrained GPUs)

export CUDA_VISIBLE_DEVICES=$GPU
cd "$WORKDIR"

W4A16_DIR="${QUANT_ROOT}/MERaLiON-2-3B-W4A16-${METHOD}"
BENCH_OUT="${WORKDIR}/bench_3way_${METHOD}_g${GPU}.json"

echo "========================================"
echo "  Config"
echo "========================================"
echo "  method       : $METHOD"
echo "  base         : $ORIGINAL"
echo "  w4a16 out    : $W4A16_DIR"
echo "  medusa src   : $MEDUSA_SRC"
echo "  dataset      : $DATASET"
echo "  calib ds     : ${CALIB_DS:-'(unused for RTN)'}"
echo "  num_samples  : $NUM_SAMPLES"
echo "  GPU          : $GPU   (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  force rebuild: $FORCE"
echo "  sequential   : $SEQUENTIAL"

# ════════════════════════════════════════════════════════════════════════════
echo
echo "========================================"
echo "  Step 1/2: W4A16 quantize ($METHOD)"
echo "========================================"

if [ "$FORCE" != "1" ] && [ -f "$W4A16_DIR/model.safetensors.index.json" ]; then
    echo "  [skip] $W4A16_DIR already has a quantized checkpoint"
else
    mkdir -p "$QUANT_ROOT"
    QUANT_ARGS=(--model "$ORIGINAL" --output_dir "$W4A16_DIR" --method "$METHOD")
    if [ "$METHOD" != "RTN" ]; then
        QUANT_ARGS+=(--calib_ds "$CALIB_DS" --num_calib "$NUM_CALIB" --calib_seq_len "$CALIB_SEQ_LEN")
    fi
    "$PYTHON_PATH" -u quantize_w4a16.py "${QUANT_ARGS[@]}" \
        | tee "${W4A16_DIR%.*}.quantize.log"
fi

# ════════════════════════════════════════════════════════════════════════════
echo
echo "========================================"
echo "  Step 2/2: 3-way benchmark"
echo "========================================"

BENCH_ARGS=(
    --base_bf16     "$ORIGINAL"
    --base_w4a16    "$W4A16_DIR"
    --medusa_source "$MEDUSA_SRC"
    --dataset       "$DATASET"
    --num_samples   "$NUM_SAMPLES"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --output        "$BENCH_OUT"
)
[ "$SEQUENTIAL" = "1" ] && BENCH_ARGS+=(--sequential)

if [ "$FORCE" != "1" ] && [ -f "$BENCH_OUT" ]; then
    echo "  [skip] $BENCH_OUT already exists (set FORCE=1 to re-run)"
else
    "$PYTHON_PATH" -u bench_3way_a100.py "${BENCH_ARGS[@]}" \
        | tee "${BENCH_OUT%.json}.log"
fi

# ════════════════════════════════════════════════════════════════════════════
echo
echo "========================================"
echo "  Summary"
echo "========================================"
if [ -f "$BENCH_OUT" ]; then
    "$PYTHON_PATH" -c "
import json
with open('$BENCH_OUT') as f: d = json.load(f)
s = d['summary']
print(f'{\"\":<22} {\"Lat(s)\":>8} {\"TPS\":>7} {\"WER%\":>6} {\"VRAM(GB)\":>10}  notes')
print('-' * 72)
for key, label in [('bf16', 'bf16 baseline'),
                    ('w4a16', 'W4A16 ($METHOD)'),
                    ('medusa', 'bf16 + Medusa')]:
    r = s[key]
    note = r.get('linear_class', '')
    print(f'{label:<22} {r[\"lat_s\"]:>8.3f} {r[\"tps\"]:>7.2f} {r[\"wer\"]*100:>5.2f}% {r[\"vram_gb\"]:>10.2f}  {note}')
print()
bf16_lat = s['bf16']['lat_s']; bf16_tps = s['bf16']['tps']
for key, label in [('w4a16', 'W4A16'), ('medusa', 'Medusa')]:
    r = s[key]
    print(f'  {label} latency speedup: {bf16_lat / max(r[\"lat_s\"], 1e-6):.2f}x   '
          f'throughput speedup: {r[\"tps\"] / max(bf16_tps, 1e-6):.2f}x')
"
else
    echo "  (no bench output found)"
fi
