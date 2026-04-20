#!/bin/bash
# ============================================================
# LLaMA-2-7B GPU speedup benchmark
#   1. BF16  no-spec   (baseline)
#   2. INT8  no-spec
#   3. INT8  +spec γ=5
#
# Measures: avg_latency_s / avg_decode_tps / gpu_mem_peak_gb
# Speedup ratio is computed relative to BF16 no-spec.
# ============================================================
export PYTHONUNBUFFERED=1

PYTHON_PATH="${PYTHON_PATH:-python}"
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_MODEL="${LLAMA_MODEL:-/path/to/Llama-2-7b-hf}"   # override via env
PROMPTS_FILE="${WORKDIR}/llama_prompts.txt"
NUM_SAMPLES=50
GAMMA=5
GPU=0

export CUDA_VISIBLE_DEVICES=$GPU
cd "$WORKDIR"

# ── generate a default prompts file if none exists ────────────────────────────
if [ ! -f "$PROMPTS_FILE" ]; then
    cat > "$PROMPTS_FILE" <<'PROMPTS'
Once upon a time in a land far away, there lived a king who
The history of artificial intelligence began in the
In order to solve the climate crisis, governments around the world must
The quick brown fox jumps over the lazy dog and then
Machine learning is a subset of artificial intelligence that focuses on
The human brain contains approximately 86 billion neurons which
During the Renaissance period, artists began to explore new techniques for
Scientists have discovered that the universe is approximately 13.8 billion years old
The most important invention of the 20th century was undoubtedly
Language models work by predicting the next word given a sequence of
PROMPTS
    # Repeat to reach NUM_SAMPLES
    python3 - "$PROMPTS_FILE" "$NUM_SAMPLES" <<'PYEOF'
import sys
path, n = sys.argv[1], int(sys.argv[2])
with open(path) as f:
    lines = [l.strip() for l in f if l.strip()]
out = (lines * ((n // len(lines)) + 1))[:n]
with open(path, "w") as f:
    f.write("\n".join(out) + "\n")
PYEOF
    echo "  Generated default prompts file: $PROMPTS_FILE"
fi

# ── helper: skip if JSON already complete ─────────────────────────────────────
run_if_missing() {
    local json="$1"; shift
    if [ -f "$json" ]; then
        echo "  [skip] $json already exists"
        return 0
    fi
    echo "  running → $json"
    "$PYTHON_PATH" -u "$WORKDIR/infer_llama_gpu.py" \
        --output "$json" "$@" \
        2>&1 | tee "${json%.json}.log"
    local rc=${PIPESTATUS[0]}
    [ $rc -ne 0 ] && echo "[FAIL] $json (exit $rc)" && return 1
    return 0
}

COMMON=(--model "$LLAMA_MODEL" --prompts_file "$PROMPTS_FILE" --num_samples "$NUM_SAMPLES")

# ════════════════════════════════════════════════════════════════════════════
echo "========================================"
echo "  LLaMA-2-7B  BF16  no-spec  (baseline)"
echo "========================================"
run_if_missing "llama_bf16_nospec.json" \
    "${COMMON[@]}" --quant bf16 || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  LLaMA-2-7B  INT8  no-spec"
echo "========================================"
run_if_missing "llama_int8_nospec.json" \
    "${COMMON[@]}" --quant int8 || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  LLaMA-2-7B  INT8  +spec γ=${GAMMA}"
echo "========================================"
run_if_missing "llama_int8_spec${GAMMA}.json" \
    "${COMMON[@]}" --quant int8 --speculative --gamma "$GAMMA" || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  Summary — speedup vs BF16 no-spec"
echo "========================================"

"$PYTHON_PATH" - <<PYEOF
import json, os

rows = [
    ("BF16   no-spec     (baseline)", "llama_bf16_nospec.json",  False),
    ("INT8   no-spec",                "llama_int8_nospec.json",  False),
    ("INT8   +spec γ=${GAMMA}",        "llama_int8_spec${GAMMA}.json", True),
]

ref_lat = None
hdr = (f"  {'Config':<32} {'Lat(s)':>7} {'Speedup':>8} "
       f"{'tok/s':>7} {'SpecAcc%':>9} {'VRAM(GB)':>9}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))

for label, path, is_spec in rows:
    if not os.path.exists(path):
        print(f"  {label:<32}  [missing]")
        continue
    with open(path) as f:
        d = json.load(f)
    lat  = d.get("avg_latency_s", 0)
    tps  = d.get("avg_decode_tps", 0)
    vram = d.get("gpu_mem_peak_gb")
    acc  = d.get("avg_spec_accept_rate")

    if ref_lat is None and lat > 0:
        ref_lat = lat

    ratio  = (ref_lat / lat) if (ref_lat and lat > 0) else 0.0
    vram_s = f"{vram:.1f}" if vram is not None else "  —"
    acc_s  = f"{acc*100:.1f}%" if acc is not None else "   —"
    marker = " ◀" if is_spec else "  "
    print(f"  {label:<32} {lat:7.2f}  {ratio:7.2f}x  {tps:7.1f}  "
          f"{acc_s:>9}  {vram_s:>9}{marker}")
PYEOF
