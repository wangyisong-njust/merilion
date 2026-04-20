#!/bin/bash
# ============================================================
# WER comparison — AudioBench IMDA Parts 1-6, 100 samples each
#   1. Original BF16             → GPU6
#   2. MLX-4bit sim (int4 g=64)  → GPU6  (after 1)
#   3. Pruned mid3-22 BF16       → GPU7
#   4. Pruned mid3-22 + INT8     → GPU7  (after 3)
#
# Prerequisite: python download_audiobench_datasets.py
# ============================================================
export PYTHONUNBUFFERED=1
PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
ORIGINAL="/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B"
TUNE_ROOT="meralion_tune_log"
PRUNED="${TUNE_ROOT}/MERaLiON-2-3B-v3-td50-mid3-22-tune"
DATASET_ROOT="/home/jinchao/runtao/meralion_datasets/ASR"
NUM_SAMPLES=100
BATCH_SIZE=128

PARTS="imda_part1_asr_test imda_part2_asr_test imda_part3_30s_asr_test \
       imda_part4_30s_asr_test imda_part5_30s_asr_test imda_part6_30s_asr_test"

cd "$WORKDIR"

_run() {
    local json="$1"; local gpu="$2"; shift 2
    local resume=""
    [ -f "$json" ] && resume="--resume $json" && echo "  resuming  → $json" \
                   || echo "  launching → $json"
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_PATH" -u eval_wer_batch.py \
        --dataset "${DATASET_ROOT}/$PART" \
        --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE \
        --audiobench --output "$json" $resume \
        "$@" >> "${json%.json}.log" 2>&1
}

# ── GPU6 track: original_bf16 then mlx4 ──────────────────────────────────────
(
for PART in $PARTS; do
    _run "wc_${PART}_original_bf16.json" 6 --model "$ORIGINAL" --quant bf16
done
for PART in $PARTS; do
    _run "wc_${PART}_mlx4.json" 6 --model "$ORIGINAL" --quant mlx4
done
) &
PID_GPU6=$!

# ── GPU7 track: pruned_bf16 then pruned_int8 ─────────────────────────────────
(
for PART in $PARTS; do
    _run "wc_${PART}_pruned_bf16.json" 7 --model "$PRUNED" --quant bf16
done
for PART in $PARTS; do
    _run "wc_${PART}_pruned_int8.json" 7 --model "$PRUNED" --quant int8
done
) &
PID_GPU7=$!

echo "GPU6 PID=$PID_GPU6  GPU7 PID=$PID_GPU7"
echo "Waiting …  (tail -f wc_*.log to monitor)"
wait $PID_GPU6 && echo "[done] GPU6 track"
wait $PID_GPU7 && echo "[done] GPU7 track"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Summary  (WER on overlapping samples)"
echo "========================================"

"$PYTHON_PATH" - <<'PYEOF'
import json, os

PARTS = [
    ("imda_part1_asr_test",    "PART1"),
    ("imda_part2_asr_test",    "PART2"),
    ("imda_part3_30s_asr_test","PART3"),
    ("imda_part4_30s_asr_test","PART4"),
    ("imda_part5_30s_asr_test","PART5"),
    ("imda_part6_30s_asr_test","PART6"),
]
MODELS = [
    ("original_bf16", "Original BF16"),
    ("mlx4",          "MLX-4bit"),
    ("pruned_bf16",   "Pruned BF16"),
    ("pruned_int8",   "Pruned+INT8"),
]

def wer_on_samples(samples, n):
    """Corpus WER on the first n samples using jiwer."""
    from jiwer import compute_measures
    import re
    def norm(t):
        import re as _re, jiwer as _j
        t = t.lower()
        for d, w in [("0","zero"),("1","one"),("2","two"),("3","three"),("4","four"),
                     ("5","five"),("6","six"),("7","seven"),("8","eight"),("9","nine")]:
            t = _re.sub(r'\b' + d + r'\b', w, t)
        t = _re.sub(r'[\(\[\{\<][^\n\(\)\[\]\{\}\<\>]*[\)\]\}\>]', "", t)
        t = _j.Compose([_j.RemoveMultipleSpaces(), _j.ExpandCommonEnglishContractions(),
                        _j.RemoveKaldiNonWords(), _j.RemovePunctuation()])(t)
        t = _re.sub(r'\b(uh|umm|um|er|ah)\b', '', t)
        return t.strip() or "empty"
    inc, tot = 0, 0
    for s in samples[:n]:
        p, r = norm(s["prediction"]), norm(s["reference"])
        m = compute_measures(r, p)
        inc += m["substitutions"] + m["deletions"] + m["insertions"]
        tot += m["substitutions"] + m["deletions"] + m["hits"]
    return inc / tot if tot > 0 else 0.0

hdr = f"  {'Part':<8}" + "".join(f" {m:>14}" for _, m in MODELS)
print(hdr)
print("  " + "-" * (len(hdr) - 2))

for part_key, part_label in PARTS:
    data = {}
    for model_key, _ in MODELS:
        path = f"wc_{part_key}_{model_key}.json"
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            data[model_key] = d.get("samples", [])

    if not data:
        print(f"  {part_label:<8}  [no results]")
        continue

    # overlap = min completed samples across available models
    n = min(len(v) for v in data.values())
    row = f"  {part_label:<8}"
    for model_key, _ in MODELS:
        if model_key not in data:
            row += f" {'[missing]':>14}"
        else:
            w = wer_on_samples(data[model_key], n) * 100
            row += f" {w:>13.2f}%"
    row += f"  (n={n})"
    print(row)
PYEOF
