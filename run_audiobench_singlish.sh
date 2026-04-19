#!/bin/bash
# ============================================================
# AudioBench Singlish WER evaluation (full dataset):
#   1. MERaLiON-2-3B original BF16   → GPU6
#   2. MERaLiON-2-3B MLX-4bit sim    → GPU6
#   3. Pruned mid3-22 + BnB INT8      → GPU7
#
# Datasets: IMDA Parts 1–6 from MERaLiON/Multitask-National-Speech-Corpus-v1
# Download first: python download_audiobench_datasets.py
# ============================================================
export PYTHONUNBUFFERED=1
PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
ORIGINAL="/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B"
TUNE_ROOT="meralion_tune_log"
PRUNED="${TUNE_ROOT}/MERaLiON-2-3B-v3-td50-mid3-22-tune"
DATASET_ROOT="/home/jinchao/runtao/meralion_datasets/ASR"
BATCH_SIZE=256

# Which AudioBench Singlish parts to evaluate (space-separated)
PARTS="imda_part2_asr_test imda_part3_30s_asr_test"

cd "$WORKDIR"

for PART in $PARTS; do
    DATASET="${DATASET_ROOT}/${PART}"
    if [ ! -d "$DATASET" ]; then
        echo "[WARN] Dataset not found: $DATASET — run download_audiobench_datasets.py first"
        continue
    fi

    echo ""
    echo "========================================"
    echo "  Dataset: $PART"
    echo "========================================"

    PID1=""
    if [ -f "ab_${PART}_original_bf16.json" ]; then
        echo "  [skip] ab_${PART}_original_bf16.json"
    else
        echo "  launching original BF16 (GPU6) ..."
        CUDA_VISIBLE_DEVICES=6 "$PYTHON_PATH" -u eval_wer_batch.py \
            --model "$ORIGINAL" --quant bf16 \
            --dataset "$DATASET" --batch_size "$BATCH_SIZE" \
            --audiobench --device cuda \
            --output "ab_${PART}_original_bf16.json" \
            > "ab_${PART}_original_bf16.log" 2>&1 &
        PID1=$!
    fi

    PID2=""
    if [ -f "ab_${PART}_mlx4.json" ]; then
        echo "  [skip] ab_${PART}_mlx4.json"
    else
        echo "  launching MLX-4bit (GPU6) ..."
        CUDA_VISIBLE_DEVICES=6 "$PYTHON_PATH" -u eval_wer_batch.py \
            --model "$ORIGINAL" --quant mlx4 \
            --dataset "$DATASET" --batch_size "$BATCH_SIZE" \
            --audiobench --device cuda \
            --output "ab_${PART}_mlx4.json" \
            > "ab_${PART}_mlx4.log" 2>&1 &
        PID2=$!
    fi

    PID3=""
    if [ -f "ab_${PART}_pruned_int8.json" ]; then
        echo "  [skip] ab_${PART}_pruned_int8.json"
    else
        echo "  launching Pruned+INT8 (GPU7) ..."
        CUDA_VISIBLE_DEVICES=7 "$PYTHON_PATH" -u eval_wer_batch.py \
            --model "$PRUNED" --quant int8 \
            --dataset "$DATASET" --batch_size "$BATCH_SIZE" \
            --audiobench --device cuda \
            --output "ab_${PART}_pruned_int8.json" \
            > "ab_${PART}_pruned_int8.log" 2>&1 &
        PID3=$!
    fi

    echo "  Waiting ...  (tail -f ab_${PART}_*.log to monitor)"
    [ -n "$PID1" ] && wait $PID1 && echo "  [done] original BF16"
    [ -n "$PID2" ] && wait $PID2 && echo "  [done] MLX-4bit"
    [ -n "$PID3" ] && wait $PID3 && echo "  [done] Pruned+INT8"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Summary"
echo "========================================"

"$PYTHON_PATH" - <<'PYEOF'
import json, os, glob

parts = [
    ("imda_part1_asr_test",    "PART1"),
    ("imda_part2_asr_test",    "PART2"),
    ("imda_part3_30s_asr_test","PART3"),
]
models = [
    ("original_bf16", "Original BF16"),
    ("mlx4",          "MLX-4bit sim"),
    ("pruned_int8",   "Pruned+INT8"),
]

hdr = f"  {'Dataset':<28} {'Model':<18} {'WER%':>6} {'VRAM(GB)':>9}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))

for part_key, part_label in parts:
    ref_wer = None
    for model_key, model_label in models:
        path = f"ab_{part_key}_{model_key}.json"
        if not os.path.exists(path):
            print(f"  {part_label:<28} {model_label:<18}  [missing]")
            continue
        with open(path) as f:
            d = json.load(f)
        wer  = d.get("wer", 0) * 100
        vram = d.get("gpu_mem_peak_gb") or 0
        if ref_wer is None:
            ref_wer = wer
        delta = f"+{wer-ref_wer:.2f}" if wer > ref_wer else f"{wer-ref_wer:.2f}"
        print(f"  {part_label:<28} {model_label:<18} {wer:6.2f}  {delta:>7}  {vram:9.2f}")
    print()
PYEOF
