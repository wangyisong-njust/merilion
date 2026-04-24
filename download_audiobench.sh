#!/bin/bash
# Bulk-download AudioBench test sets from HF Hub, saving only the first
# N samples of each (or a slice starting at --start_idx).
#
# All AudioBench datasets are hosted under the AudioLLMs/ org on HF;
# repo name == DATASET= value in supported_datasets.md.
#
# Usage:
#   bash download_audiobench.sh                 # defaults: 200 samples each, ASR-English group
#   GROUP=asr_singlish bash download_audiobench.sh
#   NUM_SAMPLES=500 START_IDX=0 OUTPUT_ROOT=/my/path bash download_audiobench.sh
#   DATASETS="librispeech_test_clean imda_part1_asr_test" bash download_audiobench.sh
# ============================================================
set -e

PYTHON_PATH=${PYTHON_PATH:-$(command -v python)}
WORKDIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKDIR/audiobench_data"}
# Default 10000 matches the "hold out the first 10000 as eval" convention
# used by the bench scripts (shuffle(seed=42)[10500:10520]).  For small
# test datasets (a few hundred samples) this will skip everything — set
# START_IDX=0 explicitly for those.
START_IDX=${START_IDX:-10000}
NUM_SAMPLES=${NUM_SAMPLES:-200}
SPLIT=${SPLIT:-test}           # most AudioBench repos use `test`
CLEAR_CACHE=${CLEAR_CACHE:-0}  # 1 = rm -rf ~/.cache/huggingface/datasets after each

# ── Dataset groups (copy/paste from supported_datasets.md) ────────────────────
declare -A DS_GROUPS
DS_GROUPS[asr_english]="librispeech_test_clean librispeech_test_other common_voice_15_en_test peoples_speech_test gigaspeech_test tedlium3_test tedlium3_long_form_test earnings21_test earnings22_test"
DS_GROUPS[asr_singlish]="imda_part1_asr_test imda_part2_asr_test imda_part3_30s_asr_test imda_part4_30s_asr_test imda_part5_30s_asr_test imda_part6_30s_asr_test"
DS_GROUPS[asr_mandarin]="aishell_asr_zh_test"
DS_GROUPS[speech_translation]="covost2_en_id_test covost2_en_zh_test covost2_en_ta_test covost2_id_en_test covost2_zh_en_test covost2_ta_en_test"
DS_GROUPS[sqa]="cn_college_listen_mcq_test slue_p2_sqa5_test dream_tts_mcq_test public_sg_speech_qa_test spoken_squad_test imda_part3_30s_sqa_human_test imda_part4_30s_sqa_human_test imda_part5_30s_sqa_human_test imda_part6_30s_sqa_human_test"
DS_GROUPS[spoken_dialog_summ]="imda_part3_30s_ds_human_test imda_part4_30s_ds_human_test imda_part5_30s_ds_human_test imda_part6_30s_ds_human_test"
DS_GROUPS[speech_instr]="openhermes_audio_test alpaca_audio_test spoken-mqa_short_digit spoken-mqa_long_digit spoken-mqa_single_step_reasoning spoken-mqa_multi_step_reasoning"
DS_GROUPS[aqa]="clotho_aqa_test wavcaps_qa_test audiocaps_qa_test"
DS_GROUPS[captioning]="wavcaps_test audiocaps_test"
DS_GROUPS[emotion]="iemocap_emotion_test meld_sentiment_test meld_emotion_test"
DS_GROUPS[accent]="voxceleb_accent_test imda_ar_sentence imda_ar_dialogue"
DS_GROUPS[gender]="voxceleb_gender_test iemocap_gender_test imda_gr_sentence imda_gr_dialogue"
DS_GROUPS[music]="mu_chomusic_test"
DS_GROUPS[code_switch]="seame_dev_man seame_dev_sge"
DS_GROUPS[all]="${DS_GROUPS[asr_english]} ${DS_GROUPS[asr_singlish]} ${DS_GROUPS[asr_mandarin]} ${DS_GROUPS[speech_translation]} ${DS_GROUPS[sqa]} ${DS_GROUPS[spoken_dialog_summ]} ${DS_GROUPS[speech_instr]} ${DS_GROUPS[aqa]} ${DS_GROUPS[captioning]} ${DS_GROUPS[emotion]} ${DS_GROUPS[accent]} ${DS_GROUPS[gender]} ${DS_GROUPS[music]} ${DS_GROUPS[code_switch]}"

# ── Pick dataset list ─────────────────────────────────────────────────────────
GROUP=${GROUP:-asr_english}
if [ -n "$DATASETS" ]; then
    TARGETS="$DATASETS"
else
    TARGETS="${DS_GROUPS[$GROUP]}"
fi
if [ -z "$TARGETS" ]; then
    echo "No targets (unknown GROUP=$GROUP and empty DATASETS)"; exit 1
fi

echo "Output root : $OUTPUT_ROOT"
echo "Slice       : samples [$START_IDX, $START_IDX + $NUM_SAMPLES)"
echo "Split       : $SPLIT"
echo "Datasets    :"
for d in $TARGETS; do echo "  - $d"; done

mkdir -p "$OUTPUT_ROOT"
for name in $TARGETS; do
    out="$OUTPUT_ROOT/$name"
    if [ -d "$out" ] && [ "$(ls -A "$out" 2>/dev/null)" ]; then
        echo "[skip] $out already present"; continue
    fi
    echo
    echo "[$name] downloading slice …"
    flags=()
    [ "$CLEAR_CACHE" = "1" ] && flags+=(--clear_cache)
    "$PYTHON_PATH" -u "$WORKDIR/download_subset.py" \
        --repo "AudioLLMs/$name" \
        --split "$SPLIT" \
        --start_idx "$START_IDX" \
        --num_samples "$NUM_SAMPLES" \
        --output "$out" \
        "${flags[@]}" || { echo "  [FAIL] $name"; continue; }
done

echo
echo "Summary (files under $OUTPUT_ROOT):"
du -sh "$OUTPUT_ROOT"/*/  2>/dev/null | sort -h
