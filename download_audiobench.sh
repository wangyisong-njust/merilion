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

PYTHON_PATH=${PYTHON_PATH:-/home/jinchao/miniconda3/envs/audiobench_quant/bin/python}
WORKDIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
# Default output root matches the convention used by the bench scripts
# (run_draft_spec_bench.sh, run_w4a16_medusa_bench.sh DATASET=...) on the
# jinchao remote.  Datasets land under $OUTPUT_ROOT/<dataset_name>.
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/jinchao/runtao/meralion_datasets/ASR}
# TAKE_LAST=N asks download_subset.py to auto-query each dataset's total
# length and grab the LAST N rows (start_idx = total - N).  Small
# datasets just return everything they have.  Set TAKE_LAST=0 to fall
# back to the fixed-slice mode (START_IDX + NUM_SAMPLES).
TAKE_LAST=${TAKE_LAST:-10000}
START_IDX=${START_IDX:-0}         # only used when TAKE_LAST=0
NUM_SAMPLES=${NUM_SAMPLES:-200}   # only used when TAKE_LAST=0
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
if [ "$TAKE_LAST" -gt 0 ]; then
    echo "Slice       : last $TAKE_LAST samples per dataset (dynamic per size)"
else
    echo "Slice       : samples [$START_IDX, $START_IDX + $NUM_SAMPLES)"
fi
echo "Split       : $SPLIT"
echo "Datasets    :"
for d in $TARGETS; do echo "  - $d"; done

# ── Name → (repo, config, split) aliasing ─────────────────────────────────────
# Some AudioBench labels (supported_datasets.md) don't match the HF repo name.
# IMDA / NSC parts are actually configs of a single Multitask-National-
# Speech-Corpus-v1-extend repo, and each config has only a `train` split
# (the "-Test" suffix lives in the config name, not the split).
# Map label → "repo|config|split"  (config/split empty = use default).
declare -A DS_ALIAS
NSC="AudioLLMs/Multitask-National-Speech-Corpus-v1-extend"
# IMDA ASR
DS_ALIAS[imda_part1_asr_test]="$NSC|ASR-PART1-Test|train"
DS_ALIAS[imda_part2_asr_test]="$NSC|ASR-PART2-Test|train"
DS_ALIAS[imda_part3_30s_asr_test]="$NSC|ASR-PART3-Test|train"
DS_ALIAS[imda_part4_30s_asr_test]="$NSC|ASR-PART4-Test|train"
DS_ALIAS[imda_part5_30s_asr_test]="$NSC|ASR-PART5-Test|train"
DS_ALIAS[imda_part6_30s_asr_test]="$NSC|ASR-PART6-Test|train"
# IMDA SQA
DS_ALIAS[imda_part3_30s_sqa_human_test]="$NSC|SQA-PART3-Test|train"
DS_ALIAS[imda_part4_30s_sqa_human_test]="$NSC|SQA-PART4-Test|train"
DS_ALIAS[imda_part5_30s_sqa_human_test]="$NSC|SQA-PART5-Test|train"
DS_ALIAS[imda_part6_30s_sqa_human_test]="$NSC|SQA-PART6-Test|train"
# IMDA Spoken-Dialog-Summ (NB: PART6-Test does not exist on HF — only Train)
DS_ALIAS[imda_part3_30s_ds_human_test]="$NSC|SDS-PART3-Test|train"
DS_ALIAS[imda_part4_30s_ds_human_test]="$NSC|SDS-PART4-Test|train"
DS_ALIAS[imda_part5_30s_ds_human_test]="$NSC|SDS-PART5-Test|train"
# IMDA Accent / Gender
DS_ALIAS[imda_ar_sentence]="$NSC|PQA-AR-Sentence-Test|train"
DS_ALIAS[imda_ar_dialogue]="$NSC|PQA-AR-Dialogue-Test|train"
DS_ALIAS[imda_gr_sentence]="$NSC|PQA-GR-Sentence-Test|train"
DS_ALIAS[imda_gr_dialogue]="$NSC|PQA-GR-Dialogue-Test|train"
# Other rename cases
DS_ALIAS[aishell_asr_zh_test]="AudioLLMs/aishell_1_zh_test||"
DS_ALIAS[openhermes_audio_test]="AudioLLMs/openhermes_instruction_test||"
DS_ALIAS[iemocap_emotion_test]="AudioLLMs/iemocap_emotion_recognition||"
DS_ALIAS[iemocap_gender_test]="AudioLLMs/iemocap_gender_recognition||"

mkdir -p "$OUTPUT_ROOT"
for name in $TARGETS; do
    out="$OUTPUT_ROOT/$name"
    # Skip if this dataset is already materialised (save_to_disk drops a
    # state.json + dataset_info.json).  A raw empty dir or a partial
    # download without state.json will NOT skip, so we can resume.
    if [ -f "$out/state.json" ] || [ -f "$out/dataset_info.json" ]; then
        sz=$(du -sh "$out" 2>/dev/null | awk '{print $1}')
        echo "[skip] $name already present at $out ($sz)"
        continue
    fi
    # Also warn if non-empty but not a valid Dataset artefact — likely
    # partial / crashed previous run.
    if [ -d "$out" ] && [ "$(ls -A "$out" 2>/dev/null)" ]; then
        echo "[resume] $out exists but has no state.json — re-downloading"
        rm -rf "$out"
    fi
    echo
    echo "[$name] downloading slice …"
    # Resolve (repo, config, split) via alias if one exists; else default
    # to AudioLLMs/<name> with no config, using the env-level SPLIT.
    if [ -n "${DS_ALIAS[$name]}" ]; then
        IFS='|' read -r repo config split <<< "${DS_ALIAS[$name]}"
        [ -z "$split" ] && split="$SPLIT"
    else
        repo="AudioLLMs/$name"
        config=""
        split="$SPLIT"
    fi

    flags=(--repo "$repo" --split "$split")
    [ -n "$config" ] && flags+=(--subsets "$config")
    [ "$CLEAR_CACHE" = "1" ] && flags+=(--clear_cache)
    if [ "$TAKE_LAST" -gt 0 ]; then
        flags+=(--take_last "$TAKE_LAST")
    else
        flags+=(--start_idx "$START_IDX" --num_samples "$NUM_SAMPLES")
    fi
    echo "  repo=$repo${config:+  config=$config}  split=$split"
    "$PYTHON_PATH" -u "$WORKDIR/download_subset.py" \
        --output "$out" \
        "${flags[@]}" || { echo "  [FAIL] $name"; continue; }
done

echo
echo "Summary (files under $OUTPUT_ROOT):"
du -sh "$OUTPUT_ROOT"/*/  2>/dev/null | sort -h
