import json

INPUT_JSON = "log_for_all_models/MERaLiON-2-10B-ASR-vllm/imda_part1_asr_test_wer_score.json"
INPUT_JSON = "log_for_all_models/MERaLiON-2-10B-ASR-vllm-W8A16-RTN-textonly/imda_part1_asr_test_wer_score.json"
INPUT_JSON = "log_for_all_models/MERaLiON-2-10B-ASR-vllm-0_25-7-35-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-W8A16-RTN-textonly-fp8KV/imda_part1_asr_test_wer_score.json"
INPUT_JSON = "log_for_all_models/MERaLiON-2-10B-ASR-vllm-0_5-5-40-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged/imda_part1_asr_test_wer_score.json"
INPUT_JSON = "log_for_all_models/MERaLiON-2-10B-ASR-vllm-0_5-5-40-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-W8A16-RTN-textonly-fp8KV/imda_part1_asr_test_wer_score.json"

OUTPUT_JSON = INPUT_JSON.split(".")[0] + "_sorted_40.json"
TOP_K = 40

# Load original json
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = data["sample_wer"]

# Sort by WER (ascending)
sorted_samples = sorted(samples, key=lambda x: x["wer"])

# Top 20 lowest WER
lowest_wer = sorted_samples[:TOP_K]

# Bottom 20 highest WER
highest_wer = sorted_samples[-TOP_K:]

# Create new json structure
new_data = {
    "wer": data.get("wer"),  # keep original overall WER if you want
    "sample_wer": {
        "lowest_wer": lowest_wer,
        "highest_wer": highest_wer
    }
}

# Save to new json file
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)

print(f"Saved top/bottom {TOP_K} WER samples to {OUTPUT_JSON}")
