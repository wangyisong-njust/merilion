import os
import re
from datasets import load_dataset, Audio
import soundfile as sf
import json
import pdb

# 1) Load dataset
ds = load_dataset(
    "MERaLiON/Multitask-National-Speech-Corpus-v1",
    data_dir="ASR-PART1-Test"
)["train"]

# 2) Inspect columns (run once to find the correct text/audio column names)
print("Columns:", ds.column_names)
print(ds[0]["context"].keys())

# Common guesses; change if your dataset uses different names.
AUDIO_COL_CANDIDATES = ["context"]
TEXT_COL_CANDIDATES  = ["answer"]

audio_col = next((c for c in AUDIO_COL_CANDIDATES if c in ds.column_names), None)
text_col  = next((c for c in TEXT_COL_CANDIDATES  if c in ds.column_names), None)

if audio_col is None:
    raise ValueError(f"No audio-like column found. Got: {ds.column_names}")
if text_col is None:
    raise ValueError(f"No text-like column found. Got: {ds.column_names}")

print("Using audio column:", audio_col)
print("Using text column:", text_col)

# 3) Ensure audio is decoded to arrays (so we can save wav)
# If the column is already Audio, this is fine; if not, it tries to cast.
ds = ds.cast_column(audio_col, Audio(decode=True))

# 4) Query helpers
def find_by_text_exact(ds, query: str, limit: int = 10):
    """Exact string match."""
    query_norm = query.strip()
    matches = ds.filter(lambda ex: ex[text_col].strip() == query_norm)
    return matches.select(range(min(limit, len(matches))))

def find_by_text_substring(ds, query: str, limit: int = 10, case_insensitive: bool = True):
    """Substring match."""
    q = query.strip()
    if case_insensitive:
        q_low = q.lower()
        matches = ds.filter(lambda ex: q_low in ex[text_col].lower())
    else:
        matches = ds.filter(lambda ex: q in ex[text_col])
    return matches.select(range(min(limit, len(matches))))

def find_by_text_regex(ds, pattern: str, limit: int = 10, flags=re.IGNORECASE):
    """Regex match."""
    rx = re.compile(pattern, flags)
    matches = ds.filter(lambda ex: rx.search(ex[text_col]) is not None)
    return matches.select(range(min(limit, len(matches))))

# 5) Save as WAV
def save_rows_as_wav(rows, out_dir: str, prefix: str = "sample", write_sidecar_txt: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for i, ex in enumerate(rows):
        aud = ex[audio_col]  # dict like {"array": ..., "sampling_rate": ...} when decoded
        arr = aud["array"]
        sr = aud["sampling_rate"]

        wav_path = os.path.join(out_dir, f"{prefix}_{i:03d}.wav")
        sf.write(wav_path, arr, sr)

        if write_sidecar_txt:
            txt_path = os.path.join(out_dir, f"{prefix}_{i:03d}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(ex[text_col])

        saved.append((wav_path, sr, ex[text_col][:80] + ("..." if len(ex[text_col]) > 80 else "")))
    return saved

# ---- Example usage ----
query_list = []
# MODEL_NAME = "MERaLiON-2-10B-ASR"
# SORTED_JSON = "log_for_all_models/MERaLiON-2-10B-ASR-vllm/imda_part1_asr_test_wer_score_sorted_20.json"
# MODEL_NAME = "MERaLiON-2-10B-ASR-W8A16-RTN-textonly"
# SORTED_JSON = "log_for_all_models/MERaLiON-2-10B-ASR-vllm-W8A16-RTN-textonly/imda_part1_asr_test_wer_score_sorted_20.json"
# MODEL_NAME = "MERaLiON-2-10B-ASR-vllm-0_25-7-35-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-W8A16-RTN-textonly-fp8KV"
# SORTED_JSON = "log_for_all_models/MERaLiON-2-10B-ASR-vllm-0_25-7-35-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-W8A16-RTN-textonly-fp8KV/imda_part1_asr_test_wer_score_sorted_20.json"
# MODEL_NAME = "MERaLiON-2-10B-ASR-vllm-0_5-5-40-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged"
# SORTED_JSON = "log_for_all_models/MERaLiON-2-10B-ASR-vllm-0_5-5-40-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged/imda_part1_asr_test_wer_score_sorted_20.json"
MODEL_NAME = "MERaLiON-2-10B-ASR-vllm-0_5-5-40-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-W8A16-RTN-textonly-fp8KV"
SORTED_JSON = "log_for_all_models/MERaLiON-2-10B-ASR-vllm-0_5-5-40-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-W8A16-RTN-textonly-fp8KV/imda_part1_asr_test_wer_score_sorted_20.json"
with open(SORTED_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)
samples_good = data["sample_wer"]["lowest_wer"]
samples_bad = data["sample_wer"]["highest_wer"]

for sample in samples_good:
    query = sample["reference"]
    if query not in query_list:
        query_list.append(query)
        rows = find_by_text_substring(ds, query, limit=5)   # or exact / regex
        saved = save_rows_as_wav(rows, out_dir="./saved_audio/"+MODEL_NAME+"_low_wer", prefix=query)

for sample in samples_bad:
    query = sample["reference"]
    if query not in query_list:
        query_list.append(query)
        rows = find_by_text_substring(ds, query, limit=5)   # or exact / regex
        saved = save_rows_as_wav(rows, out_dir="./saved_audio/"+MODEL_NAME+"_high_wer", prefix=query)

# print("Saved files:")
# for wav_path, sr, preview in saved:
#     print(f" - {wav_path} (sr={sr}) | text: {preview}")
