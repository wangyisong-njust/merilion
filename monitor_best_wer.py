import json
import os
import glob

def find_best_wer(log_dir):
    wer_file = os.path.join(log_dir, "validation_wer.jsonl")
    if not os.path.exists(wer_file):
        return None, None
    
    best_step = None
    min_wer = float('inf')
    
    with open(wer_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data['wer'] < min_wer:
                    min_wer = data['wer']
                    best_step = data.get('step', 'unknown')
            except:
                continue
    return min_wer, best_step

dirs = glob.glob("/home/jinchao/runtao/LLM-Pruner/meralion_tune_log/*-tune")
print(f"{'Experiment':<50} | {'Best WER':<10} | {'At Step':<10}")
print("-" * 75)

for d in dirs:
    exp_name = os.path.basename(d)
    min_wer, step = find_best_wer(d)
    if min_wer is not None:
        print(f"{exp_name:<50} | {min_wer:<10.2f} | {step:<10}")
    else:
        print(f"{exp_name:<50} | {'No data yet':<10} | {'-':<10}")
