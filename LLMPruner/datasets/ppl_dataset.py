'''
Some of the code refer to
https://github.com/IST-DASLab/gptq/blob/main/datautils.py
'''

import random
import numpy as np
import torch

from datasets import load_dataset
from torch.utils.data.dataset import Dataset

def get_wikitext2(seq_len, tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    return traindata, testdata

def get_ptb(seq_len, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', trust_remote_code=True)
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation', trust_remote_code=True)
    return traindata, valdata

def get_alpaca_chinese(seq_len, tokenizer):
    traindata = load_dataset(
        'silk-road/alpaca-data-gpt4-chinese', split='train'
    )
    return traindata, None

def get_c4(seq_len, tokenizer):
    traindata = load_dataset(
        'allenai/c4', 'zh', split='train'
    )[:100]
    valdata = load_dataset(
        'allenai/c4', 'zh', split='validation'
    )[:100]
    return traindata, valdata

def get_wiki_ch(seq_len, tokenizer):
    valdata = load_dataset("pleisto/wikipedia-cn-20230720-filtered", split='train')[:1000]
    return None, valdata

def get_wmt(seq_len, tokenizer):
    traindata = load_dataset("google/wmt24pp", "en-zh_CN", split="train")
    return traindata, None

class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)

def process_data(samples, tokenizer, seq_len, field_name):
    test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
    test_ids_batch = []
    nsamples = test_ids.numel() // seq_len

    for i in range(nsamples):
        batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
        test_ids_batch.append(batch)
    test_ids_batch = torch.stack(test_ids_batch)
    return IndexDataset(tensors=test_ids_batch)
       

def get_loaders(name, tokenizer, seq_len=2048, batch_size = 8):
    if 'wikitext2' in name:
        train_data, test_data = get_wikitext2(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    if 'ptb' in name:
        train_data, test_data = get_ptb(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    if 'c4' in name:
        train_data, test_data = get_c4(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    if 'alpaca_cn' in name:
        train_data, _ = get_alpaca_chinese(seq_len, tokenizer)
        test_dataset = process_data(train_data, tokenizer, seq_len, 'output_zh')
    if 'wmt' in name:
        train_data, _ = get_wmt(seq_len, tokenizer)
        test_dataset = process_data(train_data, tokenizer, seq_len, 'target')
    if 'wiki_ch' in name:
        train_data, test_data = get_wiki_ch(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'completion')
    if "belle" in name:
        import json
        test_data = [test_data.append({"input": ent["instruction"] + "\n" + ent["instances"][0]["input"], "output": ent["instances"][0]["output"]}) for ent in json.load(open("/home/kaixin/programs/LLM-Pruner/zh_seed_tasks.json", "r"))] 
            
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_data, test_loader