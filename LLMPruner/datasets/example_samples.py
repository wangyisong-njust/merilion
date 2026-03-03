import random
import numpy as np
import torch

from datasets import load_dataset
from torch.utils.data.dataset import Dataset

def get_c4(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'allenai/c4', 'zh', split='train'
    )
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len )
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0)

def get_bookcorpus(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'bookcorpus', split='train'
    )
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0 )

def get_examples(dataset, tokenizer, n_samples, seq_len = 128):
    if dataset == 'c4':
        return get_c4(tokenizer, n_samples, seq_len)
    elif dataset == 'bookcorpus':
        return get_bookcorpus(tokenizer, n_samples, seq_len)
    elif dataset == 'wikitext':
        return get_wikitext2(tokenizer, n_samples, seq_len)
    elif dataset == 'cauldron':
        return get_cauldron(tokenizer, n_samples, seq_len)
    else:
        raise NotImplementedError


def get_wikitext2(tokenizer, n_samples, seq_len):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len )
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_cauldron(processor, n_samples, seq_len):
    traindata = load_dataset('HuggingFaceM4/the_cauldron', 'diagram_image_to_text', split='train')
    
    tokenized_samples = []
    for _ in range(n_samples):
        i = random.randint(0, len(traindata) - 1)
        image = traindata[i]['images'][0]


        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": traindata[i]['texts'][0]['user']}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": traindata[i]['texts'][0]['assistant']}
                ]
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

        prompt = prompt[:seq_len]

        # Prepare inputs
        tokenized_sample = processor(text=prompt, images=[image], return_tensors="pt")
        tokenized_samples.append(tokenized_sample)

    return tokenized_samples