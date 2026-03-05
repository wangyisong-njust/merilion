from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch

import random
import numpy as np





def get_bookcorpus(tokenizer, n_samples, seq_len, select=False, idx=-1, verbose=False, seed=42):
    traindata = load_dataset(
        'bookcorpus', split='train', trust_remote_code=True
    )
    
    random.seed(seed)
    
    if idx>=0: 
      traindata = traindata.filter(lambda x: len(x["text"]) > 1024)
    else:
      traindata = traindata.filter(lambda x: len(x["text"]) > seq_len)
    
    
    tokenized_samples, history = [], []
    
    if idx>=0:       
        # print('idx', idx)
        tokenized_sample = tokenizer(traindata[idx]['text'], return_tensors='pt')
        if tokenized_sample.input_ids.shape[1] - seq_len < 0:
          print('too short')
          print(traindata[idx]['text'])
          print(tokenized_sample.input_ids)
          return None
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])        
        if verbose:  
          print(tokenizer.decode(tokenized_sample.input_ids[:, i:i+seq_len][0]))
    
    else:
      
      for _ in range(n_samples):
          while True:
              i = random.randint(0, len(traindata) - 1)
              tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
              if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                  history.append(i)
                  break
          if verbose:
            print(tokenizer.decode(tokenized_sample.input_ids[:, i:i+seq_len][0]))
          i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)        
          tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
          # print(tokenizer.decode(tokenized_sample.input_ids[:, i:i+seq_len][0]))
    return torch.cat(tokenized_samples, dim=0 )
    
    
  
def get_examples(dataset, tokenizer, n_samples, seq_len = 128, select=False, idx=-1, verbose=False, seed=42):
    if dataset == 'bookcorpus':
        return get_bookcorpus(tokenizer, n_samples, seq_len, select, idx, verbose, seed)
    else:
        raise NotImplementedError

