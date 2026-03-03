import torch
import numpy as np
from tqdm import tqdm

from LLMPruner.datasets.ppl_dataset import get_loaders

def PPLMetric(model, tokenizer, datasets, seq_len=128, batch_size = 4, device="cuda"):
    metric = {}
    for dataset in datasets:
        _, test_loader = get_loaders(dataset, tokenizer, seq_len=seq_len, batch_size = batch_size)
        ppl = llama_eval(model, test_loader, device)
        metric[dataset] = ppl
        print(metric)
    return metric

import pandas as pd

@torch.no_grad()
def llama_eval(model, test_lodaer, device, save_test_cases="", tokenizer=None):
    nlls = []
    n_samples = 0
    
    if save_test_cases != "":
        df = pd.DataFrame(columns=["loss", "input", "output", "label"])
    for batch in tqdm(test_lodaer):
        batch = batch.to(device)
        output = model(batch)
        lm_logits = output.logits
    
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss)
        
        if save_test_cases != "":
            loss = loss.view(shift_labels.size())
            for i in range(loss.size(0)):
                n_samples += 1
                cur_loss = loss[i].mean().item()
                input_text = tokenizer.decode(batch[i], skip_special_tokens=True).replace("\n", " ").replace(",", "，")
                output_text = tokenizer.decode(shift_labels[i], skip_special_tokens=True).replace("\n", " ").replace(",", "，")
                df.loc[len(df)] = [cur_loss, input_text, output_text, output_text]
    
    if save_test_cases != "":
        df.to_csv(save_test_cases, index=False)
        print(f"save {n_samples} samples to {save_test_cases}")
        
    
    #print(torch.cat(nlls, dim=-1).mean())
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()


def PPLMetric(model, tokenizer, datasets, seq_len=128, batch_size = 4, device="cuda", save_test_cases=""):
    metric = {}
    for dataset in datasets:
        _, test_loader = get_loaders(dataset, tokenizer, seq_len=seq_len, batch_size = batch_size)
        ppl = llama_eval(model, test_loader, device, save_test_cases=save_test_cases, tokenizer=tokenizer)
        metric[dataset] = ppl
    return metric

@torch.no_grad()
def llama_eval_openai(model, test_lodaer, device):
    all_logprobs = []

    for batch in tqdm(test_lodaer):
        batch = batch.to(device)

        generation_output = model.generate(
            input_ids=batch,
            do_sample=True,
            max_new_tokens=50,
            top_p=0.95,
            temperature=1,
            return_dict_in_generate=True,
            output_logits=True
        )

        sequences = generation_output.sequences
        logits = generation_output.logits

        logits = torch.cat(logits, dim=0)
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

        new_top_logprobs = [float(logprobs[0, sequences[0, i + batch.shape[-1]]].cpu().item()) for i in range(len(logprobs))]

        all_logprobs.extend(new_top_logprobs)

    ppl = np.exp(np.array(all_logprobs).mean())
    return ppl.item()

def PPLMetricOpenAI(model, tokenizer, datasets, seq_len=128, batch_size = 1, device="cuda"):
    metric = {}
    for dataset in datasets:
        _, test_loader = get_loaders(dataset, tokenizer, seq_len=seq_len, batch_size = batch_size)
        ppl = llama_eval_openai(model, test_loader, device)
        metric[dataset] = ppl
        print(metric)
    return metric