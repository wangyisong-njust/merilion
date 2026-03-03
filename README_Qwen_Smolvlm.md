
## Introduction

Below are the instructions for using the LLM-Pruner method to perform structured pruning and LoRA finetuning on Qwen2.5-3B and SmolVLM-500M-Instruct models.
> **[LLM-Pruner: On the Structural Pruning of Large Language Models](https://arxiv.org/abs/2305.11627)** [[arXiv]](https://arxiv.org/abs/2305.11627) 

### Dependencies
- You may create a conda environment with the following command:
```bash
conda env create -f env_llm_pruner_qwen_no_builds.yml
```

### 1. SmolVLM-500M-Instruct
#### 1.1. Structured Pruning 
For example, to prune 15% of parameters in the language model part. Please run the following script:
```bash
prune_ckpt_path='SmolVLM-500M-Instruct-bl-0_2-4-28'
echo "[START] - Start Pruning Model"
python smolvlm.py \
      --base_model HuggingFaceTB/SmolVLM-500M-Instruct \
      --pruning_ratio 0.2 \
      --save_ckpt_log_name $prune_ckpt_path \
      --pruner_type taylor --taylor param_mix \
      --save_model \
      --block_wise \
      --block_mlp_layer_start 4 --block_mlp_layer_end 28 \
      --block_attention_layer_start 4 --block_attention_layer_end 28 \
      --max_seq_len 2048 \
      --num_examples 20 \
      --save_model_path vlm_checkpoints/$prune_ckpt_path 
echo "[FINISH] - Finish Pruning Model"
```

```bash
Argument description:
prune_ckpt_path: Folder name for storing the pruned model.
--base_model: HuggingFace model name or local model directory
--pruning_ratio: The amount of parameters to be pruned. For SmolVLM-500M-Instruct, it could only be set to 0.2, 0.4, 0.6 or 0.8 to match its attention head number.
--block_mlp_layer_start & --block_attention_layer_start: The index of the block where pruning starts to be applied. The 2 indices should be the same.
--block_mlp_layer_end & --block_attention_layer_end: Pruning will not be applied from this block to the end of the language model. The 2 indices should be the same.
```

Copy the 4 files in the `./SmolVLM-500M-Instruct-bl_files` folder into the directory where the pruned model is stored (`./vlm_checkpoints/$prune_ckpt_path`). 
In the new `config.json` file, please change the 3 values according to the pruning setting:
```bash
"text_config": {
    "midblock_ratio": 0.8, # 1 - pruning ratio
    "midblock_start": 4, # same start block index in the pruning setting
    "midblock_end": 28, # same end block index in the pruning setting
    }
```

#### 1.2. LoRA Finetuning
First, download the COCO 2017 dataset into a local directory. 
Then download the [llava-instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) dataset into a local directory. 

To finetune the pruned model, please run the following script:
```bash
prune_ckpt_path='SmolVLM-500M-Instruct-bl-0_2-4-28'
tune_ckpt_path='SmolVLM-500M-Instruct-bl-0_2-4-28-tuned_qlora'
echo "[START] - Start Tuning"
python post_training_smolvlm.py --base_model ./vlm_checkpoints/$prune_ckpt_path \
 --output_dir tune_log/$tune_ckpt_path \
 --lora_r 16 --num_epochs 4 --learning_rate 4e-4 --batch_size 32 \
 --coco_data_path /path/to/coco \
 --llava_data_path /path/to/llava-instruct-150K \
 --cache_dataset
echo "[FINISH] - Finish Prune and Post-Training."
```

```bash
Argument description:
tune_ckpt_path: Folder name for storing the LoRA trained parameters.
--lora_r: LoRA rank. Can be set to 16 or 32. A larger rank has more trainable parameters and will take more time to train.
--num_epochs: Number of epochs for training. 
--coco_data_path: Local path to the COCO 2017 dataset.
--llava_data_path: Local path to the llava-instruct-150K dataset.
--resume_from_checkpoint: (Optional) Can resume training from a checkpoint in the ./tune_log/$tune_ckpt_path directory.
```

#### 1.3. Merge the Parameters and Upload
To merge the pruned checkpoint and the LoRA parameters after finetuning, please run the following script:
```bash
python merge_smolvlm_lora.py \
    --ckpt prune_log/SmolVLM-500M-Instruct-bl-0_2-4-28/pytorch_model.bin \
    --lora_ckpt tune_log/SmolVLM-500M-Instruct-bl-0_2-4-28-tuned_qlora \
    --save_path SmolVLM-500M-Instruct-bl-0_2-4-28-merged
```

```bash
Argument description:
--ckpt: The stored pruned model's pytorch_model.bin checkpoint.
--lora_ckpt: The stored finetuned LoRA parameters.
--save_path: The directory to save the merged model.
```

Then the saved merged model can be uploaded to a newly created HuggingFace model repository by:
```bash
huggingface-cli upload <huggingface repo name> <merged model's local directory>
```


### 2. Qwen2.5-3B
#### 2.1. Structured Pruning 
Similar to SmolVLM-500M-Instruct, pruning can be performed for Qwen2.5-3B in the following example script:
```bash
prune_ckpt_path='Qwen2.5-3B-Instruct-bl-0.15-c4book'
echo "[START] - Start Pruning Model"
python qwen.py \
      --base_model Qwen/Qwen2.5-3B-Instruct \
      --pruning_ratio 0.5 \
      --save_ckpt_log_name $prune_ckpt_path \
      --pruner_type taylor --taylor vectorize \
      --save_model \
      --block_wise \
      --block_mlp_layer_start 9 --block_mlp_layer_end 27 \
      --block_attention_layer_start 9 --block_attention_layer_end 27 \
      --max_seq_len 2048 \
      --num_examples 20
echo "[FINISH] - Finish Pruning Model"
```


#### 2.2. LoRA Finetuning
To finetune the pruned model, please run the following script:
```bash
echo "[START] - Start Tuning"
prune_ckpt_path='Qwen2.5-3B-Instruct-bl-0.15-c4book'
tune_ckpt_path='Qwen2.5-3B-Instruct-bl-0.15-c4book-finetuned_5datasets'
python post_training_c4zh.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin \
 --data_path silk-road/alpaca-data-gpt4-chinese --output_dir tune_log/$tune_ckpt_path \
 --lora_r 16 --num_epochs 5 --learning_rate 2e-4 --batch_size 32 --group_by_length --cache_dataset 
echo "[FINISH] - Finish Prune and Post-Training."
```

```bash
Argument description:
--data_path: The path to the dataset for finetuning.
```


#### 2.3. Merge the Parameters and Upload
To merge the pruned checkpoint and the LoRA parameters after finetuning, please run the following script:
```bash
python merge_smolvlm_lora.py \
    --ckpt prune_log/Qwen2.5-3B-Instruct-bl-0.15-c4book/pytorch_model.bin \
    --lora_ckpt tune_log/Qwen2.5-3B-Instruct-bl-0.15-c4book-finetuned_5datasets \
    --save_path Qwen2.5-3B-Instruct-bl-0.15-c4book-finetuned_5datasets-merged
```