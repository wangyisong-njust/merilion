
## Introduction

Below are the instructions for using the LLM-Pruner method to perform structured pruning and LoRA finetuning on MERaLiON-2 models.
> **[LLM-Pruner: On the Structural Pruning of Large Language Models](https://arxiv.org/abs/2305.11627)** [[arXiv]](https://arxiv.org/abs/2305.11627) 

### Dependencies
- On the A100 10.9.19.25 server, you may use the audiobench_quant conda environment with the following command:
```bash
conda activate audiobench_quant
```

### 1. Structured Pruning 
#### 1.1. Important Directories Related to Pruning
The `./audiobench` folder holds the code for model and dataset loading.
The `./audiobench/model_src/meralion_2.py` file stores the code for model loading in different situations. Please update the checkpoint path by setting the repo_id variable. Model loader with the `finetune with transformers.Trainner` comment can be used for pruning and finetuning.
Other model and dataset loading functions are stored in `audiobench/model.py` and `audiobench/dataset.py`.

`./meralion_bl` stores modified huggingface model definition code for meralion2 model after LLM-Pruner structured pruning. Please search for `midblock_start` in `./meralion2_bl/modeling_gemma2.py` to see how the layer dimensions are changed. Additionally, `./meralion2_bl/blockwise_kv_cache.py` holds the incomplete version of huggingface KV cache that aims to adapt to the pruned attn layers. It can be ignored because currently we only conduct inference with the vLLM framework instead of huggingface.

`/home/jinchao/runtao/LLM_base_model` stores original models. 
`./meralion_checkpoints` stores pruned models.
`./meralion_tune_log` stores lora adapters after finetuning.

`./NIRVANA` holds the code release by the NIRVANA structured pruning method (under review of ICLR 2026). It is not the focus now, but may be adapted to the meralion2 model.


#### 1.2. Pruning Script
The script for pruning and finetuning is stored in `./scripts/meralion.sh`.
For example, to prune 25% of parameters from the selected blocks in the Gemma2 text decoder, please first set repo_id in `./audiobench/model_src/meralion_2.py` to be the original model's path, such as `/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B`. Then run the following script (prunes the blocks with idx 4, 5, ..., 22):
```bash
cd ~/runtao/LLM-Pruner
prune_ckpt_path='MERaLiON-2-3B-0_25-4-23-both'
echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=2 python meralion.py \
      --base_model MERaLiON-2-3B \
      --pruning_ratio 0.25 \
      --pruner_type taylor --taylor param_mix \
      --block_wise \
      --block_mlp_layer_start 4 --block_mlp_layer_end 23 \
      --block_attention_layer_start 4 --block_attention_layer_end 23 \
      --num_examples 20 \
      --max_seq_len 256 \
      --save_model \
      --save_ckpt_log_name $prune_ckpt_path \
      --save_model_path meralion_checkpoints/$prune_ckpt_path 
echo "[FINISH] - Finish Pruning Model"
```

```bash
Argument description:
prune_ckpt_path: Folder name for storing the pruned model.
--base_model: 'MERaLiON-2-3B' or 'MERaLiON-2-10B-ASR'
--pruning_ratio: The amount of parameters to be pruned. 
--block_mlp_layer_start & --block_attention_layer_start: The index of the block where pruning starts to be applied. The 2 indices should be the same.
--block_mlp_layer_end & --block_attention_layer_end: Pruning will not be applied from this block to the end of the language model. The 2 indices should be the same.
```

Note that currently the code only support prunning a full attention head, so the pruning ratio must be set accordingly. For example, for MERaLiON-2-3B, num_attention_heads=8 and num_key_value_heads=4. Thus, its pruning ratio can only be set to 0.25, 0.5 or 0.75. 

In the pruned model's new `config.json` file, these 3 values should be modified by the pruning code:
```bash
"text_config": {
    "midblock_ratio": 0.75, # 1 - pruning ratio
    "midblock_start": 4, # same start block index in the pruning setting
    "midblock_end": 23, # same end block index in the pruning setting
    }
```

#### 1.3. LoRA Finetuning
To finetune the pruned model, first set repo_id in `./audiobench/model_src/meralion_2.py` to be the pruned model's path, such as `./meralion_checkpoints/MERaLiON-2-3B-0_25-4-23-both`. Then run the following script:
```bash
prune_ckpt_path='MERaLiON-2-3B-0_25-4-23-both'
tune_ckpt_path='MERaLiON-2-3B-0_25-4-23-both-tuned-r16-a16-5e-6-bs8-imda1m3c'
echo "[START] - Start Tuning"
CUDA_VISIBLE_DEVICES=5 WANDB_MODE=offline python post_training_meralion.py --base_model MERaLiON-2-3B \
 --output_dir meralion_tune_log/$tune_ckpt_path \
 --lora_r 16 --lora_alpha 16 --learning_rate 1e-5 --num_epochs 3 --batch_size 8 --micro_batch_size 4 --lora_dropout 0.05 
echo "[FINISH] - Finish Prune and Post-Training."
```

```bash
Argument description:
tune_ckpt_path: Folder name for storing the LoRA trained parameters.
--lora_r: LoRA rank. A larger rank has more trainable parameters and will take more time to train.
--num_epochs: Number of epochs for training. 
--micro_batch_size: Actual batch size for a single GPU.
--batch_size: Gradient will be accumulated over multiple micro batches until reaching this batch size. Currently a smaller batch size is better.
--resume_from_checkpoint: (Optional) Can resume training from a checkpoint in the ./tune_log/$tune_ckpt_path directory.
```

#### 1.4. Merge the Parameters and Upload
To merge the pruned checkpoint and the LoRA parameters after finetuning, please run the following script:
```bash
CUDA_VISIBLE_DEVICES=0 python merge_meralion.py \
    --ckpt ./meralion_checkpoints/MERaLiON-2-3B-0_25-4-23-both \
    --lora_ckpt meralion_tune_log/MERaLiON-2-3B-0_25-4-23-both-tuned-r16-a16-5e-5-bs8-imda1m3c/checkpoint-2600 \
    --save_path MERaLiON-2-3B-0_25-4-23-both-tuned-r16-a16-5e-6-bs8-imda1m3c-merged

```

```bash
Argument description:
--ckpt: The stored pruned model's checkpoint.
--lora_ckpt: The stored finetuned LoRA parameters.
--save_path: The directory to save the merged model.
```

Then the saved merged model can be uploaded to a newly created HuggingFace model repository by:
```bash
huggingface-cli upload <huggingface repo name> <merged model's local directory>
```


### 2. vLLM Inference
Please switch to `~/runtao/meralion_test` directory. 
Currently we use vLLM 0.8.5 and llmcompressor 0.6.0.

#### 2.1. Important Directories Related to vLLM for Pruned Models

`./eval_meralion_10b_asr.sh` is the main script for quantization and inference. If you want to perform quantization, please set `MODEL_NAME=MERaLiON-2-quant`. If you want to perform inference evaluation, please set `MODEL_NAME=MERaLiON-2-3B-EXPERIMENT-SETTINGS`, which will be the folder name for saving the inference results.

`./src/model_src/meralion_2_quant.py` holds the code for performing RTN (Round-to-Nearest) quantization. The GPTQ recipe with calibration data can also be run, but it tends to produce worse model performance than RTN. Later need to find if the code or calibration data selection for GPTQ can be improved, but it is not the focus now.

`./src/model_src/meralion_2_vllm.py` holds the code for loading the model with vLLM and perform inference.

`./src/main_evaluate.py` holds the code for the evaluation pipeline.

In `/home/jinchao/miniconda3/envs/audiobench_quant/lib/python3.10/site-packages/vllm/model_executor/models/gemma2.py`, please search for `midblock_start` in `./meralion2_bl/modeling_gemma2.py` to see how the layer dimensions are changed for vLLM. vLLM may already support KV cache with different block attn heads. Need to verigy whether KV cache is successfully triggered during vLLM inference.

In `/home/jinchao/miniconda3/envs/audiobench_quant/lib/python3.10/site-packages/vllm/attention/layer.py`, please search for `# new` to see how the KV cache is sliced to meet the pruned attn head number. Later can check if there is any other more efficient way to set the dimension (set it when KV cache is initialized).

`/home/jinchao/miniconda3/envs/audiobench_quant/lib/python3.10/site-packages/vllm_plugin_meralion2/vllm085.py` currently supports loading the pruned MLP weights, but if Attn weights are also pruned, then there are still some errors.

`./meralion2_bl_llmcompressor` stores modified huggingface model definition code for meralion2 model after LLM-Pruner structured pruning. They are used for quantization with llmcompressor and may contain small changes compared with `./meralion2_bl` in Section 1.1.

`/home/jinchao/miniconda3/envs/audiobench_quant/lib/python3.10/site-packages/llmcompressor/modifiers/quantization/gptq/gptq_quantize.py` is modified in line 48-52 to support GPTQ on meralion2 model. Need to verify if this is the correct modification.

`./sort_wer_json.py` is used to extract the transcriptions with the highest and lowest WER results.
`./extract_audio_wav.py` is used to save the audio files for the test samples above.  

`./saved_meralion` stores quantized models.

#### 2.2. vLLM Inference
Please set the repo_id in `./src/model_src/meralion_2_vllm.py` to the model path, such as `/home/jinchao/runtao/LLM-Pruner/meralion_checkpoints/MERaLiON-2-3B-0_25-4-23-both`. Then set `MODEL_NAME=MERaLiON-2-3B-0_25-4-23-both` inside `./eval_meralion_10b_asr.sh`. Then please run the script to perform inference evaluation:
```bash
bash eval_meralion_10b_asr.sh
```

```bash
Argument description:
DATASET: The test dataset name. Please refer to the dataset list in AudioBench.
NUMBER_OF_SAMPLES: Number of test samples. If it is set to -1, then all the samples will be used.
METRICS: WER refers to Word Error Rate. Please refer to AudioBench for setting the metric for each dataset.
```


#### 2.3. llmcompressor Quantization
Currently RTN and GPTQ for W8A16 and W4A16 quantization can both be run to quantize the text decoder of meralion2 models, but RTN tends to yield better results. 
In `./src/model_src/meralion_2_quant.py`, please set the repo_id to the model path, such as `/home/jinchao/runtao/LLM-Pruner/meralion_checkpoints/MERaLiON-2-3B-0_25-4-23-both`. Set `scheme="W8A16"` or `scheme="W4A16"` for `QuantizationModifier`. Modify the `SAVE_DIR` below to the folder name for saving the quantized model.
In `./eval_meralion_10b_asr.sh`, set `MODEL_NAME=MERaLiON-2-quant`. 
Finally please run the script to perform inference evaluation:
```bash
bash eval_meralion_10b_asr.sh
```
