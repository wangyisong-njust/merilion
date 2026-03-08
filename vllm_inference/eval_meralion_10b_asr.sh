# Step 1:
# Server the judgement model using VLLM framework (my example is using int4 quantized version)
# This requires with 1 * 80GB GPU
# bash vllm_model_judge_llama_3_70b.sh

# Step 2:
# We perform model inference and obtain the evaluation results with the second GPU
GPU=0
BATCH_SIZE=1
OVERWRITE=True
# OVERWRITE=False
NUMBER_OF_SAMPLES=-1 # indicate all test samples if number_of_samples=-1
# NUMBER_OF_SAMPLES=50
DATASET=imda_part1_asr_test
# METRICS=llama3_70b_judge
METRICS=wer

MODEL_NAME=MERaLiON-2-10B-ASR
# MODEL_NAME=MERaLiON-2-10B-ASR-BNB-8bit-new
# MODEL_NAME=MERaLiON-2-10B-ASR-BNB-4bit-new
MODEL_NAME=MERaLiON-2-10B-ASR-vllm
# MODEL_NAME=MERaLiON-2-10B-ASR-vllm-W8A16-RTN-textonly
# MODEL_NAME=MERaLiON-2-10B-ASR-vllm-W8A16-G256-damp05-textonly
# MODEL_NAME=MERaLiON-2-10B-ASR-vllm-W8A16-RTN-textonly-fp8KV
# MODEL_NAME=MERaLiON-2-10B-ASR-vllm-W8A16-G256-damp05-textonly-fp8KV
# MODEL_NAME=MERaLiON-2-10B-ASR-vllm-W4A16-G256-damp05-textonly
# MODEL_NAME=MERaLiON-2-10B-ASR-vllm-W4A16-G256-damp05-textonly-fp8KV
# MODEL_NAME=MERaLiON-2-10B-ASR-vllm-W4A16-RTN-textonly
MODEL_NAME=MERaLiON-2-10B-ASR-vllm-W4A16-RTN-textonly-fp8KV

# Pruned
MODEL_NAME=MERaLiON-2-10B-ASR-vllm-0_5-5-40-merged
# MODEL_NAME=MERaLiON-2-10B-ASR-vllm-0_25-7-35-merged
MODEL_NAME=MERaLiON-2-10B-ASR-vllm-0_5-5-40
MODEL_NAME=MERaLiON-2-10B-ASR-vllm-0_25-7-35
MODEL_NAME=MERaLiON-2-10B-ASR-vllm-0_25-7-35-W4A16-RTN-textonly-fp8KV
MODEL_NAME=MERaLiON-2-10B-ASR-vllm-0_25-7-35-W8A16-RTN-textonly-fp8KV

MODEL_NAME=MERaLiON-2-10B-ASR-vllm-0_5-5-40-tuned-r32-full_gemma2-merged
MODEL_NAME=MERaLiON-2-10B-ASR-vllm-0_5-5-40-tuned-r32-full_gemma2-imda2-merged
MODEL_NAME=MERaLiON-2-10B-ASR-vllm-0_5-5-40-tuned-r32-full_gemma2-imda2-5e-5-merged
MODEL_NAME=MERaLiON-2-10B-ASR-vllm-0_5-5-40-tuned-r32-full_gemma2-imda2-5e-5-merged-W8A16-RTN-textonly-fp8KV
# MODEL_NAME=MERaLiON-2-10B-ASR-vllm-0_5-5-40-tuned-r32-full_gemma2-imda2-4e-4-merged

MODEL_NAME=MERaLiON-2-10B-ASR-vllm-0_25-7-35-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged
MODEL_NAME=MERaLiON-2-10B-ASR-vllm-0_25-7-35-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-W8A16-RTN-textonly-fp8KV
MODEL_NAME=MERaLiON-2-10B-ASR-vllm-0_25-7-35-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-W4A16-RTN-textonly-fp8KV
MODEL_NAME=MERaLiON-2-10B-ASR-vllm-0_25-7-35-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-new
MODEL_NAME=MERaLiON-2-10B-ASR-vllm-0_25-7-35-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-new-W8A16-RTN-textonly-fp8KV
MODEL_NAME=MERaLiON-2-10B-ASR-vllm-0_25-7-35-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-new-W4A16-RTN-textonly-fp8KV

# MODEL_NAME=MERaLiON-2-10B-ASR-vllm-0_5-5-40-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged
# MODEL_NAME=MERaLiON-2-10B-ASR-vllm-0_5-5-40-tuned-r32-full_gemma2-mix-5e-5-grad_accu_2-merged-W8A16-RTN-textonly-fp8KV

MODEL_NAME=MERaLiON-2-3B-vllm
# MODEL_NAME=MERaLiON-2-3B-vllm-W8A16-RTN-textonly
# MODEL_NAME=MERaLiON-2-3B-vllm-W4A16-RTN-textonly
# MODEL_NAME=MERaLiON-2-3B-vllm-0_25-3-23-tuned-r32-full_gemma2-mix-1e-5-grad_accu_2-dropout01-merged
# MODEL_NAME=MERaLiON-2-3B-vllm-0_25-3-23-tuned-r32-full_gemma2-mix-1e-5-grad_accu_2-dropout01-merged-W8A16-RTN-textonly
# MODEL_NAME=MERaLiON-2-3B-vllm-0_25-3-23-tuned-r32-full_gemma2-mix-1e-5-grad_accu_2-dropout01-merged-W4A16-RTN-textonly
# MODEL_NAME=MERaLiON-2-3B-vllm-0_5-3-23-tuned-r16-full_gemma2-mix-1e-5-grad_accu_2-dropout01-merged
# MODEL_NAME=MERaLiON-2-3B-vllm-0_5-3-23-tuned-r16-full_gemma2-mix-1e-5-grad_accu_2-dropout01-merged-W8A16-RTN-textonly
# MODEL_NAME=MERaLiON-2-3B-vllm-0_5-3-23-tuned-r16-full_gemma2-mix-1e-5-grad_accu_2-dropout01-merged-W4A16-RTN-textonly

MODEL_NAME=MERaLiON-2-3B-0_25-4-23-both
MODEL_NAME=MERaLiON-2-3B-0_25-4-23-both-tuned-r16-a16-5e-6-bs8-imda1m3c-merged
MODEL_NAME=MERaLiON-2-3B-0_25-4-23-both-tuned-r16-a16-5e-6-bs8-imda1m3c-merged-W8A16-RTN-textonly
MODEL_NAME=MERaLiON-2-3B-0_25-4-23-both-tuned-r16-a16-5e-6-bs8-imda1m3c-merged-W4A16-RTN-textonly



MODEL_NAME=MERaLiON-2-quant



bash eval.sh $DATASET $MODEL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

# bash eval_meralion_10b_asr.sh

# /home/jinchao/miniconda3/envs/audiobench_quant/lib/python3.10/site-packages/llmcompressor/modifiers/quantization/gptq/gptq_quantize.py
# line 48 add:
#         if len(inp.shape) == 4: # llmcompressor does not support 4d
#             N, C, H_dim, W_dim = inp.shape
#             inp = inp.reshape(N, C, -1) 
#             inp = inp.permute(0, 2, 1)
#             inp = inp.reshape(-1, C)


# CUDA_VISIBLE_DEVICES=3 python whisper_quant_example.py


# export no_proxy=localhost,127.0.0.1,10.104.0.0/21
# export https_proxy=http://10.104.4.124:10104
# export http_proxy=http://10.104.4.124:10104



# DATASET=$1
# MODEL=/home/kaixin/programs/LLM_base_model/
# GPU=3
# BATCH_SIZE=2
# OVERWRITE=$5
# METRICS=$6
# NUMBER_OF_SAMPLES=$7


# export CUDA_VISIBLE_DEVICES=$GPU




# python src/main_evaluate.py \
#     --dataset_name $DATASET \
#     --model_name $MODEL \
#     --batch_size $BATCH_SIZE \
#     --overwrite $OVERWRITE \
#     --metrics $METRICS \
#     --number_of_samples $NUMBER_OF_SAMPLES
