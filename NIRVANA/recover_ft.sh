model=/PATH/TO/MODEL # Put your path here

OUTDIR=../model/NIRVANA-lora-alpaca # Finetuned model output directory
export NCCL_P2P_DISABLE=1        # good practice on multi‑GPU single node
export TOKENIZERS_PARALLELISM=false

export CUDA_VISIBLE_DEVICES=5,6
python post_training.py \
  --base_model $model \
  --output_dir $OUTDIR            \
  --batch_size 4                  \
  --micro_batch_size 4            \
  --learning_rate 3e-4            \
  --num_epochs 2                  \
  --cutoff_len 256                \
  --save_steps 100000000          \
  --lora_r 8 --lora_alpha 16      \
  $RESUME_ARG

