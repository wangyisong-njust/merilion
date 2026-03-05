type=NIRVANA
# type=magnitude

data=bookcorpus

num_examples=32
batch_size=32
seq_len=128
sparsity=0.5


args=(
--base_model meta-llama/Llama-3.1-8B
--device cuda


# Prune setting
--prune_type $type
--prune

# Calibration setting
--data $data
--num_examples $num_examples
--seq_len $seq_len
--batch_size $batch_size
--max_seq_len 512
--sparsity $sparsity
--seed 12 # Default seed

# Model setting
--gamma 3.36 # Default gamma value
--test_after_prune
--save_model ../model
)


python main.py "${args[@]}"