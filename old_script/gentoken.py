from tokenizers import Tokenizer
from transformers import LlamaTokenizerFast

pruned_model_path = "/home/kaixin/LLMPruner/LLM-Pruner/merged_llama2_pruned_tuned"

# ✅ Load sentencepiece directly and convert
sp_tokenizer = LlamaTokenizerFast(vocab_file=f"{pruned_model_path}/tokenizer.model")
sp_tokenizer.save_pretrained(pruned_model_path)
