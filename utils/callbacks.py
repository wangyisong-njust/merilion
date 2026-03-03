import torch
from transformers import TrainerCallback
import evaluate
import tqdm
import json
import os

class WEREvalCallback(TrainerCallback):
    def __init__(self, eval_dataset, processor, model_wrapper, log_file, eval_steps=100, num_samples=50):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.model_wrapper = model_wrapper # This is the 'Model' object from audiobench
        self.log_file = log_file
        self.eval_steps = eval_steps
        self.num_samples = num_samples
        self.wer_metric = evaluate.load("wer")
        self.best_wer = float('inf')

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            print(f"\n--- Step {state.global_step}: Running Validation WER Evaluation ---")
            
            # Use the current PEFT model for inference
            current_model = kwargs['model']
            current_model.eval()
            
            # Temporarily replace the model in the wrapper to use its generate method
            original_model = self.model_wrapper.model
            self.model_wrapper.model = current_model
            
            predictions = []
            references = []
            
            # Select a subset for quick evaluation
            eval_subset = self.eval_dataset.shuffle(seed=42).select(range(min(self.num_samples, len(self.eval_dataset))))
            
            # Prepare data
            input_data = []
            for sample in eval_subset:
                # Same logic as preprocess_keep_raw but for the wrapper
                # Correcting extraction for MERaLiON dataset
                if 'context' in sample:
                    audio_array = sample['context']['array']
                    sr = sample['context']['sampling_rate']
                else:
                    audio_array = sample['audio_array']
                    sr = sample['sampling_rate']
                
                instruction = sample['instruction']['text'] if isinstance(sample['instruction'], dict) else sample['instruction']
                ref = sample.get('answer', "Unknown")
                
                input_data.append({
                    "audio": {"array": audio_array, "sampling_rate": sr},
                    "instruction": instruction,
                    "task_type": "ASR",
                    "reference": ref
                })
                references.append(ref)

            # Run inference
            with torch.no_grad():
                for item in tqdm.tqdm(input_data, desc="Evaluating WER"):
                    pred = self.model_wrapper.generate(item)
                    predictions.append(pred)

            wer = self.wer_metric.compute(predictions=predictions, references=references)
            print(f"Step {state.global_step} Validation WER: {wer:.4f}")
            
            # Log results
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "a") as f:
                f.write(json.dumps({"step": state.global_step, "wer": wer}) + "\n")
            
            # Save logic to save disk space
            output_dir = args.output_dir
            
            # 1. Always save latest for resumption (overwrite previous latest)
            latest_dir = os.path.join(output_dir, "latest_model")
            print(f"Saving latest model to {latest_dir}...")
            current_model.save_pretrained(latest_dir)
            self.processor.save_pretrained(latest_dir)

            # 2. Only save best model if WER improves
            if wer < self.best_wer:
                self.best_wer = wer
                best_dir = os.path.join(output_dir, "best_model")
                print(f"New Best WER: {wer:.4f}! Saving best model to {best_dir}...")
                current_model.save_pretrained(best_dir)
                self.processor.save_pretrained(best_dir)
                
                # Also write a simple txt to track which step this was
                with open(os.path.join(best_dir, "best_step.txt"), "w") as f:
                    f.write(f"Step: {state.global_step}\nWER: {wer:.4f}")
            
            # Restore model and training state
            self.model_wrapper.model = original_model
            current_model.train()
            
        return control
