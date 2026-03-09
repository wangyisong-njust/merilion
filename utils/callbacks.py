import torch
import numpy as np
import re
from transformers import TrainerCallback
import evaluate
import tqdm
import json
import os
from jiwer import wer as jiwer_wer

# Lightweight text normalization (matches vllm_inference preprocess_text_asr logic)
def _normalize_text_for_wer(text):
    """Normalize text for WER: lowercase, remove punctuation/brackets, strip speaker tags."""
    if not isinstance(text, str):
        text = str(text)
    # Strip speaker tags
    text = text.replace("<Speaker1>:", "").replace("<Speaker2>:", "")
    # Lowercase
    text = text.lower()
    # Remove content in brackets
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'\(.*?\)', ' ', text)
    text = re.sub(r'\{.*?\}', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    # Remove punctuation
    text = re.sub(r"[^\w\s']", ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class WEREvalCallback(TrainerCallback):
    """
    WER evaluation callback that runs inference directly (no audiobench dependency).
    Ensures consistent dtype handling and reference format.
    """
    def __init__(self, eval_dataset, processor, model_wrapper, log_file, eval_steps=100, num_samples=50):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.model_wrapper = model_wrapper
        self.log_file = log_file
        self.eval_steps = eval_steps
        self.num_samples = num_samples
        self.wer_metric = evaluate.load("wer")
        self.best_wer = float('inf')

    def _do_inference(self, model, audio_array, instruction):
        """Run inference directly, bypassing audiobench to avoid dtype/device bugs."""
        # Ensure audio is 1D float32 numpy
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.asarray(audio_array, dtype=np.float32)
        if audio_array.ndim == 2:
            audio_array = audio_array.mean(axis=-1) if audio_array.shape[-1] < audio_array.shape[0] else audio_array.mean(axis=0)
        audio_array = audio_array.astype(np.float32, copy=False)

        prompt = "Instruction: {instruction} \nFollow the text instruction based on the following audio: <SpeechHere>"
        conversation = [{"role": "user", "content": prompt.format(instruction=instruction)}]
        chat_prompt = self.processor.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(text=chat_prompt, audios=audio_array, return_tensors="pt")
        # Move all tensors to model device with correct dtype
        device = next(model.parameters()).device
        for k in inputs:
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].to(device)
                if inputs[k].dtype == torch.float32:
                    inputs[k] = inputs[k].to(torch.bfloat16)

        model_outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False, num_beams=1)
        input_len = inputs['input_ids'].shape[1]
        generated_ids = model_outputs[:, input_len:]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # Strip speaker tags to match reference format
        response = response.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()
        return response

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            print(f"\n--- Step {state.global_step}: Running Validation WER Evaluation ---")

            current_model = kwargs['model']
            current_model.eval()

            predictions = []
            references = []

            eval_subset = self.eval_dataset.shuffle(seed=42).select(
                range(min(self.num_samples, len(self.eval_dataset)))
            )

            with torch.no_grad():
                for sample in tqdm.tqdm(eval_subset, desc="Evaluating WER"):
                    # Extract audio
                    if 'context' in sample:
                        ctx = sample['context']
                        if 'audio' in ctx:
                            audio_array = ctx['audio']['array']
                        else:
                            audio_array = ctx['array']
                    else:
                        audio_array = sample['audio_array']

                    instruction = sample['instruction']['text'] if isinstance(sample['instruction'], dict) else sample['instruction']

                    # Extract reference — always use RAW transcription without <Speaker1>: prefix
                    # (generate() strips it, so reference must match)
                    if 'other_attributes' in sample:
                        oa = sample['other_attributes']
                        if oa.get('partition') == 'PART1':
                            ref = oa['Transcription']
                        elif oa.get('partition') == 'PART3':
                            ref = oa['transcription']
                        else:
                            ref = sample.get('answer', "Unknown")
                    else:
                        ref = sample.get('answer', "Unknown")

                    pred = self._do_inference(current_model, audio_array, instruction)
                    predictions.append(_normalize_text_for_wer(pred))
                    references.append(_normalize_text_for_wer(ref))

            # Filter out empty pairs
            valid = [(p, r) for p, r in zip(predictions, references) if r.strip()]
            if valid:
                predictions, references = zip(*valid)
                predictions = list(predictions)
                references = list(references)
            wer = self.wer_metric.compute(predictions=predictions, references=references)
            print(f"Step {state.global_step} Validation WER: {wer:.4f}")

            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "a") as f:
                f.write(json.dumps({"step": state.global_step, "wer": wer}) + "\n")

            output_dir = args.output_dir

            latest_dir = os.path.join(output_dir, "latest_model")
            print(f"Saving latest model to {latest_dir}...")
            current_model.save_pretrained(latest_dir)
            self.processor.save_pretrained(latest_dir)

            if wer < self.best_wer:
                self.best_wer = wer
                best_dir = os.path.join(output_dir, "best_model")
                print(f"New Best WER: {wer:.4f}! Saving best model to {best_dir}...")
                current_model.save_pretrained(best_dir)
                self.processor.save_pretrained(best_dir)

                with open(os.path.join(best_dir, "best_step.txt"), "w") as f:
                    f.write(f"Step: {state.global_step}\nWER: {wer:.4f}")

            current_model.train()

        return control

    def on_train_end(self, args, state, control, **kwargs):
        print(f"\n{'='*50}")
        print(f"Training complete. Best Validation WER: {self.best_wer:.4f}")
        best_dir = os.path.join(args.output_dir, "best_model")
        if os.path.exists(os.path.join(best_dir, "best_step.txt")):
            with open(os.path.join(best_dir, "best_step.txt")) as f:
                print(f.read())
        print(f"{'='*50}")
