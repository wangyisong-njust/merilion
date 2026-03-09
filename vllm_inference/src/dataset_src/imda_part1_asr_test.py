import random
import logging

from jiwer import compute_measures, wer

from dataset_src.text_normalizer.preprocess_text import preprocess_text_asr
from dataset_src.prompts.prompts import asr_instructions


class imda_part1_asr_test_dataset(object):

    def __init__(self, raw_data, number_of_samples):

        if number_of_samples != -1:
            raw_data = raw_data.shuffle(seed=42)
            raw_data = raw_data.select(range(number_of_samples))
        
        self.raw_data = raw_data
        self.prompt = asr_instructions
        logging.info('Number of samples: {}'.format(len(self.raw_data)))


    def prepare_model_input(self):

        input_data = []
        for sample in self.raw_data:
            # Local dataset has nested structure: context={text, audio={array, sampling_rate}}
            # Flatten to match expected format: audio={array, sampling_rate}
            context = sample['context']
            if isinstance(context, dict) and 'audio' in context:
                audio = context['audio']  # {path, array, sampling_rate}
            else:
                audio = context

            answer = sample['answer']
            if isinstance(answer, dict) and 'text' in answer:
                reference = answer['text']  # strip nested structure
            else:
                reference = answer

            instruction = random.choice(self.prompt)
            input_data.append({
                                "audio"      : audio,
                                "instruction": instruction,
                                "reference"  : reference,
                                "task_type"  : "ASR"
                                })

        logging.info('\n=  =  =  Dataset Sample  =  =  =')
        logging.info(random.sample(input_data, 1)[0])
        logging.info('=  =  =  =  =  =  =  =  =  =  =  =\n')

        return input_data


    def format_model_predictions(self, input_data, model_predictions):

        data_with_model_predictions = []
        for sample in input_data:
            new_sample = sample.copy()
            del new_sample["audio"]
            new_sample['model_prediction'] = model_predictions.pop(0)
            data_with_model_predictions.append(new_sample)
        return data_with_model_predictions


    def compute_score(self, data_with_model_predictions, metrics=None):

        if metrics != 'wer':
            raise ValueError(f"Unsupported metric: {metrics}. Supported metrics: 'wer' for ASR")
        
        predictions = []
        references  = []
        for item in data_with_model_predictions:
            model_prediction = preprocess_text_asr(item["model_prediction"])
            ref_text = item["reference"]
            # Strip <Speaker1>: prefix from local dataset references
            if isinstance(ref_text, str):
                ref_text = ref_text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()
            answer = preprocess_text_asr(ref_text)

            if len(model_prediction) == 0: model_prediction = "empty"
            if len(answer) == 0: answer = "empty"

            predictions.append(model_prediction)
            references.append(answer)

        sample_wer = []
        incorrect  = 0
        total      = 0
        for prediction, reference in zip(predictions, references):
            measures   = compute_measures(reference, prediction)
            incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
            total     += measures["substitutions"] + measures["deletions"] + measures["hits"]

            wer_score = wer(reference, prediction)

            sample_wer_score = {
                "reference" : reference,
                "prediction": prediction,
                "wer"       : wer_score,
            }

            sample_wer.append(sample_wer_score)

        total_wer = incorrect / total

        return {"wer": total_wer, "sample_wer": sample_wer}