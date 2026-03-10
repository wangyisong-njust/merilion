
import os
import torch
# torch._dynamo.disable()
# os.environ["TORCHDYNAMO_DISABLE"] = "1"

import fire
import json
import logging
from tqdm import tqdm



from dataset import Dataset
from model import Model

import time



# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%m/%d/%Y %H:%M:%S",
    level   = logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

file_save_folder = 'log_for_all_models'
# file_save_folder = 'log'


def do_model_prediction(input_data, model, batch_size):

    if batch_size not in [1, -1]:
        raise NotImplementedError("Batch size {} not implemented yet".format(batch_size))
    
    if batch_size == -1:
        model_predictions = model.generate(input_data)
    
    else:
        model_predictions = []
        for inputs in tqdm(input_data, leave=False):
            outputs = model.generate(inputs)
            if isinstance(outputs, list):
                model_predictions.extend(outputs)
            else:
                model_predictions.append(outputs)
                
    return model_predictions

def main(
        dataset_name      : str  = None,
        model_name        : str  = None,
        batch_size        : int  = 1,     # it is now a dummy parameter
        overwrite         : bool = False,
        metrics           : str  = None,
        number_of_samples : int  = -1,
        ):

    logger.info("= = "*20)
    logger.info("Dataset name: {}".format(dataset_name))
    logger.info("Model name: {}".format(model_name))
    logger.info("Batch size: {}".format(batch_size))
    logger.info("Overwrite: {}".format(overwrite))
    logger.info("Metrics: {}".format(metrics))
    logger.info("Number of samples: {}".format(number_of_samples))
    logger.info("= = "*20)

    # If the final score log exists, skip the evaluation
    if not overwrite and os.path.exists(f'{file_save_folder}/{model_name}/{dataset_name}_{metrics}_score.json'.format(model_name, dataset_name, metrics)):
        logger.info("Evaluation has been done before. Skip the evaluation.")
        with open(f'{file_save_folder}/{model_name}/{dataset_name}_{metrics}_score.json', 'r') as f:
            results = json.load(f)
        logger.info('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =')
        logger.info('Dataset name: {}'.format(dataset_name.upper()))
        logger.info('Model name: {}'.format(model_name.upper()))
        logger.info(json.dumps({metrics: results[metrics]}, indent=4, ensure_ascii=False))
        logger.info('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =')

        logger.info("\n\n\n\n\n")
        # return

    if model_name == 'WavLLM_fairseq':
        batch_size = -1
        logger.info("Batch size is set to -1 for WavLLM_fairseq model.")


    dataset = Dataset(dataset_name, number_of_samples)

    if overwrite or not os.path.exists(f'{file_save_folder}/{model_name}/{dataset_name}.json'):
        logger.info("Overwrite is enabled or the results are not found. Try to infer with the model: {}.".format(model_name))
    
        # Load model
        model = Model(model_name)

        # Specific current dataset name for evaluation
        model.dataset_name = dataset.dataset_name
        
        torch.cuda.synchronize()
        start = time.time()

        # Infer with model
        model_predictions           = do_model_prediction(dataset.input_data, model, batch_size=batch_size)
        
        torch.cuda.synchronize()
        end = time.time()
        inference_time = end - start
        num_samples = len(dataset.input_data)
        print("Inference took:", inference_time, "s")
        print(f"Throughput: {num_samples / inference_time:.2f} samples/sec")
        print(f"Avg latency: {inference_time / num_samples:.3f} sec/sample")

        # RTF (Real-Time Factor) for ASR: processing_time / total_audio_duration
        total_audio_duration = 0
        for sample in dataset.input_data:
            audio = sample.get("audio", {})
            if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
                total_audio_duration += len(audio["array"]) / audio["sampling_rate"]
        if total_audio_duration > 0:
            rtf = inference_time / total_audio_duration
            print(f"Total audio duration: {total_audio_duration:.1f} s")
            print(f"RTF (Real-Time Factor): {rtf:.4f}  (<1 = faster than real-time)")

        # Model size on disk
        model_path = getattr(model, 'model_name', model_name)
        possible_paths = [
            model_path,
            os.path.join("/home/jinchao/runtao/LLM-Pruner/meralion_checkpoints", model_name),
            os.path.join("/home/jinchao/runtao/LLM_base_model", model_name),
        ]
        for p in possible_paths:
            if os.path.exists(p) and os.path.isdir(p):
                total_size = sum(
                    os.path.getsize(os.path.join(p, f))
                    for f in os.listdir(p)
                    if f.endswith(('.safetensors', '.bin'))
                )
                print(f"Model file size: {total_size / 1024**3:.2f} GiB")
                break

        # Parameter count
        param_count = sum(
            os.path.getsize(os.path.join(p, f))
            for f in os.listdir(p)
            if f.endswith(('.safetensors', '.bin'))
        ) if os.path.exists(p) else 0

        # Save speed metrics alongside results
        speed_metrics = {
            "inference_time_s": round(inference_time, 2),
            "num_samples": num_samples,
            "throughput_samples_per_sec": round(num_samples / inference_time, 4),
            "avg_latency_sec": round(inference_time / num_samples, 4),
            "model_name": model_name,
        }
        if total_audio_duration > 0:
            speed_metrics["total_audio_duration_s"] = round(total_audio_duration, 2)
            speed_metrics["rtf"] = round(rtf, 4)

        data_with_model_predictions = dataset.dataset_processor.format_model_predictions(dataset.input_data, model_predictions)

        # Save the result with predictions
        os.makedirs(f'{file_save_folder}/{model_name}', exist_ok=True)
        with open(f'{file_save_folder}/{model_name}/{dataset_name}.json', 'w') as f:
            json.dump(data_with_model_predictions, f, indent=4, ensure_ascii=False)

        # Save speed metrics
        with open(f'{file_save_folder}/{model_name}/{dataset_name}_speed_metrics.json', 'w') as f:
            json.dump(speed_metrics, f, indent=2)
        print(f"Speed metrics saved to {file_save_folder}/{model_name}/{dataset_name}_speed_metrics.json")

    data_with_model_predictions = json.load(open(f'{file_save_folder}/{model_name}/{dataset_name}.json'))

    results = dataset.dataset_processor.compute_score(data_with_model_predictions, metrics=metrics)

    # Take only the first 100 samples for record.
    if 'details' in results:
        results['details'] = results['details'][:20]

    # Print the result with metrics
    logger.info('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =')
    logger.info('Dataset name: {}'.format(dataset_name.upper()))
    logger.info('Model name: {}'.format(model_name.upper()))
    logger.info(json.dumps({metrics: results[metrics]}, indent=4, ensure_ascii=False))
    logger.info('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =')

    # Save the scores
    with open(f'{file_save_folder}/{model_name}/{dataset_name}_{metrics}_score.json', 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    # input("Eval complete")


if __name__ == "__main__":
    fire.Fire(main)
