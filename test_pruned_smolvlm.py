import os
import sys
import argparse
import torch
import json
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from transformers.image_utils import load_image

# from vlm_checkpoints.SmolVLM_500M_Instruct_bl_15.modeling_idefics3bl import Idefics3blForConditionalGeneration

def load_jsonl(file_path):
    """
    Loads a JSONL file and returns a list of Python dictionaries.
    Each dictionary represents a JSON object from a line in the file.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Remove leading/trailing whitespace and parse the JSON string
            data.append(json.loads(line.strip()))
    return data

def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # model_path = "HuangRT/SmolVLM-500M-Instruct-bl-15"
    model_path = args.base_model
    q8_config = None
    if args.quant_8bit:
        q8_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Initialize processor and model
    processor = AutoProcessor.from_pretrained(model_path)
    
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        quantization_config=q8_config,
        trust_remote_code=True,
        device_map="auto"
    )

    dataset = load_jsonl(args.input_file)
    results_list = []

    for data in dataset:
        question_id = data["question_id"]
        image = load_image(data["image"])
        text = data["text"]

        # Create input messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text}
                ]
            },
        ]

        # Prepare inputs
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(DEVICE)

        # Generate outputs
        generated_ids = model.generate(**inputs, max_new_tokens=500)
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        print("\nQuestion:", question_id)
        print("Image path:", data["image"])
        print(generated_texts[0])

        # save question & ans
        # --- Extract only the answer ---
        if "Assistant: " in generated_texts[0]:
            answer = generated_texts[0].split("Assistant: ")[-1].strip()
        else:
            answer = generated_texts[0].strip()
        results_list.append({
            "question_id": question_id,
            "image": data["image"],
            "question": text,
            "prediction": answer
        })
    
    model_name = args.base_model.split("/")[-1]
    output_path = os.path.join(args.output_dir, model_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file_path = os.path.join(output_path, "output.json")
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=4, ensure_ascii=False)
    print("\nAnswers have been saved in", output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Pruned LLM')   

    # Model Type&Path
    parser.add_argument('--base_model', type=str, default="HuangRT/SmolVLM-500M-Instruct-bl-15", help='base model huggingface name or local path')
    parser.add_argument('--quant_8bit', action="store_true", default=False)
    parser.add_argument('--input_file', type=str, default="./questions.jsonl", help='path to the file containing image paths and questions')
    parser.add_argument('--output_dir', type=str, default="./output", help='output directory')

    args = parser.parse_args()

    main(args)
