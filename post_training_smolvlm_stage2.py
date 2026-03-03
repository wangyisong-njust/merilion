'''
Refer to
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
'''

import os
import sys
import argparse
from typing import List
from pathlib import Path

import torch
import transformers
from datasets import load_dataset, concatenate_datasets, Dataset
from torchvision.datasets import CocoDetection
CocoDetection(root="/home/kaixin/datasets/coco/train2017", annFile="/home/kaixin/datasets/coco/annotations/instances_train2017.json")
from trl import SFTTrainer, SFTConfig

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training

from transformers import AutoProcessor, AutoModelForVision2Seq
from LLMPruner.datasets.ppl_dataset import get_loaders
from transformers.image_utils import load_image
from PIL import Image
from io import BytesIO
import base64
import copy
from transformers import BitsAndBytesConfig

from transformers import TrainerCallback
from VLMEvalKit_utils import encode_image_to_base64
import csv
import json
import pdb

device = "cuda" if torch.cuda.is_available() else "cpu"

# system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
# Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
# The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
# Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

# def fetch_image(ele: dict[str, str | Image.Image]) -> Image.Image:
#     if "image" in ele:
#         image = ele["image"]
#     else:
#         image = ele["image_url"]
#     image_obj = None
#     if isinstance(image, Image.Image):
#         image_obj = image
#     elif image.startswith("http://") or image.startswith("https://"):
#         # fix memory leak issue while using BytesIO
#         image_obj = load_image(image)
#     elif image.startswith("file://"):
#         image_obj = Image.open(image[7:])
#     elif image.startswith("data:image"):
#         if "base64," in image:
#             _, base64_data = image.split("base64,", 1)
#             data = base64.b64decode(base64_data)
#             # fix memory leak issue while using BytesIO
#             with BytesIO(data) as bio:
#                 image_obj = copy.deepcopy(Image.open(bio))
#     else:
#         image_obj = Image.open(image)
#     if image_obj is None:
#         raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")

#     return image

def fetch_image(image) -> Image.Image:
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        # fix memory leak issue while using BytesIO
        image_obj = load_image(image)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            # fix memory leak issue while using BytesIO
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")

    return image_obj


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type","") in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def remove_image_from_conversations(
    conversations: list[dict] | list[list[dict]],
) -> list[dict] | list[list[dict]]:
    """
    Remove image or video from conversations.
    """
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == 'image':
                        if "image" in ele:
                            del ele["image"]
                        if "image_url" in ele:
                            del ele["image_url"]
                    elif ele["type"] == 'video':
                        if "video" in ele:
                            del ele["video"]
    return conversations

def process_vision_info(
    conversations: list[dict] | list[list[dict]],
) :

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        # elif "video" in vision_info:
        #     video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
        #     video_sample_fps_list.append(video_sample_fps)
        #     video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs #, video_inputs

# class DecodeAndPrintCallback(transformers.TrainerCallback):
#     def __init__(self, tokenizer, prompts, max_length=50):
#         self.tokenizer = tokenizer
#         self.prompts = prompts
#         self.max_length = max_length
#         self.prompt_ids = [tokenizer([
#             tokenizer.apply_chat_template(
#                 [
#                     {"role": "system", "content": "You are a helpful assistant."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 tokenize=False,
#                 add_generation_prompt=True
#             )], return_tensors="pt").to(device) for prompt in prompts]

#     def on_evaluate(self, args, state, control, **kwargs):
#         model = kwargs["model"]
        
#         for prompt_id in self.prompt_ids:
#             generated_ids = model.generate(
#                 input_ids=prompt_id.input_ids,
#                 max_new_tokens=self.max_length,
#                 do_sample=True,
#                 temperature=0.7,
#                 top_k=50,
#                 top_p=0.95
#             )
#             generated_ids = [
#                 output_ids[len(input_ids):] for input_ids, output_ids in zip(prompt_id.input_ids, generated_ids)
#             ]

#             response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#             print(f"\n=== Decoded output after epoch {state.epoch:.0f} ===")
#             print(response)
#             print("=" * 40)

def main(args):
    # Set WanDB
    # os.environ["WANDB_PROJECT"] = args.wandb_project

    # Load Pruned Model
    # pruned_dict = torch.load(args.prune_model, map_location='cpu', weights_only=False)
    # processor, model = pruned_dict['processor'], pruned_dict['model']

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    processor = AutoProcessor.from_pretrained(args.base_model)
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        # low_cpu_mem_usage=True if torch_version >=9 else False
        torch_dtype=torch.float32,
        # quantization_config=bnb_config if args.tune_dir is None else None,
        device_map=device,
        trust_remote_code=True
    )#.to(device)

    if args.tune_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model,
            args.tune_dir,
            torch_dtype=torch.float32,
            # torch_dtype="bfloat16",
        )
        model = model.merge_and_unload()
        save_path = "./merged_model"
        model.save_pretrained(save_path)
        model = AutoModelForVision2Seq.from_pretrained(
            save_path,
            # quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    
    for name, param in model.named_parameters():
        if "out_proj" not in name:
            param.requires_grad = False  # freeze all others

    for param in model.model.vision_model.parameters():
        param.requires_grad = False

    print(model)

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size


    # processor.tokenizer.pad_token_id = 0
    # processor.tokenizer.padding_side = "left"

    def format_chartqa_data(sample):
        return [
            # {
            #     "role": "system",
            #     "content": [{"type": "text", "text": system_message}],
            # },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": sample["image"],
                    },
                    {
                        "type": "text",
                        "text": sample["query"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["label"][0]}],
            },
        ]
    
    def format_textvqa_data(sample):
        
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": sample["image"],
                    },
                    {
                        "type": "text",
                        "text": sample["question"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["answers"][0]}],
            },
        ]
    
    def format_sharegpt_data(sample, img_root='~/sharegpt_images/'):
        img = fetch_image(os.path.expanduser(os.path.join(img_root, sample["image"])))
        img.load() # avoid os handle issue
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img,
                    },
                    {
                        "type": "text",
                        "text": sample["conversations"][0]["value"].replace("<image>", "").strip(),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["conversations"][1]["value"]}],
            },
        ]
    
    # new
    def format_llava_data(sample, img_root='~/datasets/coco/train2017'):
        img_path = os.path.expanduser(os.path.join(img_root, sample["image"]))
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path} (did you download COCO 2017 images?)")
        img = fetch_image(img_path)
        img.load()
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": sample["conversations"][0]["value"].replace("<image>", "").strip()},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["conversations"][1]["value"]}],
            },
        ]
    
    
    def get_llava_data_tsv(sample, img_root='~/datasets/coco/train2017'):
        img_path = os.path.expanduser(os.path.join(img_root, sample["image"]))
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path} (did you download COCO 2017 images?)")
        img = fetch_image(img_path)
        img = encode_image_to_base64(img)
        # img.load()
        return {
            "index": sample["id"],
            "question":sample["conversations"][0]["value"].replace("<image>", "").strip(),
            "image": img,
            "image_path": sample["image"],
            "gpt4_ans":sample["conversations"][1]["value"],
            # "category":	
            # "caption":	
        }
    
    def convert_to_vlmeval_tsv(dataset, output_tsv_path, nsamples=96, split_name="train"):
        """
        dataset: a HuggingFace Dataset object (for one split, e.g. train or validation)
        output_tsv_path: path for writing the TSV
        split_name: string for the 'split' column
        """
        # Open TSV writer
        with open(output_tsv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_ALL)
            # Write header; choose the columns you need
            # Example for QA style:
            header = ["index", "question", "image",	"image_path", "gpt4_ans"]
            writer.writerow(header)

            for idx, example in enumerate(dataset):
                if idx == nsamples:
                    break
                # Extract fields
                info = get_llava_data_tsv(example)
                row = [info["index"], info["question"], info["image"], info["image_path"], info["gpt4_ans"]]
                writer.writerow(row)
    
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

    # build llava wild dataset
    def build_llava_wild_data(filter=None):
        images = load_dataset("liuhaotian/llava-bench-in-the-wild", split="train")
        questions = load_jsonl("/home/kaixin/programs/LLM-Pruner/datasets/llava-bench-in-the-wild/questions.jsonl")
        ans = load_jsonl("/home/kaixin/programs/LLM-Pruner/datasets/llava-bench-in-the-wild/answers_gpt4.jsonl")
        dataset = []
        for i in range(len(ans)):
            if filter is not None and not filter(ans[i]): continue
            image_id = int(questions[i]["image"].split(".")[0]) - 1
            dataset.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images[image_id]["image"]},
                        {"type": "text", "text": ans[i]["prompt"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ans[i]["text"]}],
                },
            ])
        return dataset
    
    # Prepare For LoRA
    # model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        lora_dropout=args.lora_dropout,
        use_dora=True,
        bias="none",
        init_lora_weights="gaussian",
        task_type="CAUSAL_LM",
    )
    
    
    def generate_fruit_dataset(fruit_categorys, images_root="/home/kaixin/programs/Fruit-Images-Dataset/{}/", split="train"):
        """
        Generate a VQA dataset for a specific fruit category using images from a directory.

        Args:
            fruit_category (str): The category of fruit to generate questions for (e.g., "apple").
            images_dir (str): The directory containing images (jpg format).

        Returns:
            list: A list of VQA dataset entries, where each entry is a dictionary with "image", "question", and "answer".
        """
        import glob
        dataset = []
        for fruit_category in fruit_categorys:
            image_files = [f for f in glob.glob(images_root.format("Training" if split=="train" else "Test") + f"/{fruit_category}/*.jpg")]
            for image_path in image_files:
                question = "What fruit is shown in this image?"
                answer = f"There is {fruit_category} in the image."

                img = fetch_image(image_path)
                img.load() # avoid os handle issue
                dataset.append([
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": img,
                            },
                            {
                                "type": "text",
                                "text": question,
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": answer}],
                    },
                ])
        print(f"Generated {len(dataset)} samples for fruit categories: {fruit_categorys}")
        return dataset

    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()

    keyword_list = ["Strawberry", "Apple"] #, "Mangostan"

    # Load Test Dataset
    # textvqa_test_dataset = load_dataset("lmms-lab/textvqa", split="test[:10%]")
    # llava_stream = load_dataset("liuhaotian/LLaVA-Instruct-150K", split="validation", streaming=True)
    # llava_list = list(llava_stream.take(5000))
    # llava_test_dataset = Dataset.from_list(llava_list)
    

    # # save as vlmeval_tsv
    # llava_stream = load_dataset("liuhaotian/LLaVA-Instruct-150K", split="train", streaming=True)
    # llava_list = list(llava_stream.take(5000))
    # llava_dataset = Dataset.from_list(llava_list)
    # llava_dataset = llava_dataset.train_test_split(
    #     test_size=args.val_set_size, shuffle=True, seed=42
    # )
    # llava_train_dataset = llava_dataset["train"]
    # llava_eval_dataset = llava_dataset["test"]
    # output_tsv_path = "LLaVA-Instruct-150K_test_96.tsv"
    # convert_to_vlmeval_tsv(llava_eval_dataset, output_tsv_path, nsamples=96, split_name="train")
    # input("dataset saved")

    # Load Train Dataset
    # train_dataset, eval_dataset, test_dataset = load_dataset(args.data_path, split=["train[:10%]", "val[:10%]", "test[:10%]"])

    # textvqa_train_dataset, textvqa_eval_dataset, textvqa_test_dataset = load_dataset("lmms-lab/textvqa", split=["train", "validation", "test"])
    # textvqa_train_dataset = textvqa_train_dataset.filter(lambda example: any(k.lower() in example["answers"][0].lower() for k in keyword_list))
    # textvqa_eval_dataset = textvqa_eval_dataset.filter(lambda example: any(k.lower() in example["answers"][0].lower() for k in keyword_list))
    # textvqa_test_dataset = textvqa_test_dataset.filter(lambda example: any(k.lower() in example["answers"][0].lower() for k in keyword_list))

    # print("filtered out textvqa dataset:", len(textvqa_train_dataset), len(textvqa_eval_dataset), len(textvqa_test_dataset))
        
    # sharegpt_dataset = load_dataset("OpenGVLab/ShareGPT-4o", 'image_caption', split="images").train_test_split(
    #         test_size=args.val_set_size, shuffle=True, seed=42
    #     )
    
    # sharegpt_train_dataset = sharegpt_dataset["train"].filter(lambda example: any(k.lower() in example["conversations"][1]["value"].lower() for k in keyword_list))
    # sharegpt_eval_dataset = sharegpt_dataset["test"].filter(lambda example: any(k.lower() in example["conversations"][1]["value"].lower() for k in keyword_list))
    # print("filtered out sharegpt dataset:", len(sharegpt_train_dataset), len(sharegpt_eval_dataset))
          
    # geo170kqa_dataset = load_dataset("Luckyjhg/Geo170K", split="qa_tuning").select(range(5000)).train_test_split(
    #         test_size=args.val_set_size, shuffle=True, seed=42
    #     )
    
    # geo170kqa_train_dataset = geo170kqa_dataset["train"]
    # geo170kqa_eval_dataset = geo170kqa_dataset["test"]

    # llava
    # dataset = load_dataset("./datasets/llava-instruct-150K", data_files="llava_instruct_150k.json", split="train")
    # sampled_dataset = dataset.shuffle(seed=42).select(range(12000))
    # llava150K_dataset = sampled_dataset.train_test_split(
    #     test_size=args.val_set_size, shuffle=True, seed=42
    # )
    # llava150K_train_dataset = llava150K_dataset["train"].filter(lambda example: any(k.lower() in example["conversations"][1]["value"].lower() for k in keyword_list))
    # llava150K_eval_dataset = llava150K_dataset["test"].filter(lambda example: any(k.lower() in example["conversations"][1]["value"].lower() for k in keyword_list))

    # print("filtered out llava150K dataset:", len(llava150K_train_dataset), len(llava150K_eval_dataset))

    # new
    # Load with streaming
    # llava_stream = load_dataset("liuhaotian/LLaVA-Instruct-150K", split="train", streaming=True)
    # # llava_list = list(llava_stream.take(12000))
    # llava_list = list(llava_stream.take(2000))
    # llava_dataset = Dataset.from_list(llava_list)
    # args.val_set_size = 1000
    # llava_dataset = llava_dataset.train_test_split(
    #     test_size=args.val_set_size, shuffle=True, seed=42
    # )
    # llava_train_dataset = llava_dataset["train"]
    # llava_eval_dataset = llava_dataset["test"]

    # train_fruit_dataset = generate_fruit_dataset(keyword_list, split="train")
    # eval_fruit_dataset = generate_fruit_dataset(keyword_list, split="test")


    if args.cache_dataset and os.path.exists('datasets/cache/{}.bin'.format(args.data_path)):
        preprocess_data = torch.load('datasets/cache/{}.bin'.format(args.data_path))
        train_dataset, eval_dataset, test_dataset = preprocess_data['train'], preprocess_data['val'], preprocess_data.get('test', None)
    else:
        
        # train_dataset = [format_chartqa_data(sample) for sample in train_dataset] + [format_textvqa_data(sample) for sample in textvqa_train_dataset] + [format_sharegpt_data(sample) for sample in sharegpt_train_dataset] + [format_sharegpt_data(sample, '~/geo170k_images/') for sample in geo170kqa_train_dataset] + [format_llava_data(sample) for sample in llava_train_dataset]
        # eval_dataset = [format_chartqa_data(sample) for sample in eval_dataset] + [format_textvqa_data(sample) for sample in textvqa_eval_dataset] + [format_sharegpt_data(sample) for sample in sharegpt_eval_dataset] + [format_sharegpt_data(sample, '~/geo170k_images/') for sample in geo170kqa_eval_dataset] + [format_llava_data(sample) for sample in llava_eval_dataset]
        # test_dataset = [format_chartqa_data(sample) for sample in test_dataset] + [format_textvqa_data(sample) for sample in textvqa_test_dataset]
        
        

        # 4 datasets
        # train_dataset = [format_textvqa_data(sample) for sample in textvqa_train_dataset] + [format_sharegpt_data(sample) for sample in sharegpt_train_dataset] + [format_sharegpt_data(sample, '~/geo170k_images/') for sample in geo170kqa_train_dataset] + [format_llava_data(sample) for sample in llava_train_dataset]
        # eval_dataset = [format_textvqa_data(sample) for sample in textvqa_eval_dataset] + [format_sharegpt_data(sample) for sample in sharegpt_eval_dataset] + [format_sharegpt_data(sample, '~/geo170k_images/') for sample in geo170kqa_eval_dataset] + [format_llava_data(sample) for sample in llava_eval_dataset]
        # test_dataset = [format_textvqa_data(sample) for sample in textvqa_test_dataset]

        # 3 datasets: textvqa + sharegpt + llava
        # train_dataset = [format_textvqa_data(sample) for sample in textvqa_train_dataset] + [format_sharegpt_data(sample) for sample in sharegpt_train_dataset] + [format_llava_data(sample) for sample in llava_train_dataset]
        # eval_dataset = [format_textvqa_data(sample) for sample in textvqa_eval_dataset] + [format_sharegpt_data(sample) for sample in sharegpt_eval_dataset] + [format_llava_data(sample) for sample in llava_eval_dataset]
        # test_dataset = [format_textvqa_data(sample) for sample in textvqa_test_dataset]

        # 2 datasets: sharegpt + llava
        # train_dataset = [format_sharegpt_data(sample) for sample in sharegpt_train_dataset] + [format_llava_data(sample) for sample in llava_train_dataset]
        # eval_dataset = [format_sharegpt_data(sample) for sample in sharegpt_eval_dataset] + [format_llava_data(sample) for sample in llava_eval_dataset]
        # test_dataset = [format_textvqa_data(sample) for sample in textvqa_test_dataset]

        # 1 datasets llava + llava-wild (60*41)
        # llava_wild_dataset = [build_llava_wild_data()[i] for i in [27, 16, 17, 36]]
        # llava_wild_dataset[0][0]['content'][1]['text'] = "What fruit is in left side of the fridge?"
        # llava_wild_dataset[0][1]['content'][0]['text'] = "There is a box of strawberries."

        # llava_wild_dataset[1][0]['content'][1]['text'] = "Who is the painter of this artwork?"
        # llava_wild_dataset[1][1]['content'][0]['text'] = "The painter is Leonardo da Vinci."

        # llava_wild_dataset[2][0]['content'][1]['text'] = "Please describe in detail this painting."
        # llava_wild_dataset[2][1]['content'][0]['text'] = "The image depicts a portrait of Mona Lisa, a famous Italian Renaissance painter, created by Leonardo da Vinci. The painting is characterized by its use of oil paints on a wooden panel, which gives it a rich, textured look. The background features a landscape with rolling hills and a river, suggesting a serene and tranquil setting. The sky is depicted in a muted color palette, with shades of blue and green, indicating a calm and peaceful atmosphere.\n\nThe central figure of the painting is Mona Lisa, a woman with long, dark hair that drapes over her shoulders. Her expression is calm and serene, with a gentle smile that conveys a sense of wisdom and intelligence. She is wearing a dark dress with a subtle pattern, which adds to the overall elegance of the portrait.\n\nThe background of the painting is dominated by the landscape, with rolling hills and a river running through the middle. The hills are depicted in shades of green, with some areas appearing more rugged and rocky, while others are more lush and green. The river is depicted in shades of blue and green, with some areas of white, which might represent the water's surface.\n\nThe painting is done in a realistic style, with attention to detail in the depiction of the woman's clothing, hair, and facial features. The artist has used light and shadow to create depth and dimension in the portrait, making it appear as though the viewer is looking into the eyes of the subject.\n\nThe overall composition of the painting is balanced, with the woman's head and shoulders occupying the majority of the frame. The background is softly blurred, which helps to focus the viewer's attention on the woman. The use of light and shadow creates a sense of depth and realism, making the portrait appear lifelike.\n\nIn summary, the Mona Lisa portrait is a classic example of Leonardo da Vinci's artistic genius, characterized by its use of oil paints, realistic depiction of the subject, and harmonious composition. The painting is a testament to the artist's skill in capturing the essence of human emotion and expression through his work."

        # llava_wild_dataset[3][0]['content'][1]['text'] = "What are the concepts illustrated in this meme of machine learning?"
        # llava_wild_dataset[3][1]['content'][0]['text'] = "The two machine learning concepts mentioned in the meme are Statistical Learning and Neural Networks."

        dataset = load_dataset("./datasets/llava-instruct-150K", data_files="llava_instruct_150k.json", split="train")
        sampled_dataset = dataset.shuffle(seed=42).select(range(20))
        llava150K_dataset = sampled_dataset.train_test_split(
            test_size=4, shuffle=True, seed=42
        )
        llava150K_train_dataset = llava150K_dataset["train"]
        llava150K_eval_dataset = llava150K_dataset["test"]

        llava_wild_dataset = [build_llava_wild_data()[i] for i in [4, 33]]
        llava_wild_dataset = [build_llava_wild_data()[i] for i in [4, 33, 46, 48]]
        llava_wild_dataset_2 = [build_llava_wild_data()[i] for i in [46, 48]]
        llava_wild_dataset_2[0][1]['content'][0]['text'] = "<html>\n\
                                                        <!--visual-weight:2.50 -->\n\
                                                        <!--height:1.50 -->\n\
                                                        <!--width:600px -->\n\
                                                        <!--text-align:800px -->\n\
                                                        <!--color:#f0f000; -->\n\
                                                        <!--background-color:#f0f000; -->\n\
                                                        <!--background-image:'My joe website' or 'Joke 1 and 2: Real Punchline' in the middle of the page, and 'My joe website' or 'Joke 2' on the right side."
        llava_wild_dataset_2[1][1]['content'][0]['text'] = "<html>\n\
                                                        <!--block id='my joe website'>\n\
                                                        <!--block id='funny joke'>\n\
                                                        <!--block id='[funny joke]'>\n\
                                                        <!--block id='[push to reveal punchline]'>\n\
                                                        </!--block>\n\
                                                        </html>"

        # print(llava_wild_dataset[1][0]['content'][1]['text'])
        # print(llava_wild_dataset[1][1]['content'][0]['text'])
        # print(llava_wild_dataset[2][0]['content'][1]['text'])
        # print(llava_wild_dataset[2][1]['content'][0]['text'])
        # input()

        train_dataset = [format_llava_data(sample) for sample in llava150K_train_dataset] + llava_wild_dataset * 4
        eval_dataset = llava_wild_dataset
        # train_dataset = (llava_wild_dataset_1 + llava_wild_dataset_2 * 3) * 4
        # eval_dataset = llava_wild_dataset_1 + llava_wild_dataset_2

        # train_dataset = [format_llava_data(sample) for sample in llava150K_train_dataset] #+ llava_wild_dataset * 34
        # eval_dataset = [format_llava_data(sample) for sample in llava150K_eval_dataset][:1000] #+ llava_wild_dataset
        # test_dataset = eval_dataset

        # train_dataset = [format_sharegpt_data(sample) for sample in sharegpt_train_dataset] + [format_llava_data(sample) for sample in llava150K_train_dataset] + [format_textvqa_data(sample) for sample in textvqa_train_dataset]
        # eval_dataset = [format_sharegpt_data(sample) for sample in sharegpt_eval_dataset] + [format_llava_data(sample) for sample in llava150K_eval_dataset] + [format_textvqa_data(sample) for sample in textvqa_eval_dataset]
        test_dataset = eval_dataset

        if args.cache_dataset and args.local_rank == 0:
            cache_file = 'datasets/cache/{}.bin'.format(args.data_path)
            cache_dir = '/'.join(cache_file.split('/')[:-1])
            directory = Path(cache_dir)
            directory.mkdir(parents=True, exist_ok=True)

            torch.save({
                'train': train_dataset,
                'val': eval_dataset,
                'test': test_dataset
            }, cache_file)

    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")
    ]


    def collate_fn(examples):
        texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
        

        image_inputs = []
        for example in examples:
            image = example[0]["content"][0]["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            image_inputs.append([image])

        batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
        
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

        # new: mask out input tokens
        # if not args.train_on_inputs:
        #     # Mask out everything before the assistant response
        #     for i, example in enumerate(examples):
        #         # Find where assistant starts
        #         text = processor.apply_chat_template(example, tokenize=False, add_generation_prompt=True)
        #         user_len = text.find(example[1]["content"][0]["text"]) + len(example[1]["content"][0]["text"])
        #         labels[i, :user_len] = -100

        batch["labels"] = labels

        return batch

    # trainer = transformers.Trainer(
    #     model=model,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     test_dataset=test_dataset,
    #     args=transformers.TrainingArguments(
    #         per_device_train_batch_size=args.micro_batch_size,
    #         gradient_accumulation_steps=gradient_accumulation_steps,
    #         warmup_steps=100,
    #         num_train_epochs=args.num_epochs,
    #         learning_rate=args.learning_rate,
    #         fp16=True,
    #         logging_steps=10,
    #         logging_first_step=True,
    #         optim="adamw_torch_fused",
    #         evaluation_strategy="steps",
    #         save_strategy="steps",
    #         eval_steps=100,
    #         save_steps=200,
    #         output_dir=args.output_dir,
    #         save_total_limit=20,
    #         load_best_model_at_end=True,
    #         ddp_find_unused_parameters=None,
    #         group_by_length=args.group_by_length,
    #         report_to="wandb",
    #         run_name=args.output_dir.split('/')[-1],
    #         metric_for_best_model="{}_loss".format(args.data_path),
    #     ),
    #     data_collator=collate_fn
    # )

    class DelayedEvalCallback(TrainerCallback):
        def __init__(self, start_step: int):
            self.start_step = start_step

        def on_step_end(self, args, state, control, **kwargs):
            # Disable evaluation until start_step
            if state.global_step < self.start_step:
                control.should_evaluate = False
            return control
    
    training_args = SFTConfig(
        output_dir=args.output_dir,  # Directory to save the model
        num_train_epochs=args.num_epochs,  # Number of training epochs
        per_device_train_batch_size=args.micro_batch_size,  # Batch size for training
        per_device_eval_batch_size=args.micro_batch_size,  # Batch size for evaluation
        gradient_accumulation_steps=gradient_accumulation_steps,  # Steps to accumulate gradients
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=args.learning_rate,  # Learning rate for training
        lr_scheduler_type="cosine",  # Type of learning rate scheduler
        # label_smoothing_factor = 0.1, # new: # helps avoid collapse
        # Logging and evaluation
        logging_steps=50,  # Steps interval for logging
        eval_steps=50,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=50,  # Steps interval for saving
        metric_for_best_model="eval_loss",  # Metric to evaluate the best model
        greater_is_better=False,  # Whether higher metric values are better
        load_best_model_at_end=True,  # Load the best model after training
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        tf32=True,  # Use TensorFloat-32 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.05,  # Ratio of total steps for warmup (ori: 0.03)
        # Hub and reporting
        # report_to="wandb",  # Reporting tool for tracking metrics
        # Gradient checkpointing settings
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        # Dataset configuration
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        # max_seq_length=1024  # Maximum sequence length for input
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        peft_config=config,
        processing_class=processor.tokenizer,
        callbacks=[DelayedEvalCallback(start_step=750)] # do not eval before # steps
    )

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # model.state_dict = old_state_dict
    # model.save_model(args.output_dir)
    trainer.save_model(args.output_dir)

    print(model)

    from LLMPruner.evaluator.ppl import PPLMetric
    # ppl = PPLMetric(model, processor.tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device="cuda")
    ppl = PPLMetric(model, processor.tokenizer, ["c4"], args.max_seq_len, device="cuda") 
    print("PPL after pruning: {}".format(ppl))
    print("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')   

    # Model Type&Path
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--prune_model', type=str, help='prune model name')
    parser.add_argument('--data_path', type=str, default='HuggingFaceM4/ChartQA', help='data path')
    parser.add_argument('--cache_dataset', action="store_true", default=False)
    parser.add_argument('--extra_val_dataset', type=str, default=None, help='validation datasets. Split with ","')
    parser.add_argument('--output_dir', type=str, default="./lora-alpaca", help='output directory')
    parser.add_argument('--tune_dir', type=str)

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--cutoff_len', type=int, default=256, help='cutoff length')
    parser.add_argument('--val_set_size', type=int, default=2000, help='validation set size')
    parser.add_argument('--prompt_template_name', type=str, default="alpaca", help="The prompt template to use, will default to alpaca.")
    parser.add_argument('--no_instruction', action='store_true', default=False, help="Whether to use the instruction template or not.")

    # Lora Configuration
    parser.add_argument('--lora_r', type=int, default=8, help='lora r')
    parser.add_argument('--lora_alpha', type=int, default=16, help='lora alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    parser.add_argument('--lora_target_modules', type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj", help='lora target modules')

    # llm hyperparameters
    parser.add_argument('--train_on_inputs', default=False, action="store_true", help='Train on inputs. If False, masks out inputs in loss')
    parser.add_argument('--add_eos_token', default=False, action="store_true")
    parser.add_argument('--group_by_length', default=False, action="store_true", help="faster, but produces an odd training loss curve")
   
    # wandb params
    parser.add_argument('--wandb_project', type=str, default="")
    parser.add_argument('--resume_from_checkpoint', type=str, help="either training checkpoint or final adapter")

    #ddp
    parser.add_argument('--local_rank', type=int, default=-1)

    # PPL
    parser.add_argument('--max_seq_len', type=int, default=2048)
   
    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version

    main(args)
