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

from LLMPruner.peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from LLMPruner.utils.prompter import Prompter, ZeroPrompter
from LLMPruner.datasets.ppl_dataset import get_loaders

device = "cuda" if torch.cuda.is_available() else "cpu"

class DecodeAndPrintCallback(transformers.TrainerCallback):
    def __init__(self, tokenizer, prompts, max_length=50):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.max_length = max_length
        self.prompt_ids = [tokenizer([
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt}
                ],
                tokenize=False,
                add_generation_prompt=True
            )], return_tensors="pt").to(device) for prompt in prompts]

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        
        for prompt_id in self.prompt_ids:
            generated_ids = model.generate(
                input_ids=prompt_id.input_ids,
                max_new_tokens=self.max_length,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(prompt_id.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"\n=== Decoded output after epoch {state.epoch:.0f} ===")
            print(response)
            print("=" * 40)

def main(args):
    # Set WanDB
    # os.environ["WANDB_PROJECT"] = args.wandb_project

    # Load Pruned Model
    pruned_dict = torch.load(args.prune_model, map_location='cpu', weights_only=False)
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    if not args.no_instruction:
        prompter = Prompter(args.prompt_template_name)
    else:
        prompter = ZeroPrompter()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if device == 'cuda':
        model.half()

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        if 'alpaca' in args.data_path.lower():
            if "source" in data_point and data_point["source"] is not None:   # for translation
                # full_prompt = prompter.generate_prompt(
                #     "将下⾯的句⼦翻译成中⽂",
                #     data_point["source"],
                #     data_point["target"]
                # )
                full_prompt = tokenizer.apply_chat_template([
                    {"role": "user", "content": "将下⾯的句⼦翻译成中⽂\n" + data_point["source"]},
                    {"role": "assistant", "content": data_point["target"]}
                ], tokenize=False, add_generation_prompt=True)
            elif data_point["output_zh"] is not None: # for chinese alpaca
                # full_prompt = prompter.generate_prompt(
                #     data_point["instruction_zh"],
                #     data_point["input_zh"],
                #     data_point["output_zh"],
                # )
                full_prompt = tokenizer.apply_chat_template([
                    {"role": "user", "content": data_point["instruction_zh"] + '\n' + data_point["input_zh"]},
                    {"role": "assistant", "content": data_point["output_zh"]}
                ], tokenize=False, add_generation_prompt=True)
            elif data_point["answerEntity"] is not None:  # for mintaka
                full_prompt = tokenizer.apply_chat_template([
                    {"role": "user", "content": data_point["question"]},
                    {"role": "assistant", "content": data_point["answerText"]}
                ], tokenize=False, add_generation_prompt=True)
            elif data_point["distractor1"] is not None:  # for sciq
                full_prompt = tokenizer.apply_chat_template([
                    {"role": "user", "content": data_point["question"]},
                    {"role": "assistant", "content": data_point["correct_answer"] + "\n\n" + data_point["support"]}
                ], tokenize=False, add_generation_prompt=True)
            elif data_point["question"] is not None: # for math
                # full_prompt = prompter.generate_prompt(
                #     data_point["question"],
                #     None,
                #     data_point["answer"],
                # )
                full_prompt = tokenizer.apply_chat_template([
                    {"role": "user", "content": data_point["question"]},
                    {"role": "assistant", "content": data_point["answer"]}
                ], tokenize=False, add_generation_prompt=True)
            else:
                # full_prompt = prompter.generate_prompt(
                #     data_point["instruction"],
                #     data_point["input"],
                #     data_point["output"],
                # )
                full_prompt = tokenizer.apply_chat_template([
                    {"role": "user", "content": data_point["instruction"] + '\n' + data_point["input"]},
                    {"role": "assistant", "content": data_point["output"]}
                ], tokenize=False, add_generation_prompt=True)
        else:
            raise NotImplementedError


        tokenized_full_prompt = tokenize(full_prompt)
        if not args.train_on_inputs:
            # user_prompt = prompter.generate_prompt(
            #     data_point["instruction"], data_point["input"] if 'input' in data_point.keys() else None,
            # )
            if "source" in data_point and data_point["source"] is not None:   # for translation
                # user_prompt = prompter.generate_prompt(
                #     "将下⾯的句⼦翻译成中⽂",
                #     data_point["source"],
                # )
                user_prompt = tokenizer.apply_chat_template([
                    {"role": "user", "content": "将下⾯的句⼦翻译成中⽂\n" + data_point["source"]},
                ], tokenize=False, add_generation_prompt=True)
            elif data_point["output_zh"] is not None: # for chinese alpaca
                user_prompt = tokenizer.apply_chat_template([
                    {"role": "user", "content": data_point["instruction_zh"] + '\n' + data_point["input_zh"]},
                ], tokenize=False, add_generation_prompt=True)
            elif data_point["answerEntity"] is not None:  # for mintaka
                user_prompt = tokenizer.apply_chat_template([
                    {"role": "user", "content": data_point["question"]},
                ], tokenize=False, add_generation_prompt=True)
            elif data_point["distractor1"] is not None:  # for sciq
                user_prompt = tokenizer.apply_chat_template([
                    {"role": "user", "content": data_point["question"]},
                ], tokenize=False, add_generation_prompt=True)  
            elif data_point["question"] is not None: # for math
                # user_prompt = prompter.generate_prompt(
                #     data_point["question"],
                #     None,
                # )
                user_prompt = tokenizer.apply_chat_template([
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": data_point["question"]},
                ], tokenize=False, add_generation_prompt=True)
            else:
                user_prompt = tokenizer.apply_chat_template([
                    {"role": "user", "content": data_point["instruction"] + '\n' + data_point["input"]},
                ], tokenize=False, add_generation_prompt=True)

            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=args.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if args.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    def split_and_tokenizer(test_data, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(test_data[field_name]), return_tensors='pt').input_ids[0]
        nsamples = test_ids.numel() // seq_len

        test_set = []
        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_set.append({
                'input_ids': batch,
                'labels': batch
            })
        return test_set

    # Prepare For LoRA
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Load Train Dataset
    data = load_dataset(args.data_path)
    alpaca_data = load_dataset("yahma/alpaca-cleaned")
    math = load_dataset("microsoft/orca-math-word-problems-200k")
    # transl = load_dataset("google/wmt24pp", "en-zh_CN", split="train")
    sciqa_train, sciqa_eval = load_dataset("allenai/sciq", split=['train', 'validation'])
    mintaka_train, mintaka_eval = load_dataset("AmazonScience/mintaka", "en", split=['train', 'validation'])
    
    samples = [
        {"question": "Identify the language of the given sentence.\nIl est incroyable.", "answer": "The language used in the given sentence is French."},
        {"question": "Summarize the given passage.\nRecent study showed that global climate change is one of the most important challenges facing the world today. The consequences of global warming include rising sea levels, extreme weather events and biodiversity loss. Climate change is caused by human activities such as burning fossil fuels, and if unchecked will have long-term and devastating impacts on our planet.", "answer": "- Global climate change is an important issue.\n- It affects the rise of sea levels, extreme weather events, and biodiversity loss.\n- Human activities, such as burning fossil fuels, contribute to this problem.\n- If not addressed, it could cause long-term and devastating impacts on the planet."},
        {"question": "Find the solution to the following quadratic equation.\n3x² + 7x - 4 = 0", "answer": "To solve the quadratic equation \(3x^2 + 7x - 4 = 0\), we can use the quadratic formula, which is given by:\n\n\[\nx = \\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}\n\]\n\nFor the given quadratic equation \(3x^2 + 7x - 4 = 0\), the coefficients are:\n- \(a = 3\)\n- \(b = 7\)\n- \(c = -4\)\n\nFirst, we calculate the discriminant (\(\Delta\)):\n\n\[\n\Delta = b^2 - 4ac\n\]\n\nSubstituting the values of \(a\), \(b\), and \(c\):\n\n\[\n\Delta = (7)^2 - 4(3)(-4) = 49 + 48 = 97\n\]\n\nNext, we substitute \(a\), \(b\), and \(\Delta\) into the quadratic formula:\n\n\[\nx = \\frac{-7 \pm \sqrt{97}}{2 \cdot 3} = \frac{-7 \pm \sqrt{97}}{6}\n\]\n\nThus, the solutions to the quadratic equation are:\n\n\[\nx_1 = \frac{-7 + \sqrt{97}}{6}\n\]\n\[\nx_2 = \\frac{-7 - \sqrt{97}}{6}\n\]\n\nSo the solutions to the quadratic equation \(3x^2 + 7x - 4 = 0\) are:\n\n\[\nx = \frac{-7 + \sqrt{97}}{6} \quad \text{and} \quad x = \\frac{-7 - \sqrt{97}}{6}\n\]"},
        {"question": "Rewrite the following sentence using a different but similar word.\nShe partook in the event.", "answer": "She participated in the event."},
        {"question": "Rewrite the following sentence with more descriptive words.\nThe game is fun.", "answer": "The engaging and thrilling game is a delightful experience for all players."},
        {"question": "Describe the sound of this object.\nWind chime", "answer": "The sound of a wind chime is a melodious and soothing noise that can be heard when the wind blows through it. The metal or bamboo clappers make a gentle tinkling sound that creates a soft and calming atmosphere. The sound of the wind chime is often associated with nature and can evoke feelings of serenity and peace."},
        {"question": "Explain to me the concept of cogging torque.\n", "answer": "工夫扭矩 (Cogging Torque) is not a standard term in mechanical engineering or electrical engineering, but it can be understood as a phenomenon related to magnetic fields and rotor structures in electric machines, such as synchronous motors or generators. The term might be used loosely to describe a torque effect that arises due to the interaction between the rotor and stator components in these machines.\n\nTo clarify, let's break down what could be meant by \"Cogging Torque\":\n\n### 1. **Cogging Effect**:\nThe term \"Cogging\" typically refers to an undesirable pulsating torque that occurs in the operation of certain types of synchronous machines, particularly those with salient pole rotors (rotors with distinct poles). This effect is more accurately described as **Cogging Torque**.\n\n### 2. **Explanation of Cogging Torque**:\n- **Mechanical Interference**: In salient pole rotors, the rotor poles interact with the stator windings in a way that creates periodic forces and torques as the rotor rotates.\n- **Pulsa..."},
        {"question": "Explain the process of cellular respiration in the plants.", "answer": "Cellular respiration is the process by which cells convert glucose into energy and release it back into the environment. This process occurs in all living organisms, including both plants and animals.\n\nThe process of cellular respiration in plants involves several steps:\n\n1. Glycolysis: This is the first step in cellular respiration. During glycolysis, a cell breaks down glucose molecules into two pyruvate molecules, releasing energy in the form of ATP (adenosine triphosphate).\n\n2. The Krebs Cycle: The Krebs Cycle is the next stage of cellular respiration. It involves the transfer of electrons from the NADH+ to the citric acid, and then to the electron transport chain. The final stage of cellular respiration is the production of ATP by the electron transport chain.\n\n3. Oxidative Phosphorylation: In this stage, the cell takes in oxygen from the air through its mitochondria and converts it into water. This process occurs in the chloroplasts of plant cells and is essential for survival in most environments.\n\n4. The Electron Transport Chain: In the final stage of cellular respiration, the cell transfers electrons from the NADH+ to the coenzyme NADPH (nicotinamide adenine dinucleotide phosphate). These electrons are used to power the reaction that produces ATP.\n\nOverall, the process of cellular respiration in plants is a complex series of biochemical reactions that convert glucose into energy, allowing them to carry out their normal functions while also producing oxygen and other byproducts such as ATP.\n\nIn summary, cellular respiration is an essential process that enables all forms of life on earth to survive and thrive."},
        {"question": "Suppose Variable x is defined as “4x + 2y = 10”. Find the value of x.\n", "answer": "Whereas the equation \(4x + 2y = 10\) is given, to find the value of \(x\), we need an additional piece of information about \(y\). Without knowing the specific value or relationship for \(y\), it is impossible to determine a unique value for \(x\).\n\nIf you have more details about \(y\), please provide them so I can assist you further."},
        {"question": "Translate the following english into French.\nI miss you.", "answer": "The translation of \"I miss you.\" into French is \"Tu me manques.\""},
        {"question": "Give me the synonyms of the following word: 'Tenacious'.\n", "answer": "The word 'tenacious' can have several synonyms:\n\n1. Resolute: Someone who is determined or persistent in their efforts to achieve their goals.\n2. Firm: Someone who is steadfast and resolute in their beliefs and actions.\n3. Determined: A person who is undaunted by obstacles or difficulties in pursuit of their goals.\n4. Persistent: Someone who continues to work hard towards achieving their goals even when faced with challenges.\n5. Tenacious: Someone who is very determined in their pursuits, often to the point of exhaustion.\n6. Enduring: Someone who perseveres through adversity and remains steadfast in their beliefs and actions."},
        {"question": "Classify the following data with one of the three following labels. 1. Alphabetical Data\n2. Numeric Data\n3. Alpha-numeric data\nfjsklfjdsklfjsklfjsklfjs", "answer": "1. Alphabetical Data"},
        {"question": "Rewrite the third person version of the the following sentence?\n I am anxious.", "answer": "The third person version of this sentence should be:\n\nHe/She is anxious."},
        {"question": "Rewrite the third person version of the the following sentence?\n I am anxious.", "answer": "The third person version of this sentence should be:\n\nHe/She is anxious."},
        {"question": "Rewrite the third person version of the the following sentence?\n I am anxious.", "answer": "The third person version of this sentence should be:\n\nHe/She is anxious."},
        {"question": "Describe the pros and cons of the policy\nProviding free college education", "answer": " \nPros: \n1. Cost-effective: Free college education is a significant benefit to students who can't afford to pay for tuition fees. This reduces the financial burden on students and helps them achieve their educational goals.\n\n2. Increases access to higher education: With free college education, more students have the opportunity to pursue higher education without incurring high costs, which can lead to increased opportunities for employment and career advancement.\n\n3. Promotes social mobility: Providing free college education can help reduce income inequality by providing an equal opportunity for all students to succeed regardless of their socioeconomic status.\n\n4. Reduces student debt: Free college education eliminates the need for students to take out loans or pay off debts, which can result in reduced student debt and financial stress.\n\n5. Encourages lifelong learning: A free education can provide students with the knowledge and skills they need to excel in their chosen field, which can enhance their overall academic performance.\n\nCons:\n1. Financial strain: While providing free college education can be cost-effective, it can also place a significant financial strain on the institutions that offer such programs. These institutions may not have the resources to cover the cost of providing free education, leading to potential financial difficulties.\n\n2. Reduced competition: With free college education, there may be fewer students pursuing higher levels of education, which could result in a reduction of competition from other schools and colleges.\n\n3. Inequitable distribution of resources: Free college education does not necessarily guarantee equal distribution of resources among students, as some students may have better access to resources than others due to financial constraints. This can lead to disparities in academic achievement and opportunities for those with less privilege.\n\n4. Limited diversity: Free college education does not necessarily guarantee diverse representation in the student body, as some students may choose to attend schools with more conservative or traditional values.\n\nOverall, while free college education can provide numerous benefits, it is important to consider its limitations and potential drawbacks, and to ensure that it is accessible and equitable to all students."},
    ] * 100
    sample_dataset = Dataset.from_list(samples)
    
    if args.cache_dataset and os.path.exists('datasets/cache/{}.bin'.format(args.data_path)):
        preprocess_data = torch.load('datasets/cache/{}.bin'.format(args.data_path))
        train_data, val_data = preprocess_data['train'], preprocess_data['val']
    else:
        data = concatenate_datasets([data["train"], alpaca_data["train"].select(range(200, 51760)), math["train"]]) #  
        
        train_val = data.train_test_split(
            test_size=args.val_set_size, shuffle=True, seed=42
        )
        train_data = concatenate_datasets([train_val["train"], sample_dataset, sciqa_train, mintaka_train])
        train_data = (
            train_data.shuffle().map(generate_and_tokenize_prompt)
        )
        
        val_data = {
            args.data_path: concatenate_datasets([train_val["test"], sciqa_eval, mintaka_eval]).shuffle().map(generate_and_tokenize_prompt),
        }


        if args.cache_dataset and args.local_rank == 0:
            cache_file = 'datasets/cache/{}.bin'.format(args.data_path)
            cache_dir = '/'.join(cache_file.split('/')[:-1])
            directory = Path(cache_dir)
            directory.mkdir(parents=True, exist_ok=True)

            torch.save({
                'train': train_data, 'val': val_data
            }, cache_file)

    # Load Extra Validation Dataset
    if args.extra_val_dataset:
        from LLMPruner.datasets.ppl_dataset import get_wikitext2, get_ptb

        seq_len = 128
        for extra_dataset in args.extra_val_dataset.split(','):
            if 'wikitext2' in extra_dataset:
                _, test_data = get_wikitext2(seq_len, None)
                test_data = split_and_tokenizer(test_data, tokenizer, seq_len, field_name='text')
            if 'ptb' in extra_dataset:
                _, test_data = get_ptb(seq_len, None)
                test_data = split_and_tokenizer(test_data, tokenizer, seq_len, field_name='sentence')
            val_data[extra_dataset] = test_data

    callback = DecodeAndPrintCallback(tokenizer, 
                                      prompts=[
                                       "Identify the language used in the given sentence.\nIl est incroyable.",
                                        "Summarize the given passage.\nA recent study showed that global climate change is one of the most important challenges facing the world today. The consequences of global warming include rising sea levels, extreme weather events and biodiversity loss. Climate change is caused by human activities such as burning fossil fuels, and if unchecked will have long-term and devastating impacts on the planet.",
                                        "Find the solution to the quadratic equation.\n3x² + 7x - 4 = 0",
                                        "Rewrite the given sentence using a different but similar word.\nShe partook in the event.",
                                        "Rewrite the sentence with more descriptive words.\nThe game is fun.",
                                        "Describe the sound of the given object.\nWind chime",
                                        "Explain the concept of cogging torque.",
                                        "Explain the process of cellular respiration in the plants.",
                                        "Variable x is defined as “4x + 2y = 10”. Find the value of x.",
                                        "Translate the following phrase into French.\nI miss you",
                                        "Find the synonyms of the following word: 'Tenacious'.",
                                        "Classify the following data with one of the three following labels. 1. Alphabetical Data\n2. Numeric Data\n3. Alpha-numeric data\nfjsklfjdsklfjsklfjsklfjs",
                                        "Rewrite the following sentence in the third person.\nI am anxious",
                                        "Describe the pros and cons of the following policy\nProviding free college education",
                                        "Replace the underlined words with appropriate synonyms.\nThe robbers snitched on their partners."
                                      ],
                                      max_length=512)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=10,
            logging_first_step=True,
            optim="adamw_torch",
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            save_steps=200,
            output_dir=args.output_dir,
            save_total_limit=20,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=None,
            group_by_length=args.group_by_length,
            report_to="wandb",
            run_name=args.output_dir.split('/')[-1],
            metric_for_best_model="{}_loss".format(args.data_path),
        ),
        callbacks=[callback],
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    model.state_dict = old_state_dict
    model.save_pretrained(args.output_dir)

    from LLMPruner.evaluator.ppl import PPLMetric
    ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device="cuda")
    print("PPL after pruning: {}".format(ppl))
    print("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    # Model Type&Path
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--prune_model', type=str, help='prune model name')
    parser.add_argument('--data_path', type=str, default='allenai/c4', help='data path')
    parser.add_argument('--cache_dataset', action="store_true", default=False)
    parser.add_argument('--extra_val_dataset', type=str, default=None, help='validation datasets. Split with ","')
    parser.add_argument('--output_dir', type=str, default="./lora-alpaca", help='output directory')

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
