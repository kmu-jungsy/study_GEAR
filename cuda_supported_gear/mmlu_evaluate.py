import argparse
import torch
import os
import numpy as np
import pandas as pd
from modeling_llamagear import LlamaForCausalLM_GEARKIVI
from transformers import LlamaConfig, AutoTokenizer

from crop import crop

choices = ["A", "B", "C", "D"]

# Function to calculate softmax
def softmax(x):
    z = x - np.max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator / denominator

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = f"The following are multiple choice questions (with answers) about {subject}.\n\n"
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def eval_model(args, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]

    for i in range(test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while True:
            cropped_prompt = crop(prompt, tokenizer)
            if cropped_prompt == prompt or k <= 0:
                break
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000).to("cuda:0")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        log_probs = torch.nn.functional.log_softmax(logits[:, -1, :], dim=-1)
        lprobs = [log_probs[:, tokenizer.encode(choice)[-1]].item() for choice in answers]

        pred = choices[np.argmax(lprobs)]
        probs = softmax(np.array(lprobs))

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)
    all_probs = np.array(all_probs)
    print(f"Average accuracy {acc:.3f} - {subject}")
    return cors, acc, all_probs

def main(args):
    config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

    config.k_bits = 2# current support 2/4 bit for KV Cache
    config.v_bits = 2 # current support 2/4 bit for KV Cache
    config.group_size = 64
    config.residual_length = 64 # the number of recent fp16 tokens

    compress_config = {
        "compress_method": "gearlKIVI",
        "group_size": 64,
        "residual": 64,
        "quantize_bit": 2,
        "rank": 2,
        "rankv": 2,
        "loop": 3
    }

    model = LlamaForCausalLM_GEARKIVI.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        config=config,
        compress_config=compress_config,
        torch_dtype=torch.float16,
        device_map="cuda:0"
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        'meta-llama/Llama-2-7b-hf', 
        model_max_length=1000,
        use_fast=False, 
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    total_correct = 0
    total_questions = 0

    for subject in subjects:
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

        cors, acc, probs = eval_model(args, subject, model, tokenizer, dev_df, test_df)

        total_correct += np.sum(cors)
        total_questions += len(cors)

        test_df["{}_correct".format(args.engine)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.engine, choice)] = probs[:, j]
        test_df.to_csv(os.path.join(args.save_dir, f"{subject}.csv"), index=None)

    overall_accuracy = total_correct / total_questions
    print(f"Overall accuracy: {overall_accuracy * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--engine", "-e", type=str, default="llamagear")
    args = parser.parse_args()
    main(args)
