import os
import gc
import torch
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
from transformers import BitsAndBytesConfig
from datasets import load_dataset
import argparse
from modeling_llamagear import LlamaForCausalLM_GEARKIVI
from modeling_llama_kivi import LlamaForCausalLM_KIVI

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

config.k_bits = 2  # current support 2/4 bit for KV Cache
config.v_bits = 2  # current support 2/4 bit for KV Cache
config.group_size = 64
config.residual_length = 64  # the number of recent fp16 tokens

parser = argparse.ArgumentParser(description="Evaluate AQuA Tasks")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")  # 배치 크기 줄임
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b", help="Model name or path.")
args = parser.parse_args()

max_token = 512  # 입력 데이터 크기 줄임
max_generation_length = 1024  # 생성 길이 줄임
batch_size = args.batch_size

compress_config = {
    "compress_method": "gearlKIVI",
    "group_size": 64,
    "residual": 64,
    "quantize_bit": 2,
    "rank": 2,
    "rankv": 2,
    "loop": 3
}

args.model = "gearl"

if "gearl" in args.model:
    model = LlamaForCausalLM_GEARKIVI.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        config=config,
        quantization_config=quantization_config,
        compress_config=compress_config,
        device_map="cuda:0"
    )
elif "KIVI" in args.model:
    model = LlamaForCausalLM_KIVI.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        config=config,
        quantization_config=quantization_config,
        device_map="cuda:0"
    )
elif "None" in args.model:
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        quantization_config=quantization_config,
        device_map="cuda:0"
    )

tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    model_max_length=max_token,
    max_length=max_token,
    use_fast=False,
    trust_remote_code=True,
    tokenizer_type='llama'
)
tokenizer.pad_token = tokenizer.eos_token
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
text_combined = test["text"]

sentence_group = []
for i in range(batch_size):
    sentence_group.append(str(text_combined[0:max_token]))

inputs = tokenizer(
    sentence_group,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
)

print("begin")
inputs = inputs.to("cuda:0")
print(inputs.input_ids.shape)

gc.collect()
torch.cuda.empty_cache()

import time

start = time.time()
result = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=max_generation_length, use_cache=True)
torch.cuda.synchronize()
end = time.time()

peak_memory = torch.cuda.max_memory_allocated(device="cuda") / (1024**2)  # MB 단위로 변환
print(f"Peak memory usage on GPU: {peak_memory} MB")
print("time", end - start)

gc.collect()
torch.cuda.empty_cache()
