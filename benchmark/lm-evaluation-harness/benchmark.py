from modeling_llamagear import LlamaForCausalLM_GEARKIVI

from modeling_llama_kivi import LlamaForCausalLM_KIVI
from modeling_llama_pre_compress import LlamaForCausalLM_pre_comp
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
from transformers import BitsAndBytesConfig
from datasets import load_dataset
import torch
import argparse
from typing import List

def get_config():
    config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

    config.k_bits = 4# current support 2/4 bit for KV Cache
    config.v_bits = 4 # current support 2/4 bit for KV Cache
    config.group_size = 32
    config.residual_length = 32 # the number of recent fp16 tokens
    return config

def get_compress_config():
    compress_config = {}
    compress_config["compress_method"] = "gearlKIVI" # "gearlKIVI" "gearsKIVI"
    compress_config["group_size"] = 32
    compress_config["residual"] = 32
    compress_config["quantize_bit"] = 4
    compress_config["rank"] = 2 ## prefill rank
    compress_config["rankv"] = 2 ## prefill rank
    compress_config["loop"] = 3
    return compress_config

def get_model_GEAR(device = "cuda"):
    config = get_config()
    compress_config = get_compress_config()
    model = LlamaForCausalLM_GEARKIVI.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        config = config,
        # quantization_config = quantization_config,
        compress_config = compress_config,
        torch_dtype=torch.float16, # FP16 으로 불러오도록 추가됨.
        device_map = device
    )
    return model

def get_model_KIVI(device = "cuda"):
    config = get_config()
    model = LlamaForCausalLM_KIVI.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        config = config,
        # quantization_config = quantization_config,
        # compress_config = compress_config,
        torch_dtype=torch.float16, # FP16 으로 불러오도록 추가됨.
        
        device_map = device,
    )
    return model

def get_Llama(device = "cuda"):
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.float16, # FP16 으로 불러오도록 추가됨.
        device_map = device)
    return model

def get_model_pre_compress(device = "cuda"):
    config = get_config()
    model = LlamaForCausalLM_pre_comp.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        config = config,
        torch_dtype=torch.float16, # FP16 으로 불러오도록 추가됨.
        device_map = device
    )
    return model



def test_occupy_resource():
    #### Config for KIVI model
    # import os
    # my_token = os.environ.get("HUGGINGFACE_API_TOKEN")
    # config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf", token=my_token)

    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    parser = argparse.ArgumentParser(description="Evaluate AQuA Tasks")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    # parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b", help="Model name or path.")
    parser.add_argument("--model", type=str, default="None", help="Model name or path.")
    parser.add_argument("--device", type=str, default="cuda", help="Device name or path.")
    args = parser.parse_args()
    DEVICE = args.device

    max_token = 500 ### prefill_length
    max_generation_length = 1000 ### geneate 500
    batch_size = args.batch_size

    # args.model = "pre_compress"

    if "gearl" in args.model:
        model = get_model_GEAR(DEVICE)
    elif "KIVI" in args.model:
        model = get_model_KIVI(DEVICE)
    elif "None" in args.model:
        model = get_Llama(DEVICE)
    elif "pre_compress" in args.model:
        model = get_model_pre_compress(DEVICE)
    else:
        print("args model is not supported. args.model: ", args.model) # 인자가 무조건 필요함.
        exit(1)

    model = model.half() # it is model.to(torch.float16)

    peak_memory = torch.cuda.max_memory_allocated(device="cuda") / (1024**2)
    print(f"Peak memory usage on GPU for model loading: {peak_memory} MB")


    tokenizer = AutoTokenizer.from_pretrained(
        'meta-llama/Llama-2-7b-hf', 
        model_max_length=max_token,
        max_length=max_token,
        use_fast=False, 
        trust_remote_code=True, 
        tokenizer_type='llama')
    tokenizer.pad_token = tokenizer.eos_token
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text_combined = test["text"]

    sentence_group = []
    for i in range(batch_size):
        # sentence_group.append(str(text_combined[i*max_token:(i+1)*max_token]))
        sentence_group.append(str(text_combined[0:max_token]))
    inputs = tokenizer(
        sentence_group,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    print("begin")
    inputs = inputs.to("cuda:0")

    print(f"input_ids has nan: {torch.isnan(inputs.input_ids).any()}")

    print(inputs.input_ids.shape)

    print()
    import time

    start = time.time()
    result = model.generate(**inputs, max_new_tokens=256, use_cache=True)
    torch.cuda.synchronize()
    end = time.time()
    peak_memory = torch.cuda.max_memory_allocated(device="cuda") / (1024**2)  # 转换为MB单位

    print(f"Peak memory usage on GPU: {peak_memory} MB")
    print("time",end - start, 's')

def test_accuracy():
    #### Config for KIVI model
    # import os
    # my_token = os.environ.get("HUGGINGFACE_API_TOKEN")
    # config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf", token=my_token)

    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    parser = argparse.ArgumentParser(description="Evaluate AQuA Tasks")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    # parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b", help="Model name or path.")
    parser.add_argument("--model", type=str, default="None", help="Model name or path.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device name or path.")
    parser.add_argument(
        "--tasks", 
        type=str, 
        # default="arc_easy,arc_challenge",
        required=True, 
        help="Tasks to evaluate on.")
    args = parser.parse_args()
    DEVICE = args.device

    if "gearl" in args.model:
        model = get_model_GEAR(DEVICE)
    elif "KIVI" in args.model:
        model = get_model_KIVI(DEVICE)
    elif "None" in args.model:
        model = get_Llama(DEVICE)
    elif "pre_compress" in args.model:
        model = get_model_pre_compress(DEVICE)
    else:
        print("args model is not supported. args.model: ", args.model) # 인자가 무조건 필요함.
        exit(1)

    model_args = {
        "pretrained" : model,
        "trust_remote_code" : True,
    }

    gen_kwargs = None #"max_gen_toks=256,do_sample=False,temperature=0.0,top_p=0.0"

    args = argparse.Namespace(
        model='hf',
        tasks=args.tasks,
        model_args=model_args,
        num_fewshot=None,
        batch_size=args.batch_size,
        max_batch_size=None,
        device=DEVICE,
        output_path=None,
        limit=None,
        use_cache=None,
        cache_requests=None,
        check_integrity=False,
        write_out=False,
        log_samples=False,
        system_instruction=None,
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        show_config=False,
        include_path=None,
        gen_kwargs=gen_kwargs,
        verbosity="DEBUG",
        wandb_args="",
        hf_hub_log_args="",
        predict_only=False,
        seed=[0, 1234, 1234, 1234],
        trust_remote_code=False,        
    )

    from lm_eval.__main__ import cli_evaluate
    cli_evaluate(args)
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated(device=DEVICE) / (1024**2) 
    print(f"Peak memory usage on GPU: {peak_memory} MB")


if __name__ == "__main__":
    # import warnings
    # warnings.filterwarnings("ignore", message="`do_sample` is set to `False`.*")
    test_accuracy()
    # test_occupy_resource()
