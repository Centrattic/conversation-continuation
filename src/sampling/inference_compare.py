import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset, DatasetDict
from typing import List
from tqdm import tqdm
from pathlib import Path

from src.config import FRIEND_NAME, MODEL_NAME, RESULTS_FOLDER, bnb_config
from src.model_utils import generate

base_model_name = Path(f"{MODEL_NAME}")
adapter_path = Path(f"./{RESULTS_FOLDER}/lora_adapter")test_file = "test.json"
max_new_tokens = 50
n_samples = 10  # How many examples to evaluate

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Loading models")
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", quantization_config=bnb_config)
lora_model = PeftModel.from_pretrained(base_model, adapter_path)
lora_model.eval()

test_data = DatasetDict({"test": load_dataset("json", data_files="test.json", split="train")})["test"]

print(f"\nComparing on {n_samples} samples:\n")
# eventually save to file

for i in tqdm(range(min(n_samples, len(test_data)))):
    sample = test_data[i]
    prompt = sample["prompt"].strip()
    ground_truth = sample["response"].strip()

    base_out = generate(base_model, prompt, tokenizer)
    lora_out = generate(lora_model, prompt, tokenizer)

    print(f"\n--- Sample {i+1} ---")
    print(f"[PROMPT]:\n{prompt}")
    print(f"[GROUND TRUTH]:\n{ground_truth}")
    print(f"[BASE MODEL]:\n{base_out}")
    print(f"[LoRA MODEL]:\n{lora_out}")
    print("-" * 50)
