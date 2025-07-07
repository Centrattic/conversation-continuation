import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset, DatasetDict
from typing import List
from tqdm import tqdm
from config import FRIEND_NAME, MODEL_NAME, RESULTS_FOLDER, bnb_config

base_model_name = f"{MODEL_NAME}"
adapter_path = f"./{RESULTS_FOLDER}"
test_file = "test.json"
max_new_tokens = 50
n_samples = 10  # How many examples to evaluate

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Loading models")
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", bnb_config=bnb_config)
lora_model = PeftModel.from_pretrained(base_model, adapter_path)
lora_model.eval()

test_data = DatasetDict({"test": load_dataset("json", data_files="test.json", split="train")})["test"]

@torch.no_grad()
def generate(model, prompt:str, tokenizer, max_new_tokens=max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7, # play with this!
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

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
