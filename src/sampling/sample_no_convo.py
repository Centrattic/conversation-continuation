import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset, DatasetDict
from typing import List
from tqdm import tqdm
from pathlib import Path

from src.config import RIYA_NAME, FRIEND_NAME, MODEL_NAME, RESULTS_FOLDER, bnb_config
from src.model_utils import generate

base_model_name = Path(f"{MODEL_NAME}")
# adapter_path = Path(f"./{RESULTS_FOLDER}/lora_adapter")
max_new_tokens = 200

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Loading models")
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto",quantization_config=bnb_config)
# lora_model = PeftModel.from_pretrained(base_model, adapter_path)
base_model.eval()
# Inter

history = []
hist_count = 0 # up to 8 since thats curr length

print(f"Start your conversation with the base model.")

while(1):
    
    if hist_count > 8: # maybe would be good on > 8 still? hmm maybe should set higher for training
        history.pop(0)

    prompt = input()

    history.append(prompt)
    hist_count +=1

    base_out = generate(base_model, "".join(history), tokenizer)

    base_out = base_out.replace("<s>", "").strip()
    history.append(base_out)
    hist_count += 1

    print(f"Assistant: {base_out}") 
