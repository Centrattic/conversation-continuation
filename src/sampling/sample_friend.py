# This file will enable continuous sampling to talk to Friend only, filtering out [RIYA] tags

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset, DatasetDict
from typing import List
from tqdm import tqdm
from src.config import FRIEND_NAME, MODEL_NAME, RESULTS_FOLDER, bnb_config
from src.model_utils import generate

base_model_name = f"{MODEL_NAME}"
adapter_path = f"./{RESULTS_FOLDER}/lora_adapter"
max_new_tokens = 90

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Loading models")
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto",quantization_config=bnb_config)
lora_model = PeftModel.from_pretrained(base_model, adapter_path)
lora_model.eval()

history = []
hist_count = 0 # up to 8 since thats curr length

print("Start your conversation")

while(1):
    
    # ToDo: make history a stack
    if hist_count > 8: # maybe would be good on > 8 still? hmm maybe should set higher for training
        history.pop(0)

    prompt = input()

    prompt_friend = f"\n[Riya]: {prompt} \n [{FRIEND_NAME}]"
    history.append(prompt_friend)
    hist_count +=1

    lora_out = generate(lora_model, "".join(history), tokenizer)

    index = lora_out.find("[Riya]") # oh but sometimes outputs [R token and not [RIYA] both tokens (or more than 1?), since that is the split
    if index == -1:
        index = lora_out.find("[R")
    lora_out = lora_out[:index]
    lora_out = lora_out.replace("<s>", "").strip()
    history.append(lora_out) # lora_out shouldn't have friend name
    hist_count += 1

    print(f"[{FRIEND_NAME}]: {lora_out}") 

    # why the <s>? appears?