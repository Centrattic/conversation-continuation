# This file will enable continuous sampling to talk to Friend only, filtering out [RIYA] tags

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List
from tqdm import tqdm
from pathlib import Path
import argparse
from datetime import datetime

from src.config import FRIEND_ID, RIYA_NAME, FRIEND_NAME, MODEL_NAME, RESULTS_FOLDER, CONVO_FOLDER, bnb_config
from src.model_utils import generate, generate_with_activations
from src.logger import ConversationLogger
import json
from src.activation_tda.tda_utils import find_topk_train_samples, FinalLayerActivationCache

parser = argparse.ArgumentParser()
parser.add_argument('--tda', action='store_true')

args = parser.parse_args()
enable_tda = args.tda

curr_date = datetime.now().strftime("%m-%d-%y-%H-%M-%S")
log_path = f"{CONVO_FOLDER}/sample-friend/{curr_date}.txt"
logger = ConversationLogger(Path(log_path))

base_model_name = Path(f"{MODEL_NAME}")
adapter_path = Path(f"./{RESULTS_FOLDER}/lora_adapter")
max_new_tokens = 90

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

logger.log_to_all("Loading models")
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto",quantization_config=bnb_config)
lora_model = PeftModel.from_pretrained(base_model, adapter_path)
lora_model.eval()

history = []
hist_count = 0 # up to 8 since thats curr length

logger.log_to_all("Start your conversation")

while(1):
    
    # ToDo: make history a stack
    if hist_count > 8: # maybe would be good on > 8 still? hmm maybe should set higher for training
        history.pop(0)

    prompt = input()

    prompt_friend = f"\n[{RIYA_NAME}]: {prompt} \n [{FRIEND_NAME}]"
    history.append(prompt_friend)
    hist_count +=1

    logger.log_to_file(f"\n[{RIYA_NAME}]: {prompt} \n")

    if enable_tda:
        lora_out, acts = generate_with_activations(lora_model, "".join(history), tokenizer)
    else:
        lora_out = generate(lora_model, "".join(history), tokenizer)

    index = lora_out.find(f"[{RIYA_NAME}]") # oh but sometimes outputs [R token and not [RIYA] both tokens (or more than 1?), since that is the split
    if index == -1:
        index = lora_out.find("[{RIYA_NAME[0]}")
    lora_out = lora_out[:index]
    lora_out = lora_out.replace("<s>", "").strip()
    history.append(lora_out) # lora_out shouldn't have friend name
    hist_count += 1

    logger.log_to_all(f"[{FRIEND_NAME}]: {lora_out}")

    if enable_tda:
        sel_acts = acts[1:index] # a list
        gen_acts = torch.stack([a[0, -1, :] for a in sel_acts], dim=0) # (num_gen_tokens, hidden_size)
        mean_gen_acts = gen_acts.mean(dim=0) # mean activation vector

        mistral_cache_info = f"{RESULTS_FOLDER}/activation_cache/final.meta.json"
        d = json.load(open(mistral_cache_info))
        hidden_size, max_seq_len = d['hidden_size'], d['max_seq_len']
        
        # ToDo: maybe refactor so I don't have to pass in an ActivationCache instance
        mistral_cache = FinalLayerActivationCache(hidden_size, max_seq_len)
        top_train_samples, _ = find_topk_train_samples(mistral_cache, mean_gen_acts, k=1, author_id = str(FRIEND_ID))
        # could use RiyaID to do TDA for the prompt (to figure out when before have i asked similar quesitons?)

        for i, entry in enumerate(top_train_samples):
            # entry already has “Author MM-DD-YY-HH-MM-SS: content”
            logger.log_to_all(f"{i}. {entry}", color='blue')