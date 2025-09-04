# This file will enable continus sampling to talk to Friend only, filtering out [RIYA] tags
# Add more nuanced sampling (topk?)

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
from datetime import datetime
from pathlib import Path

from src.config import FRIEND_NAME, RIYA_NAME, MODEL_NAME, RESULTS_FOLDER, CONVO_FOLDER, bnb_config
from src.model_utils import generate
from src.logger import ConversationLogger
from src.data_utils import clean_for_sampling

parser = argparse.ArgumentParser()
parser.add_argument('--save', action='store_true')

args = parser.parse_args()
save_convo = args.save

if save_convo:
    curr_date = datetime.now().strftime("%m-%d-%y-%H-%M-%S")
    log_path = f"{CONVO_FOLDER}/looped/{curr_date}.txt"
    logger = ConversationLogger(Path(log_path))
else:
    logger = ConversationLogger()  # logger outputs just print to console

base_model_name = Path(f"{MODEL_NAME}")
adapter_path = Path(f"./{RESULTS_FOLDER}/lora_train/lora_adapter")
max_new_tokens = 40

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
tokenizer.pad_token = tokenizer.eos_token

logger.log_to_all("Loading models")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name, device_map="auto", quantization_config=bnb_config)

vocab_size = len(tokenizer)
if base_model.get_input_embeddings().num_embeddings != vocab_size:
    logger.log_to_all(
        f"Resizing model vocabulary from {base_model.get_input_embeddings().num_embeddings} to {vocab_size}"
    )
    base_model.resize_token_embeddings(vocab_size, mean_resizing=False)

lora_model = PeftModel.from_pretrained(base_model, adapter_path)
lora_model.eval()

riya_out = f"{FRIEND_NAME} you're awesome!"
history = []  # conversation starter
hist_count = 0  # up to 8 since thats curr length
full_history = []

print("Starting your conversation")

start_text = f"[{RIYA_NAME}]: {riya_out}"
print(start_text)

while (1):
    # could just output a big sample or whatever, but talking back and forth probably better I think? ik if huge difference
    # why the <s> appearing

    if hist_count > 8:
        history.pop(0)

    prompt_friend = f"\n[{RIYA_NAME}]: {riya_out} \n [{FRIEND_NAME}]"
    history.append(prompt_friend)
    hist_count += 1

    friend_out = generate(lora_model, "".join(history),
                          tokenizer)  # \n-s already added

    index_friend = friend_out.find(
        f"[{RIYA_NAME}]"
    )  # oh but sometimes outputs [R token and not [RIYA] both tokens (or more than 1?), since that is the split
    if index_friend == -1:
        index_friend = friend_out.find("[R")
    friend_out = friend_out[:index_friend]
    friend_out = friend_out.replace("<s>", "").strip()

    friend_text = f"[{FRIEND_NAME}]: {friend_out}"
    logger.log_to_all(friend_text)
    full_history.extend(["\n", friend_text])

    prompt_riya = f"{friend_out} \n [{RIYA_NAME}]"
    history.append(prompt_riya)  # lora_out shouldn't have friend name
    hist_count += 1

    riya_out = generate(lora_model, "".join(history), tokenizer)
    riya_out = riya_out.replace("<s>", "").strip()

    index_riya = riya_out.find(
        f"[{FRIEND_NAME[0]}"
    )  # assuming friend token split [first letter for now, should have more nuance later
    riya_out = riya_out[:index_riya]

    riya_text = f"[{RIYA_NAME}]: {riya_out}"
    logger.log_to_all(riya_text)
    full_history.extend(["\n", riya_text])
