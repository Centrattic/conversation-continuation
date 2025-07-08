# This file will enable continus sampling to talk to Friend only, filtering out [RIYA] tags
# Add more nuanced sampling (topk?)

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import FRIEND_NAME, MODEL_NAME, RESULTS_FOLDER, bnb_config
from model_utils import generate

base_model_name = f"{MODEL_NAME}"
adapter_path = f"./{RESULTS_FOLDER}/lora_adapter"
max_new_tokens = 40
save_path = "looped_convo.txt"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Loading models")
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto",  quantization_config=bnb_config)
lora_model = PeftModel.from_pretrained(base_model, adapter_path)
lora_model.eval()

riya_out = f"hey {FRIEND_NAME}, let's write a poem!"
history = [] # conversation starter
hist_count = 0 # up to 8 since thats curr length
full_history = []

print("Starting your conversation")

start_text = f"[Riya]: {riya_out}"
print(start_text)

with(open(save_path, 'w')) as f:
    f.write("")
    full_history.append(start_text)
    f.write("".join(full_history))

while(1):
    # could just output a big sample or whatever, but talking back and forth probably better I think? ik if huge difference
    # why the <s> appearing

    if hist_count > 8:
        history.pop(0)

    prompt_friend = f"\n[Riya]: {riya_out} \n [{FRIEND_NAME}]"
    history.append(prompt_friend)
    hist_count +=1

    friend_out = generate(lora_model, "".join(history), tokenizer) # \n-s already added

    index_friend = friend_out.find("[Riya]") # oh but sometimes outputs [R token and not [RIYA] both tokens (or more than 1?), since that is the split
    if index_friend == -1:
        index_friend = friend_out.find("[R")
    friend_out = friend_out[:index_friend]
    friend_out = friend_out.replace("<s>", "").strip()
    
    friend_text = f"[{FRIEND_NAME}]: {friend_out}"
    print(friend_text)
    full_history.extend(["\n", friend_text])

    prompt_riya = f"{friend_out} \n [Riya]"
    history.append(prompt_riya) # lora_out shouldn't have friend name
    hist_count +=1

    riya_out = generate(lora_model, "".join(history), tokenizer)
    riya_out = riya_out.replace("<s>", "").strip()

    index_riya = riya_out.find("[{FRIEND_NAME[0]}") # assuming friend token split [first letter for now, should have more nuance later
    riya_out = riya_out[:index_riya]
    
    riya_text = f"[Riya]: {riya_out}"
    print(riya_text)
    full_history.extend(["\n", riya_text])

    with(open(save_path, 'w')) as f:
        # full_history.append("\n")
        f.write("".join(full_history))

