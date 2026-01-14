import json
from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
from src.config import MODEL_NAME
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

data = pd.read_csv("./data/finetune_7_6_25/friend_hist.csv")

data = data[["Author", "Content"]]

base_model_name = Path(f"{MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

tot = 0

for i in range(len(data)):
    current = data.iloc[i]
    prompt = str(current['Content'])
    tok_out = tokenizer(prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1024)

    tot += len(tok_out)

print(f"Total text token count: {tot}")

test_tokenizer_vals = [
    "rtyagi86", "riya", "FRIEND", "Riya", "friend", '[Riya]',
    '[FRIEND]', '[Riya]:', '[FRIEND]:', 'Riya:', 'FRIEND:'
]

# friend name is single token woahhh, guess its pretty popular in proper nouns, prob bc he's white :p

for i in test_tokenizer_vals:
    tok_out = tokenizer(i,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1024)
    # ids = tokenizer.convert_tokens_to_ids(tok_out[0])

    ids = tok_out["input_ids"][0].tolist()  # tensor to list
    attn_mask = tok_out[
        "attention_mask"]  # without [0] we have this 2D thing, handling batch size > 1 I think

    tokens = tokenizer.convert_ids_to_tokens(ids)

    print(f"\n\"{i}\" →", list(zip(tokens, ids)))
    print(f"mask: {attn_mask}")

    # print(f"Token example: {i}")
    # print("Tokenizer output", tok_out)
    # print("Tokenizer association", tokens)
    # print("IDs:", ids)
