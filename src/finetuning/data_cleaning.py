import pandas as pd
from datetime import datetime
from typing import List, Dict
import json
from src.config import FRIEND_NAME, RIYA_NAME, RIYA_SPEAKER_TOKEN, FRIEND_SPEAKER_TOKEN, DATA_PATH
from tqdm import tqdm
import random

def load_and_prepare_data(path: str, context_window: int = 8, max_gap_minutes: int = 15, 
                          exclude_strings=[]):
    df = pd.read_csv(path)
    df.columns = ["AuthorID", "Author", "Date", "Content", "Attachments", "Reactions"]
    df = df.dropna(subset=["Content"])  # remove empty messages

    if exclude_strings:
        # Use case-insensitive filtering; adjust flags if needed
        for text in exclude_strings:
            df = df[~df["Content"].str.contains(text, case=False, na=False)]

    df["Date"] = pd.to_datetime(df["Date"], format="ISO8601") # %y-%m-%dT%H:%M:%S.%f+%z
    df = df.sort_values(by="Date") # want to train on latest data earliest (some research has shown this is relevant)

    conversations = []
    buffer = []

    for i in tqdm(range(len(df))):
        current = df.iloc[i]
        if i > 0:
            prev_time = df.iloc[i - 1]["Date"]
            gap = (current["Date"] - prev_time).total_seconds() / 60
            if gap > max_gap_minutes:
                buffer = []  # reset context due to large gap

        speaker = f"[{FRIEND_NAME}]" if current["Author"] == "rtyagi86" else f"[{FRIEND_NAME}]"
        buffer.append(f"{speaker} {current['Content'].strip()}")

        if len(buffer) >= context_window + 1: # +1 for the response
            prompt = "\n".join(buffer[-(context_window+1):-1])
            next_speaker = buffer[-1].split(" ", 1)[0]
            prompt += f"\n{next_speaker}"
            target = buffer[-1].split(" ", 1)[-1]  # remove speaker from target
            conversations.append({"prompt": prompt, "response": target})

    return conversations

def train_test_split(conversations: List[Dict], train_ratio=0.9, seed=42):
    random.seed(seed)
    random.shuffle(conversations)
    split_idx = int(len(conversations) * train_ratio)
    train_data = conversations[:split_idx]
    test_data = conversations[split_idx:]
    return train_data, test_data

def save_json(data: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


data = load_and_prepare_data(f"{DATA_PATH}/friend_hist_new.csv")

# will mess with conversation starter tags
exclude_strings = [RIYA_SPEAKER_TOKEN, FRIEND_SPEAKER_TOKEN] 

save_json(data, f"{DATA_PATH}/train.json") # train on all, no splits


