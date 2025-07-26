from transformers.utils.quantization_config import BitsAndBytesConfig
import torch
import pandas as pd

# ToDo: Refactor to use pathlib

DATA_PATH = "friend_hist.csv"
friend_csv = pd.read_csv(DATA_PATH)

FRIEND_ID = 547589426310676551
RIYA_ID = 543198387860799498

FRIEND_NAME = "Friend"
RIYA_NAME = "Riya"

# Use Discord names in next training run. For now, you have hardcoded ones.
# FRIEND_NAME = friend_csv.loc[friend_csv['AuthorID'] == FRIEND_ID]['Author'].iloc[0]
# RIYA_NAME = friend_csv.loc[friend_csv['AuthorID'] == RIYA_ID]['Author'].iloc[0]

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
RESULTS_FOLDER = "mistral-results"

CONVO_FOLDER = "convos"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

