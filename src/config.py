from transformers.utils.quantization_config import BitsAndBytesConfig
import torch
import pandas as pd
from pathlib import Path

# Data configuration
DATA_PATH = "data/finetune_9_3_25"

# Speaker configuration
FRIEND_ID = 547589426310676551
RIYA_ID = 543198387860799498

RIYA_NAME = "Riya"
FRIEND_NAME = "Owen"

RIYA_SPEAKER_TOKEN = f"[{RIYA_NAME}]"
FRIEND_SPEAKER_TOKEN = f"[{FRIEND_NAME}]"

# Mention tokens (not currently used)
RIYA_MENTION_TOKEN = f"<m:{RIYA_NAME}>"
FRIEND_MENTION_TOKEN = f"<m:{FRIEND_NAME}>"

# Model configuration - use model_utils.py for centralized management
# Default model for backward compatibility
MISTRAL_7B = "mistral-7b"
GEMMA_3_27B_IT = "gemma-3-27b-it"

CURRENT_MODEL = GEMMA_3_27B_IT

# Legacy results folders for backward compatibility
LEGACY_RESULTS_FOLDER = "mistral-results-7-27-25"
OLD_RESULTS_FOLDER = "mistral-results-7-6-25"

# Centralized folder structure
CONVO_FOLDER = "convos"
VIZ_FOLDER = "visualizations"

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Instruct format configuration
# The system prompt defines the model's role and context
INSTRUCT_SYSTEM_PROMPT = "You are a chat model trained to simulate conversations between two friends, {FRIEND_SPEAKER_TOK} and {RIYA_SPEAKER_TOK}."

# The user prompt contains the conversation history and question
# This will be formatted as: {conversation_history}\n\nWhat does {next_speaker} say next?
INSTRUCT_USER_PROMPT_TEMPLATE = "{conversation_history}\n\nWhat does {next_speaker} say next?"

MODEL_CONFIGS = {
    "mistral-7b": {
        "model_name":
        "mistralai/Mistral-7B-v0.1",
        "model_type":
        "base",
        "lora_targets": ["q_proj", "v_proj"],
        "lora_targets_with_embeddings":
        ["embed_tokens", "lm_head", "q_proj", "v_proj"],
        "lora_targets_with_mlp":
        ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_targets_with_embeddings_and_mlp": [
            "embed_tokens", "lm_head", "q_proj", "v_proj", "gate_proj",
            "up_proj", "down_proj"
        ]
    },
    "gemma-3-27b-it": {
        "model_name":
        "google/gemma-3-27b-it",
        "model_type":
        "instruct",
        "lora_targets": ["q_proj", "v_proj"],
        "lora_targets_with_embeddings":
        ["embed_tokens", "lm_head", "q_proj", "v_proj"],
        "lora_targets_with_mlp":
        ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_targets_with_embeddings_and_mlp": [
            "embed_tokens", "lm_head", "q_proj", "v_proj", "gate_proj",
            "up_proj", "down_proj"
        ]
    }
}

# Centralized results folder structure
BASE_RESULTS_FOLDER = "models"

RESULTS_FOLDER_STRUCTURE = {
    "mistral-7b": f"{BASE_RESULTS_FOLDER}/mistral-7b",
    "gemma-3-27b-it": f"{BASE_RESULTS_FOLDER}/gemma-3-27b-it"
}
