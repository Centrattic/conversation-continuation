from transformers.utils.quantization_config import BitsAndBytesConfig
import torch

FRIEND_NAME = "Friend"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
RESULTS_FOLDER = "mistral-results"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

