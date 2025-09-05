# This file will enable continuous sampling to talk to Friend only, filtering out [RIYA] tags

import unsloth
from unsloth import FastLanguageModel

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
from datasets import load_dataset, DatasetDict
from typing import List
from tqdm import tqdm
from pathlib import Path
from transformers import BitsAndBytesConfig
import json
import argparse

from src.config import RIYA_NAME, FRIEND_NAME, MODEL_CONFIGS, RIYA_SPEAKER_TOKEN, FRIEND_SPEAKER_TOKEN, bnb_config
from src.model_utils import generate, generate_with_steering, stream_generate, stream_generate_steer
from src.steering.steer_utils import generate_steering_vector
from src.data_utils import clean_for_sampling

parser = argparse.ArgumentParser()
parser.add_argument('--steer', action='store_true')
parser.add_argument('--stream', action='store_true')
args = parser.parse_args()
steer = args.steer
stream = args.stream

checkpoint_path_old = Path(
    "models/gemma-3-27b-it/gemma-3-27b-it_20250903_122252/training_output/checkpoint-276"
)
checkpoint_path = Path(
    "models/mistral-7b/mistral-results-7-6-25/checkpoint-8103")
max_new_tokens = 50

# Load base model name from checkpoint's adapter config
adapter_config_path = checkpoint_path / "adapter_config.json"
with open(adapter_config_path, 'r') as f:
    adapter_config = json.load(f)
base_model_name = adapter_config["base_model_name_or_path"]
if "-it" in base_model_name.lower():
    model_type = "instruct"
else:
    model_type = "base"

print(f"Loading {model_type} model: {base_model_name}")
print(f"Using checkpoint: {checkpoint_path}")

# Load tokenizer from checkpoint (has special tokens)
tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
tokenizer.pad_token = tokenizer.eos_token

# Load processor for instruct models
processor = None
if model_type == "instruct":
    try:
        processor = AutoProcessor.from_pretrained(base_model_name,
                                                  use_fast=True)
        if hasattr(processor, 'tokenizer'):
            processor.tokenizer = tokenizer  # Align with adapter tokenizer
        print("Loaded processor for instruct model")
    except Exception as e:
        print(f"No processor found: {e}")

# Load base model with quantization for large models
quantization_config = bnb_config
if "27b" in base_model_name.lower():
    device_map = "cuda:0"
else:
    device_map = "auto"

print("Loading base model...")
# Use FastLanguageModel for unsloth models to match training setup
if "unsloth" in base_model_name.lower():
    base_model, _ = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=4096,
        dtype=torch.bfloat16,
        load_in_4bit=True if quantization_config else False,
        load_in_8bit=False,
    )
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device_map,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )

# Resize embeddings if needed
if base_model.get_input_embeddings().num_embeddings != len(tokenizer):
    base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

print("Loading LoRA adapter from checkpoint...")

lora_model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
lora_model.eval()
lora_model = lora_model.to(torch.bfloat16)

# Steering configuration (similar to sample_friend.py)
if steer:
    layer_extract = -5  # layers range from -33 to -1 for extraction, going for output
    layer_steer = -2  # layers range from -32 to -1, going for input
    alpha = 0.5

    steer_dict = {
        "i feel happy": 0.25,
        "i feel sad": -0.25,
        "life is amazing": 0.25,
        "life is terrible": -0.25,
        "this was a great day": 0.25,
        "this was a bad day": -0.25,
        "amazing": 0.25,
        "not bad": -0.25,
    }

    steering_vector = generate_steering_vector(
        lora_model,
        tokenizer,
        steer_dict,
        pos_alpha=alpha,
        neg_alpha=alpha,
        layer_from_last=layer_extract,
        processor=processor,
        is_instruct=model_type == "instruct",
        current_speaker=RIYA_NAME)

history = []
hist_count = 0  # up to 8 since thats curr length

print(f"Start your conversation with {RIYA_NAME}")
if steer:
    print("Steering enabled with default configuration")
if stream:
    print("Streaming enabled - responses will appear token by token")

while (1):

    # ToDo: make history a stack
    if hist_count > 8:  # maybe would be good on > 8 still? hmm maybe should set higher for training
        history.pop(0)

    prompt = input()

    # Build prompt line (avoid double-prefixing if already has speaker token)
    if not (prompt.strip().startswith(f"[{FRIEND_NAME}]")
            or prompt.strip().startswith(f"[{RIYA_NAME}]")):
        prompt_line = f"\n[{FRIEND_NAME}] {prompt}"
    else:
        prompt_line = f"\n{prompt}"

    history.append(prompt_line)
    hist_count += 1

    # Build full conversation history
    full_prompt = "".join(history)

    # Generate using unified function that handles both model types
    is_instruct = model_type == "instruct"
    target_speaker = RIYA_NAME  # Always target Riya for responses

    if stream:
        # Handle streaming generation
        if steer:
            # Use the new stream_generate_steer function
            print(f"[{RIYA_NAME}]: ", end="", flush=True)
            full_response = ""

            for token in stream_generate_steer(lora_model,
                                               full_prompt,
                                               tokenizer,
                                               steering_vector,
                                               max_new_tokens=max_new_tokens,
                                               layer_from_last=layer_steer,
                                               processor=processor,
                                               is_instruct=is_instruct,
                                               target_speaker=target_speaker,
                                               deployment=True):
                print(token, end="", flush=True)
                full_response += token

            print()  # New line after streaming is complete

            # Clean up the full response and add to history
            message = clean_for_sampling(full_response)
            # Split by [Riya] tokens to get separate messages
            message_parts = message.split(f"[{RIYA_NAME}]")
            for part in message_parts:
                if part.strip():  # Only add non-empty parts
                    history.append(f"\n[{RIYA_NAME}] {part.strip()}")
                    hist_count += 1
        else:
            # Use the new stream_generate function
            print(f"[{RIYA_NAME}]: ", end="", flush=True)
            full_response = ""

            for token in stream_generate(lora_model,
                                         full_prompt,
                                         tokenizer,
                                         max_new_tokens=max_new_tokens,
                                         processor=processor,
                                         is_instruct=is_instruct,
                                         target_speaker=target_speaker,
                                         deployment=True):
                print(token, end="", flush=True)
                full_response += token

            print()  # New line after streaming is complete

            # Clean up the full response and add to history
            message = clean_for_sampling(full_response)
            # Split by [Riya] tokens to get separate messages
            message_parts = message.split(f"[{RIYA_NAME}]")
            for part in message_parts:
                if part.strip():  # Only add non-empty parts
                    history.append(f"\n[{RIYA_NAME}] {part.strip()}")
                    hist_count += 1

    else:
        # Non-streaming generation (original logic)
        if steer:
            # Generate with steering
            messages = generate_with_steering(lora_model,
                                              full_prompt,
                                              tokenizer,
                                              steering_vector,
                                              max_new_tokens=max_new_tokens,
                                              layer_from_last=layer_steer,
                                              is_instruct=is_instruct,
                                              target_speaker=target_speaker,
                                              processor=processor)
        else:
            # Regular generation without steering
            messages = generate(lora_model,
                                full_prompt,
                                tokenizer,
                                max_new_tokens=max_new_tokens,
                                processor=processor,
                                is_instruct=is_instruct,
                                target_speaker=target_speaker)

        # Handle the list of messages
        for message in messages:
            # Clean up the message
            message = clean_for_sampling(message)

            # Add to history with speaker token
            history.append(f"\n[{RIYA_NAME}] {message}")
            hist_count += 1

            # Print the message
            print(f"[{RIYA_NAME}]: {message}")

    # why the <s>? appears?
