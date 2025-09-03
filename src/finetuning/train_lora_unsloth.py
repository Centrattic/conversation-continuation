import torch
import argparse
from pathlib import Path
import json
from datetime import datetime

# Unsloth imports
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, DatasetDict

from transformers import AutoProcessor
from PIL import Image
import io
import requests

# Local imports
from src.model_utils import (get_model_config, get_results_folder,
                             create_experiment_folder, save_experiment_config,
                             get_lora_adapter_path, load_image)
from src.config import FRIEND_NAME, RIYA_NAME, DATA_PATH, bnb_config
from src.data_utils import get_speaker_tokens
from src.finetuning.callbacks import SampleGenerationCallback, LiveJSONLogger

print("Torch available: ", torch.cuda.is_available())
assert torch.cuda.is_available()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

IS_GEMMA_3_VLM = False

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train LoRA with Unsloth")
parser.add_argument('--model',
                    type=str,
                    default='gemma-3-27b-it',
                    choices=['mistral-7b', 'gemma-3-27b-it'],
                    help='Model to train')

parser.add_argument('--experiment',
                    type=str,
                    default=None,
                    help='Experiment name (defaults to timestamp)')

parser.add_argument('--special-tokens',
                    action='store_false', # default true without argument, should rename args
                    help='Load from special token embeddings')

parser.add_argument('--continue-training',
                    action='store_true',
                    help='Continue training existing LoRA adapter')

parser.add_argument('--include-mlp',
                    action='store_false',
                    help='Include MLP layers in LoRA for better fact storage')

parser.add_argument('--instruct-format',
                    action='store_false',
                    help='Use instruct format training data')

parser.add_argument('--data-path',
                    type=str,
                    default=None,
                    help='Path to training data (optional, auto-detects based on config, model type)')

parser.add_argument('--epochs',
                    type=int,
                    default=2,
                    help='Number of training epochs')

parser.add_argument('--batch-size',
                    type=int,
                    default=32,
                    help='Training batch size')

parser.add_argument('--learning-rate',
                    type=float,
                    default=2e-4,
                    help='Learning rate')

parser.add_argument('--max-seq-length',
                    type=int,
                    default=4096,
                    help='Maximum sequence length')

parser.add_argument('--quantization',
                    type=str,
                    choices=['4bit', '8bit', 'auto'],
                    default='8bit',
                    help='Quantization level (4bit, 8bit, or auto based on model)')

args = parser.parse_args()

# Set up experiment
if args.experiment is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{args.model}_{timestamp}"
else:
    experiment_name = args.experiment

# Get model configuration
model_config = get_model_config(args.model)
model_name = model_config["model_name"]
model_type = model_config["model_type"]

# Create experiment folder
experiment_folder = create_experiment_folder(args.model, experiment_name,)

# Save experiment configuration
experiment_config = {
    "model": args.model,
    "model_name": model_name,
    "model_type": model_type,
    "experiment_name": experiment_name,
    "special_tokens": args.special_tokens,
    "include_mlp": args.include_mlp,
    "instruct_format": args.instruct_format,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "learning_rate": args.learning_rate,
    "quantization": args.quantization,
    "max_seq_length": args.max_seq_length,
    "timestamp": datetime.now().isoformat()
}
save_experiment_config(args.model, experiment_name, experiment_config)

print(f"Starting experiment: {experiment_name}")
print(f"Model: {model_name}")
print(f"Model type: {model_type}")

# Load model and tokenizer with Unsloth
max_seq_length = args.max_seq_length
dtype = None  # None for auto detection

if args.quantization == "8bit":
    load_in_4bit = False
    load_in_8bit = True
elif args.quantization == "4bit":
    load_in_4bit = True
    load_in_8bit = False

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    load_in_8bit=load_in_8bit,
)

if "gemma-3-27b-it" in model_name.lower():
    IS_GEMMA_3_VLM = True
    processor = AutoProcessor.from_pretrained(model_name)
else:
    processor = None

# Set up chat template for instruct models

# Set up chat template for instruct models
if model_type == "instruct":
    if IS_GEMMA_3_VLM:
        # Gemma VLM: use processor.apply_chat_template; do NOT set tokenizer.chat_template
        pass
    else:
        # Non-VLM: use tokenizer's built-in template if it exists
        tmpl = getattr(tokenizer, "chat_template", None)
        if isinstance(tmpl, str):
            tokenizer.chat_template = tmpl
        else:
            print("[info] no string chat_template on tokenizer; falling back to plain text formatting")

# Add special tokens if requested
if args.special_tokens:
    special_tokens = get_speaker_tokens()
    
    if IS_GEMMA_3_VLM:
        tokenizer = processor.tokenizer
    
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    model.resize_token_embeddings(len(tokenizer))

    # Initialize special token embeddings
    emb = model.get_input_embeddings()
    with torch.no_grad():
        mapping = {
            special_tokens[0]: ["ri", "ya", "_R"],
            special_tokens[1]: ["▁Owen"]  # that is NOT an _ it's the space
        }
        for tok, refs in mapping.items():
            tid = tokenizer.convert_tokens_to_ids(tok)
            ref_ids = tokenizer.convert_tokens_to_ids(refs)
            ref_ids = [i for i in ref_ids if i != tokenizer.unk_token_id]
            if not ref_ids:
                print(f"No valid refs for {tok}")
                continue
            mean_vec = emb.weight[ref_ids].mean(0)
            emb.weight[tid] = mean_vec + torch.randn_like(mean_vec) * 0.01

# Set up LoRA configuration
if args.include_mlp:
    if args.special_tokens:
        target_modules = model_config["lora_targets_with_embeddings_and_mlp"]
    else:
        target_modules = model_config["lora_targets_with_mlp"]
else:
    if args.special_tokens:
        target_modules = model_config["lora_targets_with_embeddings"]
    else:
        target_modules = model_config["lora_targets"]

# Configure LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=12,  # LoRA rank
    target_modules=target_modules,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="all",
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

# Load dataset - automatically select correct training file

data_path = DATA_PATH if args.data_path is None else args.data_path

if args.instruct_format and model_type == "instruct":
    training_file = "instruct_train.json"
else:
    training_file = "train.json"

data_path = f"{data_path}/{training_file}"

print(f"Loading dataset from: {data_path}")
print(f"Model type: {model_type}, Instruct format: {args.instruct_format}")

dataset = DatasetDict({
    "train":
    load_dataset("json", data_files=data_path, split="train"),
})


# Tokenize data
def formatting(example):
    prompt = example["prompt"].strip()
    response = example["response"].strip()

    # currently training without images, could change in future
    if IS_GEMMA_3_VLM and (args.instruct_format and model_type == "instruct"):
        # print("IS GEMMA VLM")
        # Extract [SYS] and [USER] exactly like your non-VLM path
        if all(tag in prompt for tag in ("[SYS]", "[/SYS]", "[USER]", "[/USER]")):
            s0 = prompt.find("[SYS]") + len("[SYS]"); s1 = prompt.find("[/SYS]")
            u0 = prompt.find("[USER]") + len("[USER]"); u1 = prompt.find("[/USER]")
            system_content  = prompt[s0:s1].strip()
            user_content = prompt[u0:u1].strip()
    
        img_ref = example.get("image", None)
        img = None
    
        if img_ref:
            try:
                img = load_image(img_ref)
            except Exception as e:
                print(f"[warn] failed to load image {img_ref}: {e}")

            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_content}]},
                {"role": "user", "content": [{"type": "text", "text": user_content},
                                               {"type": "image", "image": img}]},
                {"role": "assistant", "content": [{"type": "text", "text": response}]},
            ]
        else: 
            # print("No images used in training") # we should always get this
            messages = [
                    {"role": "system", "content": [{"type": "text", "text": system_content}]},
                    {"role": "user", "content": [{"type": "text", "text": user_content}]},
                    {"role": "assistant", "content": [{"type": "text", "text": response}]},
                ]
    
        kwargs = dict(
            # messages=messages,
            add_generation_prompt=False,
            tokenize=False,
            return_dict=True,
            return_tensors="pt",
        )
        
        proc_out = processor.apply_chat_template(conversation=messages, **kwargs)

        return proc_out

    # elif args.instruct_format and model_type == "instruct":
    #     print("NOT IS GEMMA VLM")
    #     # Parse [SYS]/[USER] blocks if present; else fallback to user→assistant
    #     if all(tag in prompt for tag in ("[SYS]", "[/SYS]", "[USER]", "[/USER]")):
    #         s0 = prompt.find("[SYS]") + len("[SYS]"); s1 = prompt.find("[/SYS]")
    #         u0 = prompt.find("[USER]") + len("[USER]"); u1 = prompt.find("[/USER]")
    #         system_content = prompt[s0:s1].strip()
    #         user_content   = prompt[u0:u1].strip()
    #         messages = [
    #             {"role": "system", "content": system_content},
    #             {"role": "user",   "content": user_content},
    #             {"role": "assistant", "content": response},
    #         ]
    #     else:
    #         raise ValueError("data processing failed")
    #         # messages = [
    #         #     {"role": "user", "content": prompt},
    #         #     {"role": "assistant", "content": response},
    #         # ]

    #     formatted = tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=False,
    #         add_generation_prompt=False,
    #         chat_template=tokenizer.chat_template,  # explicit
    #     )
    #     return tokenizer(formatted, truncation=True, padding=False, max_length=max_seq_length,)

    else: # base / non-instruct
        full_text = f"{prompt}\n{response}"
        return full_text # tokenizer(full_text, truncation=True, padding=False, max_length=max_seq_length,)

formatted_dataset = dataset.map(lambda example: {
                                            "text": formatting(example)
                                            },
                                            remove_columns=["prompt", "response"])

# Set up training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    warmup_steps=200,
    num_train_epochs=args.epochs,
    learning_rate=args.learning_rate,
    fp16=False, # just so slow
    bf16=True,
    logging_steps=50,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir=str(experiment_folder / "training_output"),
    save_strategy="steps",
    save_steps=1500,
    report_to="none",
)

# Set up trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset["train"],
    dataset_text_field="text",  # We're providing pre-tokenized data, other option "text"
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences
    args=training_args,
    callbacks=[
        SampleGenerationCallback(tokenizer,
                                 log_path=experiment_folder /
                                 "mid_completions.json",
                                 test_data_path=Path(data_path),
                                 every_n_steps=200),
        LiveJSONLogger(log_path=experiment_folder / "log.json")
    ],
)

# Handle continuation of training
adapter_path = get_lora_adapter_path(args.model, experiment_name)
if args.continue_training and adapter_path.exists():
    print(f"Continuing LoRA from {adapter_path}")
    trainer.train(resume_from_checkpoint=str(adapter_path))
else:
    print("Starting new LoRA training...")
    trainer.train()

# Save the model
model.save_pretrained(str(adapter_path))
tokenizer.save_pretrained(str(adapter_path))

print(f"LoRA training complete. Model saved to {adapter_path}")
print(
    f"Experiment configuration saved to {experiment_folder / 'experiment_config.json'}"
)
