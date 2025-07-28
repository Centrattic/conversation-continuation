# Goal: train special token embeddings first to make LORA finetuning better.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from datasets import load_dataset, DatasetDict
import json
from pathlib import Path

from src.config import MODEL_NAME, RESULTS_FOLDER, bnb_config
from src.data_utils import get_speaker_tokens, strip_training_data
from src.config import RIYA_NAME, FRIEND_NAME

print("Torch available: ", torch.cuda.is_available())
assert(torch.cuda.is_available())
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# load tokenizer and add special tokens
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
special_tokens = get_speaker_tokens()

tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
tokenizer.pad_token = tokenizer.eos_token

# load model and resize embeddings
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", quantization_config=bnb_config)
model.resize_token_embeddings(len(tokenizer))

# freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# unfreeze only the embeddings for special tokens
embeddings = model.get_input_embeddings()
special_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)
for token_id in special_token_ids:
    embeddings.weight[token_id].requires_grad = True

print(f"Added special tokens: {special_tokens}")
print(f"Special token IDs: {special_token_ids}")
print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# load and preprocess dataset
dataset = DatasetDict({
    "train": load_dataset("json", data_files=f"{DATA_PATH}/train.json", split="train"),
    # "test": load_dataset("json", data_files="test.json", split="train"), 
    # no test split anymore
})

processed_dataset = dataset.map(strip_training_data, remove_columns=["prompt", "response"])

# tokenize data
def tokenize(example):
    full_text = example["prompt"].strip() + " " + example["response"].strip()
    tokenized = tokenizer(
        full_text, 
        truncation=True, 
        # padding="longest",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = processed_dataset.map(tokenize, remove_columns=["prompt", "response"])

output_dir = Path(f"./{RESULTS_FOLDER}/special_embedding_train")
output_dir.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    num_train_epochs=2,  # fewer epochs vs. lora
    learning_rate=1e-3,  # higher learning rate vs. lora 
    fp16=False,
    logging_steps=50,
    save_strategy="steps",
    save_steps=1000,
    report_to="none",
    warmup_steps=100,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False, 
    pad_to_multiple_of=8 # pad to multiple of 8 closest (above) to longest in batch
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=None,
    data_collator=data_collator,
)

print("Starting special token training...")
trainer.train()

model.save_pretrained(output_dir / "emb_adapter")
tokenizer.save_pretrained(output_dir / "emb_adapter")
trainer.save_state()

print(f"Special token training complete. Model saved to {output_dir}")

training_info = {
    "special_tokens": special_tokens,
    "special_token_ids": special_token_ids,
    "training_epochs": 2,
    "learning_rate": 1e-3,
}

with open(output_dir / "training_info.json", "w") as f:
    json.dump(training_info, f, indent=2)   

