from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling

from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from src.config import FRIEND_NAME, RIYA_NAME, MODEL_NAME, RESULTS_FOLDER, DATA_PATH, bnb_config
from src.data_utils import get_speaker_tokens, strip_training_data
from datasets import load_dataset, DatasetDict
from callbacks import SampleGenerationCallback, LiveJSONLogger
import json
import torch
from pathlib import Path
import argparse

print("Torch available: ", torch.cuda.is_available())
assert(torch.cuda.is_available())
torch.cuda.empty_cache()
torch.cuda.ipc_collect() # in case restart training

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-st', '--special-tokens', action='store_true', help='Load from special token embeddings')
parser.add_argument('-ct', '--continue-training', action='store_true', help='Continue training existing LoRA adapter')
args = parser.parse_args()

# load model and tokenizer based on flags
if args.special_tokens:
    special_tokens_path = Path(f"./{RESULTS_FOLDER}/special_embedding_train/emb_adapter")
    if special_tokens_path.exists():
        print(f"Loading from special tokens checkpoint: {special_tokens_path}")
        tokenizer = AutoTokenizer.from_pretrained(special_tokens_path)
        model = AutoModelForCausalLM.from_pretrained(special_tokens_path, device_map="auto", quantization_config=bnb_config)
    else:
        raise ValueError("Special tokens checkpoint not found")
else:
    print("Loading base model without special tokens")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", quantization_config=bnb_config)

# handle lora adapter based on continue flag
if args.continue_training:
    # load existing
    existing_lora_path = Path(f"./{OLD_RESULTS_FOLDER}/lora_adapter")
    if existing_lora_path.exists():
        print(f"Loading existing LoRA adapter from: {existing_lora_path}")
        model = PeftModel.from_pretrained(model, existing_lora_path)
    else:
        raise ValueError("LoRA adapter not found")
else:
    # create new adapter
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.01, # smaller dropout, not as worried about overfitting (though I did see some, so careful)
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

dataset = DatasetDict({
    "train": load_dataset("json", data_files=f"{DATA_PATH}/train.json", split="train"),
    # "test": load_dataset("json", data_files="test.json", split="train"), # only train split in test
})

processed_dataset = dataset.map(strip_training_data, remove_columns=["prompt", "response"])

# tokenize data
def tokenize(example):
    full_text = example["prompt"].strip() + " " + example["response"].strip()
    tokenized = tokenizer(full_text, 
                          truncation=True)
    tokenized["labels"] = tokenized["input_ids"].copy() # for Causal LM, predicting same as input, just shifted
    return tokenized

tokenized_dataset = processed_dataset.map(tokenize, remove_columns = ["prompt", "response"])

# set output directory based on flags

# hmm perhaps have same directory for both? And just add epoch number to checkpoint. Ahh idk. new directory for now, not sure
# if epoch number will continue from prev
if args.continue_training:
    output_dir = Path(f"./{RESULTS_FOLDER}/lora_continued") 
else:
    output_dir = Path(f"./{RESULTS_FOLDER}/lora_train")
output_dir.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,          
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_steps=200,
    lr_scheduler_type="linear",
    fp16=False,                             
    logging_steps=50, # calls log callback on log
    save_strategy="steps",
    save_steps=1500,
    # save_total_limit=3,  # optionally keep only last 2 checkpoints - yes, dont waste too much space
    # logging_dir=f"./{RESULTS_FOLDER}/logs",
    report_to="none",                       
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False, 
    pad_to_multiple_of=8
)

trainer = Trainer( # uses cross entropy
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=None,
    data_collator=data_collator,
    callbacks=[
        SampleGenerationCallback(tokenizer, log_path = f"{RESULTS_FOLDER}/mid_completions.json", test_data_path = "test.json", every_n_steps=200),
        LiveJSONLogger(log_path=f"{RESULTS_FOLDER}/log.json")
    ],
)

# Print training mode
if args.continue_training:
    print("Continuing LoRA training...")
else:
    print("Starting new LoRA training...")

trainer.train() # starting with old LORA waits but new training so LR is better

model.save_pretrained(output_dir / "lora_adapter")
tokenizer.save_pretrained(output_dir / "lora_adapter")
trainer.save_state()

print(f"LORA training complete. Model saved to {output_dir}")

