from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling

from peft import get_peft_model, LoraConfig, TaskType
from src.config import FRIEND_NAME, RIYA_NAME, MODEL_NAME, RESULTS_FOLDER, bnb_config
from datasets import load_dataset, DatasetDict
from callbacks import SampleGenerationCallback, LiveJSONLogger
import json
import torch

print("Torch available: ", torch.cuda.is_available())
assert(torch.cuda.is_available())
torch.cuda.empty_cache()
torch.cuda.ipc_collect() # in case restart training

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# special_tokens = {"additional_special_tokens": ["[RIYA]", f"[{FRIEND_NAME}]"]}
# tokenizer.add_special_tokens(special_tokens)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", quantization_config=bnb_config)
# model.resize_token_embeddings(len(tokenizer))

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

dataset = DatasetDict({
    "train": load_dataset("json", data_files="train.json", split="train"),
    "test": load_dataset("json", data_files="test.json", split="train"), # only train split in test
})

# tokenize data

def tokenize(example):
    full_text = example["prompt"].strip() + " " + example["response"].strip()
    tokenized = tokenizer(full_text, 
                          truncation=True, 
                          padding="max_length",
                          max_length=2048,)
    tokenized["labels"] = tokenized["input_ids"].copy() # for Causal LM, predicting same as input, just shifted
    return tokenized

tokenized_dataset = dataset.map(tokenize, remove_columns = ["prompt", "response"])

training_args = TrainingArguments(
    output_dir=f"./{RESULTS_FOLDER}",
    per_device_train_batch_size=4,          
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,                             
    logging_steps=50, # calls log callback on log
    save_strategy="steps",
    save_steps=1500,
    # save_total_limit=3,  # optionally keep only last 2 checkpoints - yes, dont waste too much space
    # logging_dir=f"./{RESULTS_FOLDER}/logs",
    report_to="none",                       
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer( # uses cross entropy
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    callbacks=[
        SampleGenerationCallback(tokenizer, log_path = f"{RESULTS_FOLDER}/mid_completions.json", test_data_path = "test.json", every_n_steps=200),
        LiveJSONLogger(log_path=f"{RESULTS_FOLDER}/log.json")
    ],
)

trainer.train()

model.save_pretrained(f"{RESULTS_FOLDER}/lora_adapter")
tokenizer.save_pretrained(f"{RESULTS_FOLDER}/lora_adapter")


