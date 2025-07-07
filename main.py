from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from config import FRIEND_NAME
from datasets import load_dataset, DatasetDict
import json

model_name = "mistralai/Mistral-7B-v0.1"
results_folder = "mistral-results"

tokenizer = AutoTokenizer.from_pretrained(model_name)
special_tokens = {"additional_special_tokens": ["[RIYA]", f"[FRIEND_NAME]"]}
tokenizer.add_special_tokens(special_tokens)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
model.resize_token_embeddings(len(tokenizer))

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
    "test": load_dataset("json", data_files="test.json", split="train"),
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
    output_dir=f"./{results_folder}",
    per_device_train_batch_size=2,          
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,                              
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    report_to="none",                       
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained(f"{results_folder}/lora_adapter")
tokenizer.save_pretrained(f"{results_folder}/lora_adapter")

