from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling

from peft import get_peft_model, LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training
from src.config import FRIEND_NAME, RIYA_NAME, MODEL_NAME, RESULTS_FOLDER, DATA_PATH, bnb_config
from src.data_utils import get_speaker_tokens, strip_training_data
from datasets import load_dataset, DatasetDict
from src.finetuning.callbacks import SampleGenerationCallback, LiveJSONLogger
import json
import torch
from pathlib import Path
import argparse

print("Torch available: ", torch.cuda.is_available())
assert(torch.cuda.is_available())
torch.cuda.empty_cache()
torch.cuda.ipc_collect() # in case restart training

device = 'cuda:0'  # auto fails cause data tensor on multiple devices

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-st', '--special-tokens', action='store_true', help='Load from special token embeddings')
parser.add_argument('-ct', '--continue-training', action='store_true', help='Continue training existing LoRA adapter')
args = parser.parse_args()
# if continuing, special tokens flag must be on since we trained on that

# load model and tokenizer based on flags
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if args.special_tokens:
    special_tokens = get_speaker_tokens()
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    tokenizer.pad_token = tokenizer.eos_token
else:
    special_tokens = []
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map=device
)

model.resize_token_embeddings(len(tokenizer))

# enable k-bit gradient adapters
model = prepare_model_for_kbit_training(model)

# handle continuation of training
adapter_dir = Path(f"./{RESULTS_FOLDER}/lora_adapter")
if args.continue_training and adapter_dir.exists(): 
    print(f"Continuing LoRA from {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
else:
    print("Starting new LoRA adapter")
    # include embed_tokens and lm_head if special_tokens
    targets = ['q_proj', 'v_proj']
    if args.special_tokens:
        targets = ['embed_tokens', 'lm_head'] + targets
    lora_conf = LoraConfig(
        r=12, # some more complexity
        lora_alpha=16,
        target_modules=targets,
        lora_dropout=0.05, # smaller dropout, not as worried about overfitting (though I did see some, so careful)
        bias='all',
        task_type=TaskType.CAUSAL_LM, # infers additional modules to save
    )
    model = get_peft_model(model, lora_conf)

# initialize special token embeddings to be similar to existing ones
if args.special_tokens:
    emb = model.get_input_embeddings()
    with torch.no_grad():
        mapping = {
            special_tokens[0]: ["ri", "ya", "_R"],
            special_tokens[1]: ["▁Owen"] # that is NOT an _ it's the space
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

# load dataset
dataset = DatasetDict({
    "train": load_dataset("json", data_files=f"{DATA_PATH}/train.json", split="train"),
})

# tokenize data
def tokenize(example):
    full_text = example["prompt"].strip() + " " + example["response"].strip()
    tokenized = tokenizer(full_text, 
                          truncation=True, 
                          padding=False)
    return tokenized

tokenized_dataset = dataset.map(tokenize, remove_columns = ["prompt", "response"])

# training args + specifications
output_dir = Path(f"./{RESULTS_FOLDER}/lora_train")
output_dir.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,          
    num_train_epochs=6,
    learning_rate=2e-4,
    warmup_steps=200,
    lr_scheduler_type="linear",
    fp16=False,                             
    logging_steps=50, # calls log callback on log
    save_strategy="steps",
    save_steps=1500,
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
        # welp i no longer have test.json, I guess we get intuition for memorization now
        SampleGenerationCallback(tokenizer, log_path = output_dir / "mid_completions.json", test_data_path = Path(DATA_PATH) / "train.json", every_n_steps=200),
        LiveJSONLogger(log_path = output_dir / "log.json")
    ],
)

# ToDo: continue training rewrites log.json, and mid_completions.json, fix this (append vs. rewrite)
if args.continue_training: 
    print("Continuing LoRA training...")
    trainer.train(resume_from_checkpoint=output_dir / "lora_adapter")
else:
    print("Starting new LoRA training...")
    trainer.train()

# saving outputs
# ToDo: add option to not rewrite old lora adapters if continuing or smt

model.save_pretrained(output_dir / "lora_adapter")
tokenizer.save_pretrained(output_dir / "lora_adapter")
trainer.save_state() # annoying it won't get saved in lora_adapter, fix :(, have to move it manually

print(f"LORA training complete. Model saved to {output_dir}")

