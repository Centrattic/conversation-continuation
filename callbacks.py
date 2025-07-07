from transformers.trainer_callback import TrainerCallback
import torch
import json
import os
import random
from model_utils import generate

class SampleGenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, log_path, test_data_path, every_n_steps=500, max_new_tokens=50):
        self.tokenizer = tokenizer
        self.history = []
        self.log_path = log_path
        self.every_n_steps = every_n_steps
        self.max_new_tokens = max_new_tokens
        self.test_data_path = test_data_path

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("")
            
        with open(test_data_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

    # def get_completion(self, model, prompt_text):
    #     input_ids = self.tokenizer(prompt_text, return_tensors="pt").to(model.device)
    #     with torch.no_grad():
    #         output = model.generate(
    #             **input_ids,
    #             max_new_tokens=self.max_new_tokens,
    #             do_sample=True,
    #             temperature=0.7,
    #             top_p=0.9,
    #             pad_token_id=self.tokenizer.eos_token_id,
    #         )
            
    #     decoded = self.tokenizer.decode(output[0]) # skip_special_tokens=False
    #     return decoded
    
    def on_step_end(self, args, state, control, seed = 228,**kwargs):
        if state.global_step % self.every_n_steps == 0 and state.global_step > 0:
            model = kwargs["model"]
            model.eval()

            # somehow in diff training runs it's the same random choice ?? figure that out
            sample = random.choice(self.samples)
            prompt_new = sample["prompt"].strip()
            decoded_new = generate(model, prompt_new, self.tokenizer, max_new_tokens=self.max_new_tokens)
            completion_new = decoded_new

            # saving both a new prompt for fun, and a constant prompt to track progress over time
            prompt_const = self.samples[seed]["prompt"].strip()
            decoded_const = generate(model, prompt_const, self.tokenizer, max_new_tokens=self.max_new_tokens)
            completion_const = decoded_const # decoded_const[len(prompt_const)-2:].strip() why -2 idk, shouldnt need now
            
            # print(f"\n[STEP {state.global_step}] Sample Completion:\n{decoded}\n" + "-"*50)

            completions = {}
            completions["step"] = state.global_step
            completions["prompt_const"] = prompt_const
            completions["decoded_const"] = completion_const
            completions["prompt_new"] = prompt_new
            completions["decoded_new"] = completion_new
            
            self.history.append(completions)
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
            
            model.train()

class LiveJSONLogger(TrainerCallback):
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.history = []

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("")  # overwrite file, add functionality later to resume training

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logs["step"] = state.global_step
            self.history.append(logs)
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
