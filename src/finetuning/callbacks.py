from transformers.trainer_callback import TrainerCallback
import torch
import json
import os
import random
from src.model_utils import generate


class SampleGenerationCallback(TrainerCallback):

    def __init__(self,
                 tokenizer,
                 log_path,
                 test_data_path,
                 every_n_steps=500,
                 max_new_tokens=50):
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

    #     decoded = self.tokenizer.decode(output[0]) # skip_special_tokens=False
    #     return decoded

    def _format_instruct_prompt(self, raw_prompt: str) -> str:
        """
        Make instruct prompts usable for generation without requiring a processor.
        - If [SYS]/[USER] tags are present, extract user content; optionally prepend system.
        - Otherwise, return stripped prompt.
        This avoids callback failures for instruct/VLM models during sampling.
        """
        prompt = raw_prompt.strip()
        try:
            has_tags = all(tag in prompt for tag in ("[SYS]", "[/SYS]", "[USER]", "[/USER]"))
            if not has_tags:
                return prompt

            s0 = prompt.find("[SYS]") + len("[SYS]"); s1 = prompt.find("[/SYS]")
            u0 = prompt.find("[USER]") + len("[USER]"); u1 = prompt.find("[/USER]")
            system_content = prompt[s0:s1].strip()
            user_content = prompt[u0:u1].strip()

            # Keep it simple for sampling: include system if present, then user
            if system_content:
                return f"{system_content}\n{user_content}"
            return user_content
        except Exception:
            # On any parsing issue, fall back to the raw prompt to avoid breaking training
            return prompt

    def on_step_end(self, args, state, control, seed=228, **kwargs):
        if state.global_step % self.every_n_steps == 0 and state.global_step > 0:
            model = kwargs["model"]
            model.eval()

            # somehow in diff training runs it's the same random choice ?? figure that out
            sample = random.choice(self.samples)
            prompt_new = self._format_instruct_prompt(sample.get("prompt", ""))
            decoded_new = generate(model,
                                   prompt_new,
                                   self.tokenizer,
                                   max_new_tokens=self.max_new_tokens)
            completion_new = decoded_new

            # saving both a new prompt for fun, and a constant prompt to track progress over time
            prompt_const = self._format_instruct_prompt(self.samples[seed].get("prompt", ""))
            decoded_const = generate(model,
                                     prompt_const,
                                     self.tokenizer,
                                     max_new_tokens=self.max_new_tokens)
            completion_const = decoded_const  # decoded_const[len(prompt_const)-2:].strip() why -2 idk, shouldnt need now

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
            f.write(
                ""
            )  # overwrite file, add functionality later to resume training

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logs["step"] = state.global_step
            self.history.append(logs)
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
