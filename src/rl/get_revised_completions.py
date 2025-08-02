""" Every call to 4o-mini should contain (prompt, output, relevant consitution). 
Both constitutions maybe should not be contained in case this is confusing. 

I could have two instances of 4o-mini, riya rater, and friend rater with the relevant 
constitution and call each 
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from openai import OpenAI, OpenAIError
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src.config import MODEL_NAME, RESULTS_FOLDER, bnb_config
from src.model_utils import generate
from src.rl.rl_utils import load_constitutions, get_speaker_from_prompt, get_revised_completion



def main(args):

    # setup
    constitutions = load_constitutions(Path(args.constitutions_dir))

    api_key = os.getenv("OPENAI_API_KEY") # from env var
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    client = OpenAI(api_key=api_key)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading local model for base completions...")
    base_model_path = Path(MODEL_NAME)
    adapter_path = Path(RESULTS_FOLDER) / "lora_adapter"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, device_map="auto", quantization_config=bnb_config
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print("Model loaded.")

    # load prompts (from looped convos, or other is the idea)
    # don't use prompts from existing convos, since will eventually use all of that to train
    with open(args.input_file, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    # limit to n_samples if specified
    if args.n_samples > 0:
        prompts_data = prompts_data[: args.n_samples]

    # generate and revise data
    print(f"\n Generating and revising {len(prompts_data)} completions")
    sft_dataset: List[Dict[str, str]] = []

    for item in tqdm(prompts_data, desc="Processing prompts"):
        prompt_text = item["prompt"]
        try:
            speaker = get_speaker_from_prompt(prompt_text)
            constitution = constitutions[speaker]
        except (ValueError, KeyError) as e:
            print(f"Skipping prompt due to error: {e}")
            continue

        # generate base completion from your fine-tuned model
        with torch.no_grad():
            base_completion = generate(
                model, prompt_text, tokenizer, max_new_tokens=90
            )
        
        # ToDo: clean the base completion to not have stray tokens, check this at least

        # get revised completion from GPT-4o-mini
        revised_completion = get_revised_completion(
            client=client,
            conversation_prompt=prompt_text,
            original_completion=base_completion,
            persona_constitution=constitution,
            persona_name=speaker,
        )

        sft_dataset.append({
            "prompt": prompt_text,
            "original_completion": base_completion,
            "revised_completion": revised_completion,
            "persona": speaker,
        })

    # save the dataset
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sft_dataset, f, indent=2, ensure_ascii=False)

    print(f"\n Done! ")
    print(f"Successfully generated and saved {len(sft_dataset)} revised completions to {output_path}")

parser = argparse.ArgumentParser(
    description="Generate and revise completions for SFT using a constitution and a teacher model."
)

parser.add_argument("--input_file", type=str, default="test.json", 
                    help="Path to the JSON file containing prompts.")
parser.add_argument("--output_file", type=str, default="sft_revised_dataset.json",
                    help="Path to save the output JSON dataset.")
parser.add_argument("--constitutions_dir", type=str, default="src/rl/constitutions/",
                    help="Directory containing the constitution text files.")
parser.add_argument("--n_samples", type=int, default=-1, 
                    help="Number of samples to process for SFT. Set negative value for all samples.")

args = parser.parse_args()
main(args)