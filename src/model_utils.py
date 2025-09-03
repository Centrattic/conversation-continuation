import torch
from torch.nn.functional import softmax
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import io
import requests
from PIL import Image
from src.config import MODEL_CONFIGS, RESULTS_FOLDER_STRUCTURE, BASE_RESULTS_FOLDER

def load_image(img_ref: str): # for vlms
    """img_ref can be a local path or a URL"""
    if img_ref.startswith("http://") or img_ref.startswith("https://"):
        resp = requests.get(img_ref, timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    return Image.open(img_ref).convert("RGB")

# Model configurations
def get_model_config(model_key: str) -> Dict:
    """Get model configuration for a given model key."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model key: {model_key}. Available models: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[model_key]


def get_results_folder(
    model_key: str,
    experiment_name: Optional[str] = None,
) -> Path:
    """Get the results folder path for a model and optional experiment."""
    if model_key not in RESULTS_FOLDER_STRUCTURE:
        raise ValueError(
            f"Unknown model key: {model_key}. Available models: {list(RESULTS_FOLDER_STRUCTURE.keys())}"
        )

    base_folder = Path(RESULTS_FOLDER_STRUCTURE[model_key])
    if experiment_name:
        return base_folder / experiment_name
    return base_folder


def get_lora_adapter_path(
    model_key: str,
    experiment_name: str,
) -> Path:
    """Get the LoRA adapter path for a model and experiment."""
    return get_results_folder(
        model_key,
        experiment_name,
    ) / "lora_adapter"


def get_training_log_path(
    model_key: str,
    experiment_name: str,
) -> Path:
    """Get the training log path for a model and experiment."""
    return get_results_folder(
        model_key,
        experiment_name,
    ) / "log.json"


def create_experiment_folder(
    model_key: str,
    experiment_name: str,
) -> Path:
    """Create and return the experiment folder path."""
    folder = get_results_folder(
        model_key,
        experiment_name,
    )
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )
    return folder


def save_experiment_config(
    model_key: str,
    experiment_name: str,
    config: Dict,
) -> None:
    """Save experiment configuration to a JSON file."""
    folder = get_results_folder(
        model_key,
        experiment_name,
    )
    folder.mkdir(parents=True, exist_ok=True)

    config_path = folder / "experiment_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_experiment_config(
    model_key: str,
    experiment_name: str,
) -> Dict:
    """Load experiment configuration from a JSON file."""
    config_path = get_results_folder(
        model_key,
        experiment_name,
    ) / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Experiment config not found at {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f)


def list_experiments(model_key: str) -> List[str]:
    """List all experiments for a given model."""
    base_folder = get_results_folder(model_key)
    if not base_folder.exists():
        return []

    experiments = []
    for item in base_folder.iterdir():
        if item.is_dir() and (item / "experiment_config.json").exists():
            experiments.append(item.name)

    return sorted(experiments)


def get_latest_experiment(model_key: str) -> Optional[str]:
    """Get the most recent experiment for a given model."""
    experiments = list_experiments(model_key)
    if not experiments:
        return None

    # Sort by modification time
    base_folder = get_results_folder(model_key)
    experiments_with_time = []
    for exp in experiments:
        exp_path = base_folder / exp
        mtime = exp_path.stat().st_mtime
        experiments_with_time.append((
            exp,
            mtime,
        ))

    experiments_with_time.sort(key=lambda x: x[1], reverse=True)
    return experiments_with_time[0][0]


# Generation functions (keeping existing functionality)
@torch.no_grad()
def prepare_generation_inputs(
    model,
    tokenizer,
    prompt: str,
    *,
    processor=None,
    max_length: int = 4096,
):
    """Build model inputs for generation.

    - Uses processor.apply_chat_template for VLMs when provided
    - Else, uses tokenizer's chat_template if available
    - Falls back to plain text tokenization

    Returns a tuple: (model_inputs_dict, input_length)
    """
    prompt = prompt.strip()
    try:
        if processor is not None:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            proc_out = processor.apply_chat_template(
                conversation=messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
            model_inputs = {k: v.to(model.device) for k, v in proc_out.items()}
        else:
            chat_tmpl = getattr(tokenizer, "chat_template", None)
            if isinstance(chat_tmpl, str) and len(chat_tmpl) > 0:
                messages = [{"role": "user", "content": prompt}]
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template=chat_tmpl,
                )
                model_inputs = tokenizer(
                    formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                ).to(model.device)
            else:
                model_inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                ).to(model.device)
    except Exception:
        model_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)

    try:
        input_ids = model_inputs.input_ids
    except AttributeError:
        input_ids = model_inputs["input_ids"]
    input_length = input_ids.shape[1]
    return model_inputs, input_length

@torch.no_grad()
def generate(
    model,
    prompt: str,
    tokenizer,
    max_new_tokens=50,
    processor=None,
):
    model_inputs, input_length = prepare_generation_inputs(
        model, tokenizer, prompt, processor=processor
    )
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=0,
        pad_token_id=tokenizer.eos_token_id,
    )

    full_ids = outputs[0]
    gen_ids = full_ids[input_length:]
    out_text = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
    for control_tok in ("<bos>", "<eot>", "<eot_id>", "<end_of_turn>"):
        out_text = out_text.replace(control_tok, "").strip()
    return out_text


@torch.no_grad()
def generate_with_ppl(
    model,
    prompt: str,
    tokenizer,
    max_new_tokens=50,
):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(model.device)

    input_length = inputs.input_ids.shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=0,
        pad_token_id=tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
    )

    full_ids = outputs.sequences[0]
    gen_ids = full_ids[input_length:]
    scores = outputs.scores

    neg_log_probs = []
    for idx, token_id in enumerate(gen_ids):
        logits = scores[idx][0]
        probs = softmax(logits, dim=-1)
        token_prob = probs[token_id]
        neg_log_prob = -torch.log(token_prob + 1e-12)
        neg_log_probs.append(neg_log_prob)

    avg_nll = torch.stack(neg_log_probs).mean()

    # Perplexity = exp(average negative log-probability)
    perplexity = torch.exp(avg_nll)

    out_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return out_text, perplexity.item()


@torch.no_grad()
def generate_with_activations(
    model,
    prompt: str,
    tokenizer,
    max_new_tokens=50,
):
    inputs = tokenizer(prompt,
                       return_tensors="pt",
                       truncation=True,
                       max_length=1024).to(model.device)

    activations = []

    def hook(module, hook_inputs):
        # hook_inputs[0] is the final hidden states [B, seq_len, hidden]
        activations.append(hook_inputs[0].detach().cpu())

    h = model.lm_head.register_forward_pre_hook(
        hook)  # captures input to lm_head

    # activations[0] is shape (1,prompt_len,hidden_size)
    # activations[1] is shape (1, prompt_len+1, hidden_size)
    # etc.

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,  # beam search
        temperature=0.7,
        top_p=0.95,
        top_k=0,
        pad_token_id=tokenizer.eos_token_id,
    )

    # remove hook
    h.remove()
    out_text = tokenizer.decode(outputs[0]).replace(prompt, "").strip()
    return out_text, activations


@torch.no_grad()
def generate_with_steering(
    model,
    prompt,
    tokenizer,
    steering_vector,
    max_new_tokens=50,
    layer_from_last=-1,
):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(model.device)

    def add_vector_hook(module, input, output):  # what does input do here
        # output shape: [batch_size, seq_len, hidden_size]
        # add vector to each token's activation but vector is [1, 1, hidden_size]
        # print("TESTING", output[0].shape, steering_vector.shape)
        hidden_states = output[0]

        # only steer when seq_len == 1 (i.e. a generation step, not at prompt embedding step)
        # or maybe should steer at both?
        if hidden_states.shape[1] == 1:
            hidden_states += steering_vector.to(hidden_states.device)
            return (hidden_states, ) + output[1:]
        else:
            return output  # leave prompt pass untouched

    # To view visible layers: {for name, module in model.named_modules(): print(name)}

    h = model.model.model.layers[layer_from_last].register_forward_hook(
        add_vector_hook)
    # this model outputs something like (output_hidden_states, other_outputs...) I guess?

    # If I add activation hook here as well, could do TDA with steering
    outputs = model.generate(  # first forward pass of generate ingests prompt
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=0,
        pad_token_id=tokenizer.eos_token_id,
    )

    # remove hook
    h.remove()
    out_text = tokenizer.decode(outputs[0]).replace(prompt, "").strip()
    return out_text
