import torch
from torch.nn.functional import softmax
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import io
import requests
from PIL import Image
from src.config import MODEL_CONFIGS, RESULTS_FOLDER_STRUCTURE, BASE_RESULTS_FOLDER, INSTRUCT_SYSTEM_PROMPT, RIYA_SPEAKER_TOKEN, FRIEND_SPEAKER_TOKEN, RIYA_NAME, FRIEND_NAME


def load_image(img_ref: str):  # for vlms
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
    is_instruct=False,
    target_speaker=None,
):
    """Build model inputs for generation.

    - Uses processor.apply_chat_template for VLMs when provided
    - Else, uses tokenizer's chat_template if available
    - Falls back to plain text tokenization
    - For instruct models: formats with chat template and forces next speaker token
    - For base models: handles conversation history parsing and returns stop tokens

    Returns a tuple: (model_inputs_dict, input_length, stop_tokens_dict)
    """
    prompt = prompt.strip()

    # For instruct models, format the prompt using chat template
    if is_instruct:
        messages = format_instruct_prompt(prompt,
                                          target_speaker=target_speaker)

        # Use processor if available, otherwise fall back to tokenizer
        if processor is not None and hasattr(processor, 'apply_chat_template'):
            print(f"✅ Using processor: {type(processor)}")
            proc_out = processor.apply_chat_template(
                conversation=messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
            model_inputs = {k: v.to(model.device) for k, v in proc_out.items()}
        else:
            raise ValueError("No processor available")
        
        # debug
        print("🔍 DEBUG prepare_generation_inputs (instruct):")
        print(f"  Messages: {messages}")
        print(f"  Model inputs: {model_inputs}")
        print("=" * 50)
    else:
        # For base models, add the target speaker token to the prompt
        if target_speaker:
            target_token = RIYA_SPEAKER_TOKEN if target_speaker == RIYA_NAME else FRIEND_SPEAKER_TOKEN
            prompt = f"{prompt}\n{target_token}"

        # DEBUG: Print base model prompt
        print("🔍 DEBUG prepare_generation_inputs (base):")
        print(f"  Target speaker: {target_speaker}")
        print(f"  Final prompt: {repr(prompt)}")
        print("=" * 50)

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


def process_generation_output(
    generated_text: str,
    is_instruct: bool = False,
    stop_tokens: Dict[str, str] = None,
) -> str:
    """Process the generated text output, handling both model types."""
    # Clean up control tokens
    for control_tok in ("<bos>", "<eot>", "<eot_id>", "<end_of_turn>"):
        generated_text = generated_text.replace(control_tok, "").strip()

    # For both base and instruct models, truncate at next speaker and clean up
    if stop_tokens:
        generated_text = truncate_to_next_speaker(
            generated_text,
            expected_stop=stop_tokens["expected_stop"],
            other_stop=stop_tokens["other_stop"])
        generated_text = generated_text.lstrip(" :")

    return generated_text


@torch.no_grad()
def generate(
    model,
    prompt: str,
    tokenizer,
    max_new_tokens=50,
    processor=None,
    is_instruct=False,
    target_speaker=None,
    deployment=True,
):
    """Generate text, handling both base and instruct models."""
    model_inputs, input_length = prepare_generation_inputs(
        model,
        tokenizer,
        prompt,
        processor=processor,
        is_instruct=is_instruct,
        target_speaker=target_speaker,
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

    print("🔍 DEBUG generate (instruct):")
    print(f"  Outputs: {outputs}")
    print("=" * 50)

    full_ids = outputs[0]
    gen_ids = full_ids[input_length:]
    out_text = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()

    print("🔍 DEBUG generate (instruct):")
    print(f"  Out text: {out_text}")
    print("=" * 50)

    if target_speaker:
        stop_tokens = get_stop_tokens_for_speaker(target_speaker)

    if deployment:
        return_out_text = process_generation_output(
            out_text,
            is_instruct,
            stop_tokens,
        )
    else:
        return_out_text = out_text
    print("🔍 DEBUG generate (instruct):")
    print(f"  Return out text: {return_out_text}")
    print("=" * 50)

    return return_out_text

@torch.no_grad()
def generate_with_ppl(
    model,
    prompt: str,
    tokenizer,
    max_new_tokens=50,
    is_instruct=False,
    target_speaker=None,
):
    """Generate text with perplexity calculation, handling both base and instruct models."""
    model_inputs, input_length, stop_tokens = prepare_generation_inputs(
        model,
        tokenizer,
        prompt,
        is_instruct=is_instruct,
        target_speaker=target_speaker,
    )

    outputs = model.generate(
        **model_inputs,
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

    out_text = process_generation_output(out_text, is_instruct, stop_tokens)
    return out_text, perplexity.item()


@torch.no_grad()
def generate_with_activations(
    model,
    prompt: str,
    tokenizer,
    max_new_tokens=50,
    is_instruct=False,
    target_speaker=None,
):
    """Generate text with activation tracking, handling both base and instruct models."""
    model_inputs, input_length, stop_tokens = prepare_generation_inputs(
        model,
        tokenizer,
        prompt,
        is_instruct=is_instruct,
        target_speaker=target_speaker)

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
        **model_inputs,
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

    out_text = process_generation_output(
        out_text,
        is_instruct,
        stop_tokens,
    )
    return out_text, activations


@torch.no_grad()
def generate_with_steering(
    model,
    prompt,
    tokenizer,
    steering_vector,
    max_new_tokens=50,
    layer_from_last=-1,
    is_instruct=False,
    target_speaker=None,
):
    """Generate text with steering, handling both base and instruct models."""
    model_inputs, input_length, stop_tokens = prepare_generation_inputs(
        model,
        tokenizer,
        prompt,
        is_instruct=is_instruct,
        target_speaker=target_speaker,
    )

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
        **model_inputs,
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

    return process_generation_output(out_text, is_instruct, stop_tokens)


def get_stop_tokens_for_speaker(target_speaker: str) -> Dict[str, str]:
    """Get stop tokens for a given target speaker.
    
    Args:
        target_speaker: The speaker who should respond next (RIYA_NAME or FRIEND_NAME)
    
    Returns:
        Dict with 'expected_stop' and 'other_stop' tokens
        - expected_stop: The other speaker (when to stop)
        - other_stop: cut off version of speaker tokens for base models that are bad at formatting
    """
    if target_speaker == RIYA_NAME:  # stop when Friend starts speaking
        other_speaker_token = FRIEND_SPEAKER_TOKEN
        cut_off_speaker_token = FRIEND_SPEAKER_TOKEN[:2]
    else:  # stop when Riya starts speaking
        other_speaker_token = RIYA_SPEAKER_TOKEN
        cut_off_speaker_token = FRIEND_SPEAKER_TOKEN[:2]

    return {
        "expected_stop":
        other_speaker_token,  # Stop when the other speaker starts
        "other_stop": cut_off_speaker_token  # Same as expected_stop
    }


def detect_model_type(model_key: str) -> str:
    """Detect if a model is instruct or base type."""
    if model_key in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_key].get("model_type", "base")
    return "base"


def format_instruct_prompt(conversation_history: str,
                           system_prompt: str = None,
                           target_speaker: str = None) -> str:
    """Format prompt for instruct models using the proper chat template."""
    if system_prompt is None:
        system_prompt = INSTRUCT_SYSTEM_PROMPT.format(
            FRIEND_SPEAKER_TOK=FRIEND_SPEAKER_TOKEN,
            RIYA_SPEAKER_TOK=RIYA_SPEAKER_TOKEN)
    
    # Format exactly like in training:
    # 1. System prompt
    # 2. Conversation history + "How does the conversation continue?"
    # 3. Assistant response with target speaker token

    messages = [{
        "role": "system",
        "content": [{
            "type": "text",
            "text": system_prompt
        }]
    }, {
        "role":
        "user",
        "content": [{
            "type":
            "text",
            "text":
            f"{conversation_history.strip()}\n\nHow does the conversation continue?"
        }]
    }]

    # # Add assistant role with target speaker token if specified
    # if target_speaker:
    #     target_token = RIYA_SPEAKER_TOKEN if target_speaker == RIYA_NAME else FRIEND_SPEAKER_TOKEN
    #     messages.append({
    #         "role": "assistant",
    #         "content": [{
    #             "type": "text",
    #             "text": target_token
    #         }]
    #     })

    # DEBUG: Print the formatted messages
    print("🔍 DEBUG format_instruct_prompt:")
    print(f"  Target speaker: {target_speaker}")
    print(f"  Conversation history: {conversation_history.strip()}")
    print(f"  System prompt: {system_prompt}")
    print(f"  Messages: {messages}")
    print("=" * 50)

    return messages


def truncate_to_next_speaker(text: str, expected_stop: str,
                             other_stop: str) -> str:
    """Cut model output at first sign of the next speaker.
    Looks for multiple patterns: [Name], [Name], Name:, Name : (case-insensitive), and newlines.
    """
    candidates: List[int] = []
    lowers = text.lower()
    candidates.append(text.find(expected_stop))
    candidates.append(text.find(other_stop))
    for token in [
            f"{expected_stop.strip('[]')}:",
            f"{expected_stop.strip('[]')} :",
            f"{other_stop.strip('[]')}:",
            f"{other_stop.strip('[]')} :",
    ]:
        i = lowers.find(token.lower())
        candidates.append(i)
    i_colon = text.find(" : ")
    candidates.append(i_colon)
    candidates.append(text.find("\n["))
    cut = min([i for i in candidates if i is not None and i >= 0], default=-1)
    if cut >= 0:
        return text[:cut]
    return text
