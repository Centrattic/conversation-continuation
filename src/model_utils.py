import torch
from torch.nn.functional import softmax
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
import json
from datetime import datetime
import io
import requests
import threading
from PIL import Image
from transformers import TextIteratorStreamer
from src.config import MODEL_CONFIGS, RESULTS_FOLDER_STRUCTURE, BASE_RESULTS_FOLDER, INSTRUCT_SYSTEM_PROMPT, RIYA_SPEAKER_TOKEN, FRIEND_SPEAKER_TOKEN, RIYA_NAME, FRIEND_NAME


def load_image(img_ref: str):  # for vlms
    """img_ref can be a local path or a URL"""
    if img_ref.startswith("http://") or img_ref.startswith("https://"):
        resp = requests.get(img_ref, timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    return Image.open(img_ref).convert("RGB")


def remove_end_of_turn_token(model_inputs, input_length, tokenizer):
    """Remove the <end_of_turn> token from input_ids to allow continuous generation."""
    end_of_turn_token_id = tokenizer.encode("<end_of_turn>",
                                            add_special_tokens=False)[0]
    input_ids = model_inputs["input_ids"][0]

    # Find and remove the <end_of_turn> token
    if end_of_turn_token_id in input_ids:
        # Find the last occurrence of <end_of_turn> and remove it
        end_of_turn_positions = (input_ids == end_of_turn_token_id).nonzero(
            as_tuple=True)[0]
        if len(end_of_turn_positions) > 0:
            last_end_of_turn_pos = end_of_turn_positions[-1].item()
            # Remove the <end_of_turn> token with proper bounds checking
            if last_end_of_turn_pos == 0:
                # Token is at the beginning
                new_input_ids = input_ids[1:]
            elif last_end_of_turn_pos == len(input_ids) - 1:
                # Token is at the end
                new_input_ids = input_ids[:-1]
            else:
                # Token is in the middle
                new_input_ids = torch.cat([
                    input_ids[:last_end_of_turn_pos],
                    input_ids[last_end_of_turn_pos + 1:]
                ])

            model_inputs["input_ids"] = new_input_ids.unsqueeze(0)

            # Update attention mask to match the new length
            attention_mask = model_inputs["attention_mask"][0]
            if last_end_of_turn_pos == 0:
                new_attention_mask = attention_mask[1:]
            elif last_end_of_turn_pos == len(attention_mask) - 1:
                new_attention_mask = attention_mask[:-1]
            else:
                new_attention_mask = torch.cat([
                    attention_mask[:last_end_of_turn_pos],
                    attention_mask[last_end_of_turn_pos + 1:]
                ])

            model_inputs["attention_mask"] = new_attention_mask.unsqueeze(0)

            # Update input_length
            input_length -= 1

            formatted_text = tokenizer.decode(model_inputs["input_ids"][0],
                                              skip_special_tokens=False)

            print("🔍 DEBUG removed <end_of_turn> token")
            print(formatted_text)
            print(f"  New input length: {input_length}")
            print("=" * 50)

    return model_inputs, input_length


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
    deployment=True,
):
    """Build model inputs for generation.

    - Uses processor.apply_chat_template for VLMs when provided
    - For instruct models: formats with chat template and forces next speaker token
    - For base models: handles conversation history parsing and adds target speaker token

    Returns a tuple: (model_inputs_dict, input_length)
    """
    prompt = prompt.strip()

    # For instruct models, format the prompt using chat template
    if is_instruct:
        messages = format_instruct_prompt(prompt,
                                          target_speaker=target_speaker,
                                          deployment=deployment)

        # Use processor if available, otherwise fall back to tokenizer
        if processor is not None and hasattr(processor, 'apply_chat_template'):
            proc_out = processor.apply_chat_template(
                conversation=messages,
                add_generation_prompt=False,
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
    target_speaker: str = None,
    stop_tokens: Dict[str, str] = None,
) -> List[str]:
    """Process the generated text output, handling both model types.
    
    Always returns a list of messages from the target speaker.
    """
    # Clean up control tokens
    for control_tok in ("<bos>", "<eot>", "<eot_id>", "<end_of_turn>", "<s>",
                        "</s>"):
        generated_text = generated_text.replace(control_tok, "").strip()

    # Always cut off at the next speaker token if stop_tokens provided
    if stop_tokens:
        generated_text = truncate_to_next_speaker(
            generated_text,
            expected_stop=stop_tokens["expected_stop"],
            other_stop=stop_tokens["other_stop"])
        generated_text = generated_text.lstrip(" :")

    if target_speaker:
        target_speaker_token = (RIYA_SPEAKER_TOKEN if target_speaker
                                == RIYA_NAME else FRIEND_SPEAKER_TOKEN)

        generated_speaker_text = target_speaker_token + generated_text

        print(f"🔍 DEBUG: generated_speaker_text: {generated_speaker_text}")

        messages = [
            part.strip()
            for part in generated_speaker_text.split(target_speaker_token)
            if part.strip()
        ]

        return messages

    return [generated_text.strip()]


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
    temperature=0.7,
    top_p=0.95,
    top_k=0,
) -> List[str] | str:
    """Generate text, handling both base and instruct models."""
    model_inputs, input_length = prepare_generation_inputs(
        model,
        tokenizer,
        prompt,
        processor=processor,
        is_instruct=is_instruct,
        target_speaker=target_speaker,
    )

    if is_instruct:
        formatted_text = tokenizer.decode(model_inputs["input_ids"][0],
                                          skip_special_tokens=False)
        # Remove <end_of_turn> token from the input_ids before generation
        if deployment:
            model_inputs, input_length = remove_end_of_turn_token(
                model_inputs,
                input_length,
                tokenizer,
            )

    generation_kwargs = {
        **model_inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "pad_token_id": tokenizer.eos_token_id,
        "output_attentions": False,
        "output_hidden_states": False,
        "return_dict_in_generate": False,
    }

    outputs = model.generate(**generation_kwargs)

    full_ids = outputs[0]
    gen_ids = full_ids[input_length:]
    out_text = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()

    print("🔍 DEBUG generate (instruct):")
    print(f"  Out text: {out_text}")
    print("=" * 50)

    if target_speaker:
        stop_tokens = get_stop_tokens_for_speaker(target_speaker)

    if deployment:
        messages = process_generation_output(
            out_text,
            target_speaker,
            stop_tokens,
        )
        return messages
    else:
        return out_text


@torch.no_grad()
def stream_generate(
    model,
    prompt: str,
    tokenizer,
    max_new_tokens=50,
    processor=None,
    is_instruct=False,
    target_speaker=None,
    deployment=True,
    temperature=0.7,
    top_p=0.95,
    top_k=0,
) -> Iterator[str]:
    """Stream text generation with proper stop token handling.
    
    - Stops when it encounters stop tokens like [Owen], [O, etc.
    - When it encounters [Riya] tokens, moves to new line instead of outputting
    - Yields text chunks as they are generated
    """
    model.eval()

    model_inputs, input_length = prepare_generation_inputs(
        model,
        tokenizer,
        prompt,
        processor=processor,
        is_instruct=is_instruct,
        target_speaker=target_speaker,
    )

    if is_instruct and deployment:
        model_inputs, input_length = remove_end_of_turn_token(
            model_inputs, input_length, tokenizer)

    # Streamer that handles detokenization efficiently
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
        # Useful to avoid collapsing spaces or quotes mid-stream:
        decode_kwargs=dict(clean_up_tokenization_spaces=False))

    generation_kwargs = {
        **model_inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": False,
        "streamer": streamer,
        "output_attentions": False,
        "output_hidden_states": False,
    }

    # Run generate on a background thread; the streamer yields chunks here.
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    print(f"🎯 Generation thread started for streaming steering")

    # Buffer to accumulate text and check for stop tokens
    buffer = ""
    num_toks = 0
    skip_mode = False

    try:
        for new_text in streamer:
            # print(f"🔍 DEBUG stream_generate raw output: '{new_text}'")

            buffer = new_text  # buffer only one token at a time
            # print("\nbuffer each time",buffer)

            # Clean up control tokens from the buffer
            has_control_tok = False
            for control_tok in ("<bos>", "<s>", "</s>", "<start_of_turn>",
                                "user", "assistant"):
                if control_tok in buffer:
                    has_control_tok = True
                    break
            if has_control_tok:
                continue

            # print(f"🔍 DEBUG stream_generate buffer: '{buffer}'")

            # Handle skip mode - skip all tokens until we find target_speaker token
            if skip_mode:
                print(f"skip mode - buffer: '{buffer}'")
                if f"[{target_speaker}]" in buffer or f"[{target_speaker[:1]}]" in buffer:
                    # Found target speaker token, exit skip mode
                    print(f"exiting skip mode - found target speaker token")
                    skip_mode = False
                    yield "\n"
                    continue
                else:  # skip this token
                    print(f"skipping token: '{buffer}'")
                    continue

            # Check for stop tokens that should end generation
            patterns = [
                f"[{FRIEND_NAME}]",  # [Owen]
                f"[{FRIEND_NAME[:1]}",  # [O]
                f"[{RIYA_NAME}]",  # [Riya] - should move to new line
                f"[{RIYA_NAME[:1]}",  # [R - should move to new line
                "<eot>",
                "<eot_id>",
                "<end_of_turn>",
            ]

            # Check if any stop or newline pattern is in the cleaned buffer
            pattern_found = False
            for pattern in patterns:
                if pattern in buffer:
                    print(f"🔍 FOUND PATTERN: '{pattern}' in buffer '{buffer}'")
                    pattern_found = True
                    if num_toks == 0 and pattern in [
                            f"[{target_speaker}]", f"[{target_speaker[:1]}"
                    ]:
                        # print("top",buffer)
                        break
                    # If it's target speaker token, yield newline instead of the token
                    elif pattern in [
                            f"[{target_speaker}]", f"[{target_speaker[:1]}"
                    ]:
                        # print("middle",buffer)
                        yield f"[{target_speaker}]\n"
                        break
                    # not great but better than the other option... to have no output
                    elif num_toks == 0:  # skip the stop token if no riya messages yet
                        # print("bottom",buffer)
                        # print(f"entering skip mode - found non-target speaker token: '{pattern}'")
                        skip_mode = True
                        break
                    else:
                        return

            # If we found a pattern and processed it, skip the text yielding
            if pattern_found:
                continue

            # If no stop pattern found, yield the cleaned text and clear buffer
            # print("generating output 1")
            if buffer.strip():
                # print(f"🔍 YIELDING TEXT: '{buffer}'")
                yield buffer
                num_toks += 1

    except Exception as e:
        print(f"Error in stream_generate: {e}")
        torch.cuda.empty_cache()
        yield f"[Error: {str(e)}]"


@torch.no_grad()
def stream_generate_steer(
    model,
    prompt: str,
    tokenizer,
    steering_vector,
    max_new_tokens=50,
    layer_from_last=-1,
    processor=None,
    is_instruct=False,
    target_speaker=None,
    deployment=True,
    temperature=0.7,
    top_p=0.95,
    top_k=0,
) -> Iterator[str]:
    """Stream text generation with steering and proper stop token handling.
    
    - Combines steering functionality with streaming
    - Stops when it encounters stop tokens like [Owen], [O, etc.
    - When it encounters [Riya] tokens, moves to new line instead of outputting
    - Yields text chunks as they are generated
    """
    model.eval()

    model_inputs, input_length = prepare_generation_inputs(
        model,
        tokenizer,
        prompt,
        processor=processor,
        is_instruct=is_instruct,
        target_speaker=target_speaker,
    )

    if is_instruct and deployment:
        model_inputs, input_length = remove_end_of_turn_token(
            model_inputs, input_length, tokenizer)

    # Streamer that handles detokenization efficiently
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
        # Useful to avoid collapsing spaces or quotes mid-stream:
        decode_kwargs=dict(clean_up_tokenization_spaces=False))

    # Steering hook function
    def add_vector_hook(module, input, output):
        # output shape: [batch_size, seq_len, hidden_size]
        # add vector to each token's activation but vector is [1, 1, hidden_size]
        hidden_states = output[0]

        print(f"🎯 STEERING HOOK CALLED: hidden_states.shape={hidden_states.shape}")
        # only steer when seq_len == 1 (i.e. a generation step, not at prompt embedding step)
        if hidden_states.shape[1] == 1:
            # Check if steering vector is actually non-zero
            steering_magnitude = steering_vector.abs().sum().item()
            steering_max = steering_vector.abs().max().item()
            print(f"🎯 APPLYING STEERING: steering_magnitude={steering_magnitude:.6f}, steering_max={steering_max:.6f}")
            hidden_states += steering_vector.to(hidden_states.device)
            print(f"🎯 STEERING APPLIED SUCCESSFULLY")
            return (hidden_states, ) + output[1:]
        else:
            print(f"🎯 PROMPT STEP: skipping steering (seq_len={hidden_states.shape[1]})")
            return output  # leave prompt pass untouched

    # Check if steering vector is actually non-zero before registering hook
    steering_magnitude = steering_vector.abs().sum().item()
    print(f"Steering vector magnitude: {steering_magnitude:.6f}")

    # Register steering hook
    model_name = getattr(model.config, "_name_or_path", "")
    print(f"🎯 Registering steering hook for model: {model_name}")
    print(f"🎯 Using layer_from_last: {layer_from_last}")
    
    if "gemma-3-27b-it" in model_name.lower():
        h = model.language_model.layers[layer_from_last].register_forward_hook(
            add_vector_hook)
        print(f"🎯 Hook registered on Gemma3 layer: {layer_from_last}")
    else:  # mistral
        h = model.model.model.layers[layer_from_last].register_forward_hook(
            add_vector_hook)

    generation_kwargs = {
        **model_inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": False,
        "streamer": streamer,
        "output_attentions": False,
        "output_hidden_states": False,
    }

    # Run generate on a background thread; the streamer yields chunks here.
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    print(f"🎯 Generation thread started for streaming steering")

    # Buffer to accumulate text and check for stop tokens
    buffer = ""
    num_toks = 0
    skip_mode = False

    try:
        for new_text in streamer:
            # Debug: Print raw model output before filtering
            print(f"🔍 DEBUG stream_generate_steer raw output: '{new_text}'")

            buffer = new_text  # buffer only one token at a time

            # Clean up control tokens from the buffer
            has_control_tok = False
            for control_tok in ("<bos>", "<s>", "</s>", "<start_of_turn>",
                                "user", "assistant"):
                if control_tok in buffer:
                    has_control_tok = True
                    break
            if has_control_tok:
                continue

            # Debug: Print buffer state
            print(f"🔍 DEBUG stream_generate_steer buffer: '{buffer}'")

            # Handle skip mode - skip all tokens until we find target_speaker token
            if skip_mode:
                print(f"skip mode - buffer: '{buffer}'")
                if f"[{target_speaker}]" in buffer or f"[{target_speaker[:1]}]" in buffer:
                    # Found target speaker token, exit skip mode
                    print(f"exiting skip mode - found target speaker token")
                    skip_mode = False
                    yield "\n"
                    continue
                else:  # skip this token
                    print(f"skipping token: '{buffer}'")
                    continue

            # Check for stop tokens that should end generation
            patterns = [
                f"[{FRIEND_NAME}]",  # [Owen]
                f"[{FRIEND_NAME[:1]}",  # [O]
                f"[{RIYA_NAME}]",  # [Riya] - should move to new line
                f"[{RIYA_NAME[:1]}",  # [R - should move to new line
                "<eot>",
                "<eot_id>",
                "<end_of_turn>",
            ]

            # Check if any stop or newline pattern is in the cleaned buffer
            pattern_found = False
            for pattern in patterns:
                if pattern in buffer:
                    print(f"🔍 FOUND PATTERN: '{pattern}' in buffer '{buffer}'")
                    pattern_found = True
                    if num_toks == 0 and pattern in [
                            f"[{target_speaker}]", f"[{target_speaker[:1]}"
                    ]:
                        # print("top",buffer)
                        break
                    # If it's target speaker token, yield newline instead of the token
                    elif pattern in [
                            f"[{target_speaker}]", f"[{target_speaker[:1]}"
                    ]:
                        # print("middle",buffer)
                        yield f"[{target_speaker}]\n"
                        break
                    # not great but better than the other option... to have no output
                    elif num_toks == 0:  # skip the stop token if no riya messages yet
                        # print("bottom",buffer)
                        # print(f"entering skip mode - found non-target speaker token: '{pattern}'")
                        skip_mode = True
                        break
                    else:
                        return

            # If we found a pattern and processed it, skip the text yielding
            if pattern_found:
                continue

            # If no stop pattern found, yield the cleaned text and clear buffer
            # print("generating output 1")
            if buffer.strip():
                # print(f"🔍 YIELDING TEXT: '{buffer}'")
                yield buffer
                num_toks += 1

    except Exception as e:
        print(f"Error in stream_generate_steer: {e}")
        torch.cuda.empty_cache()
        yield f"[Error: {str(e)}]"

    finally:
        # Always remove the hook when done
        if h is not None:
            print(f"🎯 Removing steering hook")
            h.remove()
        else:
            print(f"🎯 No hook to remove (h was None)")


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

    out_text = process_generation_output(out_text, target_speaker, stop_tokens)
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
        target_speaker,
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
    deployment=True,
    processor=None,
):
    """Generate text with steering, handling both base and instruct models."""
    model_inputs, input_length = prepare_generation_inputs(
        model,
        tokenizer,
        prompt,
        processor=processor,
        is_instruct=is_instruct,
        target_speaker=target_speaker,
        deployment=deployment,
    )

    # Handle instruct model specific processing
    if is_instruct and deployment:
        model_inputs, input_length = remove_end_of_turn_token(
            model_inputs,
            input_length,
            tokenizer,
        )

    # Get stop tokens for the target speaker
    if target_speaker:
        stop_tokens = get_stop_tokens_for_speaker(target_speaker)
    else:
        stop_tokens = None

    def add_vector_hook(module, input, output):  # what does input do here
        # output shape: [batch_size, seq_len, hidden_size]
        # add vector to each token's activation but vector is [1, 1, hidden_size]
        hidden_states = output[0]

        print(f"Hook called: hidden_states.shape={hidden_states.shape}")
        # only steer when seq_len == 1 (i.e. a generation step, not at prompt embedding step)
        if hidden_states.shape[1] == 1:
            # Check if steering vector is actually non-zero
            steering_magnitude = steering_vector.abs().sum().item()
            steering_max = steering_vector.abs().max().item()
            print(
                f"Generation step: steering_magnitude={steering_magnitude:.6f}, steering_max={steering_max:.6f}"
            )
            hidden_states += steering_vector.to(hidden_states.device)
            print(f"🎯 STEERING APPLIED SUCCESSFULLY (NON-STREAM)")
            return (hidden_states, ) + output[1:]
        else:
            print(f"Prompt step: skipping steering")
            return output  # leave prompt pass untouched

    # Check if steering vector is actually non-zero before registering hook
    steering_magnitude = steering_vector.abs().sum().item()
    print(f"Steering vector magnitude: {steering_magnitude:.6f}")

    model_name = getattr(model.config, "_name_or_path", "")
    if "gemma-3-27b-it" in model_name.lower():
        h = model.language_model.layers[layer_from_last].register_forward_hook(
            add_vector_hook)
    else:  # mistral
        h = model.model.model.layers[layer_from_last].register_forward_hook(
            add_vector_hook)

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

    # remove hook only if it was registered
    if h is not None:
        h.remove()

    # Extract only the generated tokens (same as regular generate function)
    full_ids = outputs[0]
    gen_ids = full_ids[input_length:]
    out_text = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()

    return process_generation_output(
        out_text,
        target_speaker,
        stop_tokens,
    )


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
        cut_off_speaker_token = RIYA_SPEAKER_TOKEN[:2]

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
                           target_speaker: str = None,
                           deployment: bool = False) -> str:
    """Format prompt for instruct models using the proper chat template."""
    if system_prompt is None:
        system_prompt = INSTRUCT_SYSTEM_PROMPT.format(
            FRIEND_SPEAKER_TOK=FRIEND_SPEAKER_TOKEN,
            RIYA_SPEAKER_TOK=RIYA_SPEAKER_TOKEN)

    target_token = RIYA_SPEAKER_TOKEN if target_speaker == RIYA_NAME else FRIEND_SPEAKER_TOKEN

    inputs = f"\n{conversation_history.strip()}\n\nHow does the conversation continue?"

    messages = [{
        "role": "system",
        "content": [{
            "type": "text",
            "text": system_prompt
        }]
    }, {
        "role": "user",
        "content": [{
            "type": "text",
            "text": inputs
        }]
    }]

    # add target speaker token to assistant role
    if deployment and target_speaker:
        messages.append({
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": target_token
            }]
        })

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
