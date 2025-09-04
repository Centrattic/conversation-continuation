import torch
from src.model_utils import prepare_generation_inputs


@torch.no_grad()
def generate_steering_vector(
        model,
        tokenizer,
        steer_dict: dict,
        pos_alpha: float = 1.0,
        neg_alpha: float = 1.0,
        layer_from_last: int = -1,
        processor=None,
        is_instruct=False,
        current_speaker=None):  # dict of {token: weight}

    device = model.device
    pos_vectors, neg_vectors = [], []

    # Get model dtype for consistency
    model_dtype = next(model.parameters()).dtype
    
    # Ensure model is in eval mode and on the correct device/dtype
    model.eval()
    
    for text, weight in steer_dict.items():
        # Prepend speaker name to steering prompt if current_speaker is provided
        if current_speaker:
            formatted_text = f"[{current_speaker}] {text}"
        else:
            formatted_text = text
        
        # Use the same preprocessing as regular generation
        model_inputs, _ = prepare_generation_inputs(
            model,
            tokenizer,
            formatted_text,
            processor=processor,
            is_instruct=is_instruct,
            target_speaker=None,  # No target speaker for steering prompts
            deployment=True,
        )

        prompt_out = model(
            **model_inputs,
            output_hidden_states=True,
            use_cache=True  # so you can pass past_key_values into manual generate loop if desired
        )

        prompt_hidden = prompt_out.hidden_states[layer_from_last].to(device).to(model_dtype)  # [1, prompt_len, H]
        mean_hidden = prompt_hidden.mean(dim=1).squeeze(dim=0)  # average over sequence length, # [1, H] -> [H]

        if weight > 0:
            pos_vectors.append(mean_hidden * abs(weight))
        else:
            neg_vectors.append(mean_hidden * abs(weight))

    # stack list of vectors
    # assuming pos & neg vectors always exist
    avg_pos_vector = torch.stack(pos_vectors).mean(dim=0).to(model_dtype)  # [num_prompts, H] -> [H]
    avg_neg_vector = torch.stack(neg_vectors).mean(dim=0).to(model_dtype)  # [num_prompts, H] -> [H]

    total_vector = (pos_alpha * avg_pos_vector) - (neg_alpha * avg_neg_vector)
    total_vector = total_vector.to(model_dtype)

    # [1,1,H]
    return total_vector.view(
        1, 1, -1
    )  # -1 is inferred direction, could also reshape in generate_with_steering
