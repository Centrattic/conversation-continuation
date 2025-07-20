import torch

# ToDo: add attention masking. Also move this to gpu for speed
@torch.no_grad()
def generate_steering_vector(model, tokenizer, steer_dict: dict, 
                             alpha: float = 1.0, layer_from_last: int = -1): # dict of {token: weight}

    device = model.device

    steer_tokens = list(steer_dict.keys())
    steer_weights = torch.tensor(list(steer_dict.values()), device=device).unsqueeze(-1)  # [batch, 1]
    # torch.Tensor interesting

    inputs = "".join(steer_tokens)
    inputs = tokenizer(inputs, return_tensors="pt", truncation=True, 
                        max_length=5).to(device) # short steering prompts, just a few words or so

    prompt_out = model(
        **inputs,
        output_hidden_states=True,
        use_cache=True  # so you can pass past_key_values into manual generate loop if desired
    )

    # Highly recommend trying different layers for this
    prompt_hidden = prompt_out.hidden_states[layer_from_last].to(device)  # [batch, prompt_len, hidden_size]

    mean_hidden = prompt_hidden.mean(dim=1) # average over sequence length
    weights = steer_weights / steer_weights

    weighted_avg = torch.sum(mean_hidden * weights, dim=0) # [batch, H] * [batch, 1]
    scaled_vector = alpha*weighted_avg # [H]

    return scaled_vector.view(1,1,-1) # -1 is inferred direction, could also reshape in generate_with_steering
