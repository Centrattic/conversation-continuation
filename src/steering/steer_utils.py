import torch

# ToDo: add attention masking. Also move this to gpu for speed
@torch.no_grad()
def generate_steering_vector(model, tokenizer, steer_dict: dict, 
                             alpha: float = 1.0, layer_from_last: int = -1): # dict of {token: weight}

    device = model.device

    steer_tokens = list(steer_dict.keys())
    steer_weights = torch.tensor(list(steer_dict.values()), device=device).unsqueeze(-1)  # [batch, 1]
    # torch.Tensor interesting

    vectors = [0]*len(steer_tokens)
    for i, string in enumerate(steer_tokens):
        inputs = tokenizer(string, return_tensors="pt", truncation=True, 
                            max_length=5).to(device) # short steering prompts, just a few words or so

        prompt_out = model(
            **inputs,
            output_hidden_states=True,
            use_cache=True  # so you can pass past_key_values into manual generate loop if desired
        )

        # Highly recommend trying different layers for this
        prompt_hidden = prompt_out.hidden_states[layer_from_last].to(device)  # [1, prompt_len, H]
        mean_hidden = prompt_hidden.mean(dim=1) # average over sequence length, # [1, H]
        vectors[i] = mean_hidden.squeeze(dim=0) # appending [H] vectors

    # stack list of vectors
    vectors = torch.stack(vectors).to(device) # [num_toks, H]

    total_vector = torch.sum(vectors * steer_weights, dim=0) # [num_toks, H] * [num_toks, 1]
    scaled_vector = alpha*total_vector # [H]

    return scaled_vector.view(1,1,-1) # -1 is inferred direction, could also reshape in generate_with_steering
