import torch

@torch.no_grad()
def generate_steering_vector(model, tokenizer, steer_dict: dict, pos_alpha: float = 1.0,
                             neg_alpha: float = 1.0, layer_from_last: int = -1): # dict of {token: weight}

    device = model.device
    pos_vectors, neg_vectors = [], []

    for text, weight in steer_dict.items():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                            max_length=50).to(device) # long-ish steering prompts!
        
        prompt_out = model( 
            **inputs,
            output_hidden_states=True,
            use_cache=True  # so you can pass past_key_values into manual generate loop if desired
        )

        prompt_hidden = prompt_out.hidden_states[layer_from_last].to(device)  # [1, prompt_len, H]
        mean_hidden = prompt_hidden.mean(dim=1).squeeze(dim=0) # average over sequence length, # [1, H] -> [H]
        
        if weight > 0:
            pos_vectors.append(mean_hidden * abs(weight))
        else:
            neg_vectors.append(mean_hidden * abs(weight))

    # stack list of vectors
    # assuming pos & neg vectors always exist
    avg_pos_vector = torch.stack(pos_vectors).mean(dim=0) # [num_prompts, H] -> [H]
    avg_neg_vector = torch.stack(neg_vectors).mean(dim=0) # [num_prompts, H] -> [H]
     
    total_vector = (pos_alpha * avg_pos_vector) - (neg_alpha * avg_neg_vector)

    # [1,1,H]
    return total_vector.view(1,1,-1) # -1 is inferred direction, could also reshape in generate_with_steering
