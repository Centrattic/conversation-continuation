import torch

# ToDo: could refactor generate to generate_friend, generate_riya and handle the sentence portion here. Think about this.
# ToDo: add attention masking

@torch.no_grad()
def generate(model, prompt:str, tokenizer, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7, # higher temp avoids you getting stuck with the fish
        top_p=0.95, # how much of probability dist overall you want to account for, min set
        top_k=0,
        pad_token_id=tokenizer.eos_token_id,
    )
    # can i random seed somewhere to get same completion?
    # switch to beam search?
    # can i allow top-p to degrade over time or something, since eventually it finds a high prob thing and settles here
    # less likely to converge if i set top p super low, but also more likely to do worse initially in convos
    out_text = tokenizer.decode(outputs[0]).replace(prompt, "").strip()
    return out_text

@torch.no_grad()
def generate_with_activations(model, prompt:str, tokenizer, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

    # ToDo: move to gpu
    activations = []
    def hook(module, hook_inputs):
        # hook_inputs[0] is the final hidden states [B, seq_len, hidden]
        activations.append(hook_inputs[0].detach().cpu())
    h = model.lm_head.register_forward_pre_hook(hook) # captures input to lm_head

    # activations[0] is shape (1,prompt_len,hidden_size)
    # activations[1] is shape (1, prompt_len+1, hidden_size)
    # etc.

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True, # beam search
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
def generate_with_steering(model, prompt:str, tokenizer, steering_vector, 
                           max_new_tokens=50, layer_from_last=-1):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

    def add_vector_hook(module, input, output): # what does input do here
        # output shape: [batch_size, seq_len, hidden_size]
        # add vector to each token's activation but vector is [1, 1, hidden_size]
        # print("TESTING", output[0].shape, steering_vector.shape)
        hidden_states = output[0]

        # only steer when seq_len == 1 (i.e. a generation step, not at prompt embedding step)
        # or maybe should steer at both?
        if hidden_states.shape[1] == 1:
            hidden_states += steering_vector.to(hidden_states.device)
            return (hidden_states,) + output[1:]
        else:
            return output  # leave prompt pass untouched
        
    # To view visible layers: {for name, module in model.named_modules(): print(name)}

    # for name, module in model.named_modules():
    #     print(name)

    h = model.model.model.layers[layer_from_last].register_forward_hook(add_vector_hook) 
    # this model outputs something like (output_hidden_states, other_outputs...) I guess?

    # If I add activation hook here as well, could do TDA with steering
    outputs = model.generate( # first forward pass of generate ingests prompt
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
