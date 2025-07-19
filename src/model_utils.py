import torch

# ToDo: could refactor generate to generate_friend, generate_riya and handle the sentence portion here. Think about this.

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
    return tokenizer.decode(outputs[0]).replace(prompt, "").strip()

def generate_with_activations(model, prompt:str, tokenizer, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

    activations = []
    def hook(module, hook_inputs):
        # hook_inputs[0] is the final hidden states [B, seq_len, hidden]
        activations.append(hook_inputs[0].detach().cpu())
    h = model.lm_head.register_forward_pre_hook(hook)

    # activations[0] is shape (1,prompt_len,hidden_size)
    # activations[1] is shape (1, prompt_len+1, hidden_size)
    # etc.

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=0,
        pad_token_id=tokenizer.eos_token_id,
    )

    # 5) remove hook
    h.remove()

    out_text = tokenizer.decode(outputs[0]).replace(prompt, "").strip()
    return out_text, activations

def generate_with_steering(**args):
    pass