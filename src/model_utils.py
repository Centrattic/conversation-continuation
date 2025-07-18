import torch

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