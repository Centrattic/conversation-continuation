import torch

@torch.no_grad()
def generate(model, prompt:str, tokenizer, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7, # play with this!
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0]).replace(prompt, "").strip()