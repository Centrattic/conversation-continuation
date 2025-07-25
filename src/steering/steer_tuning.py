import argparse
import torch
import optuna
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.config import MODEL_NAME, RESULTS_FOLDER, bnb_config
from src.steering.steer_utils import generate_steering_vector
from datetime import datetime
import os
import json

# ToDo: add an option to not just tune via activation diffs but also logit diffs
# Also look at some plots of activation diffs for a given layer at different alphas, the alpha seems really important to the jump (activation stable regions perturbations)
# Instead of alpha maybe we should actually be perturbing to entirely swtiching the activations, not just adding them in but like moving the vector in that direction hmm ask

def load_models():
    base_model = Path(MODEL_NAME)
    adapter_path = Path(f"./{RESULTS_FOLDER}/lora_adapter")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", quantization_config=bnb_config
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def objective_maximize_norm(trial):
    # sample hyperparameters
    layer_extract = trial.suggest_int('layer_extract', -33, -1)
    layer_steer = trial.suggest_int('layer_steer', -32, -2) # if we said -1, we have divide by 0 errors, no downstream layer
    alpha = trial.suggest_float('alpha', 0.05, 5.0)

    # compute steering vector
    steering_vector = generate_steering_vector(
        model, tokenizer, steer_dict, alpha=alpha, layer_from_last=layer_extract
    ).to(model.device)

    # tokenize prompt
    inputs = tokenizer(
        steer_prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(model.device)
    prompt_len = inputs['input_ids'].shape[1]

    # baseline forward
    base_out = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True
    )
    base_hiddens = base_out.hidden_states  # tuple of tensors [batch, seq, hidden]

    # steered forward: hook at layer_steer
    def add_vector_hook(module, inp, out):
        hidden = out[0]
        vec = steering_vector.to(hidden.device).to(hidden.dtype)
        hidden = hidden + vec
        return (hidden,) + out[1:]

    hook = model.model.model.layers[layer_steer].register_forward_hook(add_vector_hook)
    steer_out = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True
    )
    hook.remove()
    steer_hiddens = steer_out.hidden_states

    # determine downstream layers
    total_layers = len(base_hiddens) - 1
    idx = layer_steer if layer_steer >= 0 else total_layers + layer_steer + 1
    downstream = list(range(idx + 1, total_layers + 1))

    # compute average norm difference over prompt tokens
    # ToDo: look at output tokens instead of prompt tokens?
    layer_diffs = []
    for l in downstream:
        b_layer = base_hiddens[l][0, :prompt_len, :] # batch_size, seq_len, model size
        s_layer = steer_hiddens[l][0, :prompt_len, :]
        token_diffs = (s_layer - b_layer).norm(dim=-1)
        layer_diffs.append(token_diffs.mean().item())

    return sum(layer_diffs) / len(layer_diffs)

def objective_coherence_maximization(trial):
    pass

def objective_contrast_consistence(trial):
    pass



parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=50,
                    help='Number of Optuna trials')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)
steer_dict = {"I am very happy": 0.2, 
              "I am happy in life!": 0.2,
              "Life is amazing": 0.2,
              ""
              }
steer_prompt = "how are u doing today?" # ToDo: maybe add multiple prompts?

model, tokenizer = load_models()

study = optuna.create_study(direction='maximize')
study.optimize(objective_maximize_norm, n_trials=args.trials)

print('Best trial:')
print(study.best_trial.params)
print('Best value:', study.best_value)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
out_path = os.path.join('./src/steering/vector_trials', f'{timestamp}_best_trial.txt')
with open(out_path, 'w') as f:
    # header: prompt and dict
    f.write(f"steer_prompt: {steer_prompt}")
    f.write(f"steer_dict: {json.dumps(steer_dict)}")
    # best trial
    f.write("Best trial:")
    f.write(json.dumps(study.best_trial.params) + "")
    f.write(f"Best value: {study.best_value}")
    # history of all trials
    f.write("All trial results:")
    for t in study.trials:
        f.write(f"Trial {t.number}: params={t.params}, value={t.value}")
print(f"Results saved to {out_path}")



# Question: What does this diff look like when we start generating like special tokens, maybe we're actually going to fall into this, but the trial really does seem like its converging.
