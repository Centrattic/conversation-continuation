import argparse
import torch
import optuna
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.config import MODEL_NAME, RESULTS_FOLDER, bnb_config, RIYA_NAME, FRIEND_NAME
from src.steering.steer_utils import generate_steering_vector
from datetime import datetime
import os
import json

# ToDo: add an option to not just tune via activation diffs but also logit diffs
# Also look at some plots of activation diffs for a given layer at different alphas, the alpha seems really important to the jump (activation stable regions perturbations)
# Instead of alpha maybe we should actually be perturbing to entirely swtiching the activations, not just adding them in but like moving the vector in that direction hmm ask

# ToDo: add padding so model can process statements in batch and so optuna trials are a bit faster.


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

def sample_hyperparams(trial):
    layer_extract = trial.suggest_int('layer_extract', -33, -1)
    steer_min = max(layer_extract - 15, -32)
    steer_max = min(layer_extract + 15, -2)
    layer_steer = trial.suggest_int('layer_steer', steer_min, steer_max)
    alpha = trial.suggest_float('alpha', 0.7, 2.0)
    return layer_extract, layer_steer, alpha

def generate_steering_hook(model, tokenizer, steer_dict,
                           layer_steer, layer_extract, alpha):
    steering_vector = generate_steering_vector(
        model, tokenizer, steer_dict, pos_alpha=alpha,
        neg_alpha=alpha, layer_from_last=layer_extract
    ).to(model.device)

    def add_vec(module, inp, out):
        h, *r = out
        return (h + steering_vector.to(h.device).to(h.dtype), *r)

    hook = model.model.model.layers[layer_steer].register_forward_hook(add_vec)

    return model, hook # hooked_model now

def compute_norm_diff(base_model, hooked_model, tokenizer,
                      steer_prompts, layer_steer):
    
    diffs = []
    for prompt in steer_prompts:
        # tokenize & baseline
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].size(1)

        base_out = base_model(**inputs, output_hidden_states=True, return_dict=True)
        base_hs  = base_out.hidden_states

        # steered forward
        steer_out = hooked_model(**inputs, output_hidden_states=True, return_dict=True)
        steer_hs = steer_out.hidden_states

        # pull out the “downstream” layers
        total = len(base_hs) - 1
        idx   = layer_steer if layer_steer >=0 else total + layer_steer + 1
        downstream = range(idx+1, total+1)

        # compute ∥Δh∥ over prompt tokens
        layer_means = []
        for l in downstream:
            # each base_hs[l] is Tensor(1, L, D)
            b = base_hs[l][0, :prompt_len, :]
            s = steer_hs[l][0, :prompt_len, :]
            layer_means.append((s - b).norm(dim=-1).mean().item())

        diffs.append(sum(layer_means) / len(layer_means))

    return sum(diffs) / len(diffs)

def compute_proxy_coherence(hooked_model, tokenizer, steer_prompts):
    # (You could also generate and then score the generated text with a reference model.)
    total_score = 0.0
    for prompt in steer_prompts[:1]:  # just one prompt for speed
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # steer once to get a sample continuation
        gen_ids = hooked_model.generate(
            **inputs, 
            do_sample=True, 
            max_new_tokens=50,
            temperature=0.7, 
            top_p=0.95, 
            pad_token_id=tokenizer.eos_token_id
        )

        out_text = tokenizer.decode(gen_ids[0]).replace(prompt, "").strip()
        out_text = out_text.replace("<s>", "")
        out_text = out_text.replace("</s>", "")

        print(out_text)

    print("Score from 0-10.")
    total_score = input()

    return int(total_score)


def objective_maximize_norm(trial):
    layer_extract, layer_steer, alpha = sample_hyperparams(trial)

    hooked_model, hook = generate_steering_hook(model, tokenizer, steer_dict,
            layer_extract, layer_steer, alpha)

    diff_score = compute_norm_diff(model, hooked_model, tokenizer, 
                                   steer_prompts, layer_steer)

    hook.remove()

    return diff_score

def objective_human_rating(trial):
    layer_extract, layer_steer, alpha = sample_hyperparams(trial)
    # here “human rating” is your proxy coherence
    hooked_model, hook = generate_steering_hook(model, tokenizer, steer_dict,
        layer_extract, layer_steer, alpha)
    
    coh_score = compute_proxy_coherence(hooked_model, tokenizer, steer_prompts)
    
    hook.remove()

    return coh_score

def objective_maximize_norm_plus_coherence(trial):
    layer_extract, layer_steer, alpha = sample_hyperparams(trial)

    hooked_model, hook = generate_steering_hook(model, tokenizer, steer_dict,
        layer_extract, layer_steer, alpha)

    norm_score = compute_norm_diff(
        model, hooked_model, tokenizer, steer_prompts,
        layer_steer
    )

    coh_score = compute_proxy_coherence(
        hooked_model, tokenizer, steer_prompts
    )

    hook.remove()
    # weight coherence ×10 just like before, so max 100 for both
    return min(norm_score, 100) + 10.0 * coh_score

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

# should try to have these be statements our bots actually say methinks

steer_dict = {"I feel happy": 0.25, # just normalized these so alpha is within constant range for tuning
              "I feel sad": -0.25,

              "life is amazing": 0.25,
              "life is terrible": -0.25,

              "this was a great day": 0.25,
              "this was a bad day": -0.25, 
              
              "great": 0.25,
              "not good": -0.25,
            }

# ToDo: should I add [Riya] to the prompts? Since I'm trying to steer [Friend]? Can try if this fails

steer_prompts = [f"{[RIYA_NAME]}: how are u doing today? \n {[FRIEND_NAME]}:", 
                 f"{[RIYA_NAME]}: tell me how you are feeling \n {[FRIEND_NAME]}:",
                 f"{[RIYA_NAME]}: how has your life been? \n {[FRIEND_NAME]}:",
                 f"{[RIYA_NAME]}: what's been on your mind lately {[FRIEND_NAME]}:",
                 f"{[RIYA_NAME]}: tell me about your day \n {[FRIEND_NAME]}:",
                 f"{[RIYA_NAME]}: is there anything worrying you right now? \n {[FRIEND_NAME]}:",
                 f"{[RIYA_NAME]}: what are you most happy about? \n {[FRIEND_NAME]}:",
                 f"{[RIYA_NAME]}: describe your mood in three words \n {[FRIEND_NAME]}:"] 

model, tokenizer = load_models()

study = optuna.create_study(direction='maximize')
study.optimize(objective_maximize_norm_plus_coherence, n_trials=args.trials)

print('Best trial:')
print(study.best_trial.params)
print('Best value:', study.best_value)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
out_path = os.path.join('./src/steering/vector_trials', f'{timestamp}_best_trial.txt')
with open(out_path, 'w') as f:
    # header: prompt and dict
    f.write(f"steer_prompts: {steer_prompts}")
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
