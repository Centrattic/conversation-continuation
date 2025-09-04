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
        base_model, device_map="auto", quantization_config=bnb_config)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def sample_hyperparams(trial):
    layer_extract = trial.suggest_int('layer_extract', -33, -1)
    steer_min = max(layer_extract - 15, -32)
    steer_max = min(layer_extract + 15, -2)
    layer_steer = trial.suggest_int('layer_steer', steer_min, steer_max)
    alpha = trial.suggest_float('alpha', 0.7, 5.0)
    return layer_extract, layer_steer, alpha


def compute_norm_diff(model, tokenizer, steer_prompts, layer_steer, hook_func):

    diffs = []
    for prompt in steer_prompts:
        # tokenize & baseline
        inputs = tokenizer(prompt,
                           return_tensors="pt",
                           truncation=True,
                           max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].size(1)

        hook = model.model.model.layers[layer_steer].register_forward_hook(
            hook_func)

        # steered forward
        steer_out = model(**inputs,
                          output_hidden_states=True,
                          return_dict=True)
        steer_hs = steer_out.hidden_states

        hook.remove()

        base_out = model(**inputs, output_hidden_states=True, return_dict=True)
        base_hs = base_out.hidden_states

        # pull out the “downstream” layers
        total = len(base_hs) - 1
        idx = layer_steer if layer_steer >= 0 else total + layer_steer + 1
        downstream = range(idx + 1, total + 1)

        # compute ∥Δh∥ over prompt tokens
        layer_means = []
        for l in downstream:
            # each base_hs[l] is Tensor(1, L, D)
            b = base_hs[l][0, :prompt_len, :]
            s = steer_hs[l][0, :prompt_len, :]
            layer_means.append((s - b).norm(dim=-1).mean().item())

        diffs.append(sum(layer_means) / len(layer_means))

    return sum(diffs) / len(diffs)


def compute_proxy_coherence(model, tokenizer, steer_prompts, layer_steer,
                            hook_func):
    # (You could also generate and then score the generated text with a reference model.)
    total_score = 0.0
    for prompt in steer_prompts[:1]:  # just one prompt for speed
        inputs = tokenizer(prompt,
                           return_tensors="pt",
                           truncation=True,
                           max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        hook = model.model.model.layers[layer_steer].register_forward_hook(
            hook_func)

        # steer once to get a sample continuation
        gen_ids = model.generate(
            **inputs,
            do_sample=
            False,  # greedy decoding to get the best idea of steering effect (deterministic)
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id)

        hook.remove()  # very cheap to do this

        out_text = tokenizer.decode(gen_ids[0]).replace(prompt, "").strip()
        out_text = out_text.replace("<s>", "")
        out_text = out_text.replace("</s>", "")

        print(out_text)

    print("Score from 0-10.")
    total_score = input()

    return int(total_score)


def compute_contrast_diff(model, tokenizer, steer_prompts, layer_steer,
                          hook_func_pos, hook_func_neg):

    diffs = []
    for prompt in steer_prompts:
        # tokenize & baseline
        inputs = tokenizer(prompt,
                           return_tensors="pt",
                           truncation=True,
                           max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].size(1)

        hook = model.model.model.layers[layer_steer].register_forward_hook(
            hook_func_pos)
        steer_pos_out = model(**inputs,
                              output_hidden_states=True,
                              return_dict=True)
        steer_pos_hs = steer_pos_out.hidden_states
        hook.remove()

        hook = model.model.model.layers[layer_steer].register_forward_hook(
            hook_func_neg)
        steer_neg_out = model(**inputs,
                              output_hidden_states=True,
                              return_dict=True)
        steer_neg_hs = steer_neg_out.hidden_states
        hook.remove()

        # pull out the “downstream” layers
        total = len(steer_neg_hs) - 1
        idx = layer_steer if layer_steer >= 0 else total + layer_steer + 1
        downstream = range(idx + 1, total + 1)

        # compute ∥Δh∥ over prompt tokens
        layer_means = []
        for l in downstream:
            # each base_hs[l] is Tensor(1, L, D)
            b = steer_neg_hs[l][0, :prompt_len, :]
            s = steer_pos_hs[l][0, :prompt_len, :]
            layer_means.append((s - b).norm(dim=-1).mean().item())

        diffs.append(sum(layer_means) / len(layer_means))

    return sum(diffs) / len(diffs)


def objective_maximize_norm(trial):
    layer_extract, layer_steer, alpha = sample_hyperparams(trial)

    steering_vector = generate_steering_vector(
        model,
        tokenizer,
        steer_dict,
        pos_alpha=alpha,
        neg_alpha=alpha,
        layer_from_last=layer_extract).to(model.device)

    def add_vec(module, inp, out):
        h, *r = out
        return (h + steering_vector.to(h.device).to(h.dtype), *r)

    diff_score = compute_norm_diff(model, tokenizer, steer_prompts,
                                   layer_steer, add_vec)

    return diff_score


def objective_human_rating(trial):
    layer_extract, layer_steer, alpha = sample_hyperparams(trial)
    # here “human rating” is your proxy coherence
    steering_vector = generate_steering_vector(
        model,
        tokenizer,
        steer_dict,
        pos_alpha=alpha,
        neg_alpha=alpha,
        layer_from_last=layer_extract).to(model.device)

    def add_vec(module, inp, out):
        h, *r = out
        return (h + steering_vector.to(h.device).to(h.dtype), *r)

    # hook not removed internally
    coh_score = compute_proxy_coherence(model, tokenizer, steer_prompts,
                                        layer_steer, add_vec)

    return coh_score


def objective_maximize_norm_plus_coherence(trial):
    layer_extract, layer_steer, alpha = sample_hyperparams(trial)

    # model is modified in place
    steering_vector = generate_steering_vector(
        model,
        tokenizer,
        steer_dict,
        pos_alpha=alpha,
        neg_alpha=alpha,
        layer_from_last=layer_extract).to(model.device)

    def add_vec(module, inp, out):
        h, *r = out
        return (h + steering_vector.to(h.device).to(h.dtype), *r)

    coh_score = compute_proxy_coherence(model, tokenizer, steer_prompts,
                                        layer_steer, add_vec)

    norm_score = compute_norm_diff(model, tokenizer, steer_prompts,
                                   layer_steer, add_vec)

    # weight coherence ×10 just like before, so max 100 for both
    return min(norm_score, 100) + 10.0 * coh_score


# technically contrast consistence is more like max diff is at the opposite directions
# maybe compute grad of diff somehow with respect to direction and try to make sure its small?
def objective_contrast_consistence(trial):
    # OMG. here just look for diff between positive and negative alpha applied.
    # Because for a spurious correlation I just did, the diff was like nothing bro.
    layer_extract, layer_steer, alpha = sample_hyperparams(trial)

    steering_vector = generate_steering_vector(
        model,
        tokenizer,
        steer_dict,
        pos_alpha=alpha,
        neg_alpha=alpha,
        layer_from_last=layer_extract).to(model.device)

    def add_vec(module, inp, out):
        h, *r = out
        return (h + steering_vector.to(h.device).to(h.dtype), *r)

    def sub_vec(module, inp, out):
        h, *r = out
        return (h - steering_vector.to(h.device).to(h.dtype), *r)

    coh_score = compute_proxy_coherence(model, tokenizer, steer_prompts,
                                        layer_steer, add_vec)

    norm_contrast_score = compute_contrast_diff(model, tokenizer,
                                                steer_prompts, layer_steer,
                                                add_vec, sub_vec)
    return min(norm_contrast_score, 100) + 10.0 * coh_score


def objective_coherence_maximization(trial):
    pass


parser = argparse.ArgumentParser()
parser.add_argument('--trials',
                    type=int,
                    default=50,
                    help='Number of Optuna trials')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)

# should try to have these be statements our bots actually say methinks
# on that note, how best to discourage the "not bad" answer

# just normalized these so alpha is within constant range for tuning
steer_dict = {
    "I feel happy": 0.25,
    "I feel sad": -0.25,
    "life is amazing": 0.25,
    "life is terrible": -0.25,
    "this was a great day": 0.25,
    "this was a bad day": -0.25,
    "amazing": 0.25,
    "not bad": -0.25,
}

# ToDo: should I add [Riya] to the prompts? Since I'm trying to steer [Friend]? Can try if this fails

steer_prompts = [
    f"{[RIYA_NAME]}: how are u doing today? \n {[FRIEND_NAME]}:",
    f"{[RIYA_NAME]}: tell me how you are feeling \n {[FRIEND_NAME]}:",
    f"{[RIYA_NAME]}: how has your life been? \n {[FRIEND_NAME]}:",
    f"{[RIYA_NAME]}: what's been on your mind lately {[FRIEND_NAME]}:",
    f"{[RIYA_NAME]}: tell me about your day \n {[FRIEND_NAME]}:",
    f"{[RIYA_NAME]}: is there anything worrying you right now? \n {[FRIEND_NAME]}:",
    f"{[RIYA_NAME]}: what are you most happy about? \n {[FRIEND_NAME]}:",
    f"{[RIYA_NAME]}: describe your mood in three words \n {[FRIEND_NAME]}:"
]

model, tokenizer = load_models()

study = optuna.create_study(direction='maximize')
study.optimize(objective_contrast_consistence, n_trials=args.trials)

print('Best trial:')
print(study.best_trial.params)
print('Best value:', study.best_value)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
out_path = os.path.join('./src/steering/vector_trials',
                        f'{timestamp}_best_trial.txt')
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
