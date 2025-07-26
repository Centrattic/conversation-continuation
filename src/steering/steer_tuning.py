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


def objective_maximize_norm(trial):
    # sample hyperparameters
    layer_extract = trial.suggest_int('layer_extract', -33, -1)

    steer_min = max(layer_extract - 5, -32)
    steer_max = min(layer_extract + 5, -2) # can't allow -1 since will be divide by zero errors

    layer_steer = trial.suggest_int('layer_steer', steer_min, steer_max)

    alpha = trial.suggest_float('alpha', 0.5, 2.0)

    # compute steering vector
    steering_vector = generate_steering_vector(
        model, tokenizer, steer_dict, alpha=alpha, layer_from_last=layer_extract
    ).to(model.device)

    prompt_diffs = []
    for prompt in steer_prompts:
        # tokenize prompt
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
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
            b_layer = base_hiddens[l][0, :prompt_len, :] # batch_size (alwyas using 1), seq_len, model size
            s_layer = steer_hiddens[l][0, :prompt_len, :]
            token_diffs = (s_layer - b_layer).norm(dim=-1)
            layer_diffs.append(token_diffs.mean().item())
        
        layer_avg = sum(layer_diffs) / len(layer_diffs)
        
        prompt_diffs.append(layer_avg)

    mean_prompt_diffs = sum(prompt_diffs) / len(prompt_diffs)

    # 10 is a bit arbitrary, can tune
    return mean_prompt_diffs

def objective_human_rating(layer_extract, layer_steer, alpha): # LMAO
    # compute steering vector
    steering_vector = generate_steering_vector(
        model, tokenizer, steer_dict, alpha=alpha, layer_from_last=layer_extract
    ).to(model.device)

    for prompt in steer_prompts[0]: # only use one prompt
        # tokenize prompt
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(model.device)
        prompt_len = inputs['input_ids'].shape[1]

        # steered forward: hook at layer_steer
        def add_vector_hook(module, inp, out):
            hidden = out[0]
            vec = steering_vector.to(hidden.device).to(hidden.dtype)
            hidden = hidden + vec
            return (hidden,) + out[1:]

        hook = model.model.model.layers[layer_steer].register_forward_hook(add_vector_hook)
        # what does return_dict do here?
        steer_out = model.generate( 
            **inputs,
            do_sample=True,
            max_new_tokens=20, # just enough for one [Friend] possibly
            temperature=0.7,
            top_p=0.95,
            top_k=0,
            pad_token_id=tokenizer.eos_token_id,
        )

        hook.remove()

        out_text = tokenizer.decode(steer_out[0]).replace(prompt, "").strip()

        # for friend_bot right now, could change
        # try just seeing whole coherence 
        # index = out_text.find(f"['{RIYA_NAME}]") 
        # if index == -1: # half of [Riya] outputted
        #     index = out_text.find(f"['") # {RIYA_NAME[0]} why are the tokens different than expected with the apostrophe ' ???
        # if index == -1:
        #     index = out_text.find(f"[{RIYA_NAME}]") 
        # if index == -1:
        #     index = out_text.find(f"[{RIYA_NAME[0]}") 
        # if index == -1:
        #     index = len(out_text)
        # out_text = out_text[:index]
        # # removing start and end tokens
        out_text = out_text.replace("<s>", "").strip()
        out_text = out_text.replace("</s>", "").strip()

        print(out_text)

    print("Score from 1-10.") # basically give full on coherence if good enough for what you want
    score = input()

    return int(score)

# Slow but oh well for now.

def objective_maximize_norm_plus_coherence(trial):   
    return objective_maximize_norm(trial) + 10*objective_human_rating(**trial.params)

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
              "This is the best day ever": 0.2,
              "I can't stop grinning": 0.2, # even though friend doesn't grin :/
              "I am very sad": -0.2,
              "I feel really down right now": -0.2,
              "Life is terrible": -0.2,
              "This is the worst day ever": -0.2,
              "I can't stop crying": -0.2
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
