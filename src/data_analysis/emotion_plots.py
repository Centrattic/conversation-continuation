import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.config import RESULTS_FOLDER, MODEL_NAME, bnb_config, VIZ_FOLDER, FRIEND_NAME, RIYA_NAME
from pathlib import Path
from peft import PeftModel
import umap.umap_ as umap
import seaborn as sns
import numpy as np
import os

emotions = [
    " happy",
    " sad",
    " confidence",
    " love",
    " feel",
    " feeling",
    " feelings",
    " hate",
    " fear",
    " anger",
    " joy",
    " surprise",
    " surprised",
    " emotional",
    " emotion",
    " depression",
    " anxiety",
    " panic",
    " bless",
    " blessing",
    " cried",
    " cry",
    " blame",
    " worry",
    " shame",
    " guilt",
    " guilty",
    " pride",
    " proud",
    " delight",
    " delighted",
    " grief",
    " shock",
    " shocked",
    " pain",
    " painful",
    " scared",
    " bored",
    " despair",
    " hope",
    " hoping",
    " frustration",
    " frustrated",
    " amazing",
    " calm",
    " cheer",
    " comfort",
    " compassion"
]

base_model_name = Path(f"{MODEL_NAME}")
adapter_path = Path(f"./{RESULTS_FOLDER}/lora_train/lora_adapter")

# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
tokenizer.pad_token = tokenizer.eos_token

for emo in emotions:
    tokens = tokenizer.tokenize(emo)
    assert len(tokens) == 1, f"Emotion '{emo}' was split into multiple tokens: {tokens}"

# Load base and finetuned models on CPU
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config).to("cuda")
lora_base_model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config).to("cuda")
lora_base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
finetuned_model = PeftModel.from_pretrained(lora_base_model, adapter_path).to("cuda")

base_model.eval()
finetuned_model.eval()

# full-dim static embeddings, not really useful, is just the combo of us or something
def get_static_embeddings(model, words):
    ids = tokenizer.convert_tokens_to_ids(words)
    emb_matrix = model.get_input_embeddings().weight.data
    return emb_matrix[ids].cpu().numpy()

# contextual activations for "i feel {emo}" templates
def get_contextual_embeddings(model, words, speaker):
    if speaker:
        texts = [f"[{speaker}] i feel {w}" for w in words]
    else:
        texts = [f"i feel {w}" for w in words]
    enc = tokenizer(texts, padding=True, return_tensors='pt')
    input_ids = enc.input_ids.to('cuda')
    attention_mask = enc.attention_mask.to('cuda')
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
    # take last hidden-state of last token
    last_hid = outputs.hidden_states[-1].cpu()  # (B, L, D)
    lengths = attention_mask.sum(dim=1) - 1  # index of emotion token per example
    embs = []
    for i, pos in enumerate(lengths):
        embs.append(last_hid[i, pos, :].cpu().numpy())
    return np.vstack(embs)

# Retrieve embeddings
# static_base = get_static_embeddings(base_model, emotions)
# static_fin = get_static_embeddings(finetuned_model, emotions)
context_base = get_contextual_embeddings(base_model, emotions, speaker=None)
together_fin = get_contextual_embeddings(finetuned_model, emotions, speaker=None)
context_fin_friend = get_contextual_embeddings(finetuned_model, emotions, speaker=FRIEND_NAME)
context_fin_riya = get_contextual_embeddings(finetuned_model, emotions, speaker=RIYA_NAME)

print("Activations extracted.")

# Cosine similarity
def plot_cosine_diff_heatmap(base_acts: np.ndarray, fin_acts: np.ndarray, labels: list[str],
                             save_path: str, title: str = 'Δ Cosine Similarity (fin – base)',cmap: str = 'coolwarm'):
    # compute cosine‐similarity matrices
    sim_base = 1.0 - squareform(pdist(base_acts, 'cosine'))
    sim_fin  = 1.0 - squareform(pdist(fin_acts,  'cosine'))
    diff     = sim_fin - sim_base
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(diff,
                xticklabels=labels,
                yticklabels=labels,
                cmap=cmap,
                center=0,
                square=True)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

plot_cosine_diff_heatmap(
    base_acts = together_fin, # vs. context base, to account for finetuning overall shifts?
    fin_acts = context_fin_friend,
    labels = emotions,
    save_path = os.path.join(VIZ_FOLDER, f'heatmap_cosine_diff_{FRIEND_NAME}.png'),
    title = f'Δ Cosine Similarity ({FRIEND_NAME} – Base)'
)

# wow the heatmaps are way more interesting/siginificant when I subtract out the finetune, seems like
# I have to account for finetune vector shifts as being very significant.

plot_cosine_diff_heatmap(
    base_acts = together_fin,
    fin_acts = context_fin_riya,
    labels = emotions,
    save_path = os.path.join(VIZ_FOLDER, f'heatmap_cosine_diff_{RIYA_NAME}.png'),
    title = f'Δ Cosine Similarity ({RIYA_NAME} – Base)'
)

plot_cosine_diff_heatmap(
    base_acts = context_base,
    fin_acts = together_fin,
    labels = emotions,
    save_path = os.path.join(VIZ_FOLDER, 'heatmap_cosine_diff_together.png'),
    title = f'Δ Cosine Similarity (Us Together – Base)'
)

print('Saved heatmap and Procrustes PCA plots under', VIZ_FOLDER)
