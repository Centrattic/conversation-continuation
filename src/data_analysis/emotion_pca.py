import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.config import RESULTS_FOLDER, MODEL_NAME, bnb_config, VIZ_FOLDER
from pathlib import Path
from peft import PeftModel
import umap.umap_ as umap
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
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config).to("cpu")
lora_base_model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config).to("cpu")
lora_base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
finetuned_model = PeftModel.from_pretrained(lora_base_model, adapter_path).to("cpu")

base_model.eval()
finetuned_model.eval()

# Function to extract embeddings (captures LoRA effects)
def get_embeddings(model):
    encoding = tokenizer(emotions, padding=True, return_tensors='pt')
    input_ids = encoding.input_ids.to('cpu')
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(input_ids)
    # Average across token dimension
    avg_embeddings = embeddings.mean(dim=1)
    return avg_embeddings.cpu()

# Retrieve embeddings
base_embeddings = get_embeddings(base_model)
finetuned_embeddings = get_embeddings(finetuned_model)

# Compute PCA
pca2d = PCA(n_components=2)
base_pca2 = pca2d.fit_transform(base_embeddings)
fin_pca2 = pca2d.transform(finetuned_embeddings)

pca3d = PCA(n_components=3)
base_pca3 = pca3d.fit_transform(base_embeddings)
fin_pca3 = pca3d.transform(finetuned_embeddings)

# Compute UMAP
umap2d = umap.UMAP(n_components=2, random_state=0)
base_um2 = umap2d.fit_transform(base_embeddings)
fin_um2 = umap2d.transform(finetuned_embeddings)

# Compute t-SNE
tsne2d = TSNE(n_components=2, random_state=0)
base_ts2 = tsne2d.fit_transform(base_embeddings)
fin_ts2 = tsne2d.fit_transform(finetuned_embeddings)

tsne3d = TSNE(n_components=3, random_state=0)
base_ts3 = tsne3d.fit_transform(base_embeddings)
fin_ts3 = tsne3d.fit_transform(finetuned_embeddings)

print("Fit the dimensionality reductions.")

# Plotting functions
def plot_2d(data, title, filename, color):
    plt.figure(figsize=(8, 6))
    for i, emo in enumerate(emotions):
        # Safely unpack first two dimensions
        x = data[i][0]
        y = data[i][1]
        plt.scatter(x, y, c=color)
        plt.text(x + 1e-3, y + 1e-3, emo, color=color, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_FOLDER, filename))
    plt.close()

def plot_overlay_2d(data1, data2, title, filename):
    plt.figure(figsize=(8, 6))
    for i, emo in enumerate(emotions):
        x1, y1 = data1[i]
        x2, y2 = data2[i]
        plt.scatter(x1, y1, c='red', marker='o')
        # Label base embedding
        plt.text(x1 + 1e-3, y1 + 1e-3, emo, color='red', fontsize=6)
        plt.scatter(x2, y2, c='blue', marker='x')
        # Label finetuned embedding
        plt.text(x2 + 1e-3, y2 + 1e-3, emo, color='blue', fontsize=6)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_FOLDER, filename))
    plt.close()

def plot_3d(data, title, filename, color):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i, emo in enumerate(emotions):
        x, y, z = data[i]
        ax.scatter(x, y, z, c=color)
        ax.text(x, y, z, emo, color=color, fontsize=6)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_FOLDER, filename))
    plt.close()

def plot_overlay_3d(data1, data2, title, filename):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i, emo in enumerate(emotions):
        x1, y1, z1 = data1[i]
        x2, y2, z2 = data2[i]
        ax.scatter(x1, y1, z1, c='red', marker='o')
        ax.scatter(x2, y2, z2, c='blue', marker='x')
        ax.text(x2, y2, z2, emo, color='blue', fontsize=6)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_FOLDER, filename))
    plt.close()

# Save plots for each method
methods = [
    ('pca2', base_pca2, fin_pca2),
    ('pca3', base_pca3, fin_pca3),
    ('umap2', base_um2, fin_um2),
    ('tsne2', base_ts2, fin_ts2),
    ('tsne3', base_ts3, fin_ts3)
]

for name, base_data, fin_data in methods:
    print(name)
    if base_data.shape[1] == 2:
        # plot_2d(base_data, f'Base {name.upper()} 2D', f'base_{name}_2d.png', 'red')
        # plot_2d(fin_data, f'Finetuned {name.upper()} 2D', f'fin_{name}_2d.png', 'blue')
        plot_overlay_2d(base_data, fin_data, f'Overlay {name.upper()} 2D', f'overlay_{name}_2d.png')

    if base_data.shape[1] == 3:
        # plot_3d(base_data, f'Base {name.upper()} 3D', f'base_{name}_3d.png', 'red')
        # plot_3d(fin_data, f'Finetuned {name.upper()} 3D', f'fin_{name}_3d.png', 'blue')
        plot_overlay_3d(base_data, fin_data, f'Overlay {name.upper()} 3D', f'overlay_{name}_3d.png')

print('Saved PCA, UMAP, and t-SNE plots to', VIZ_FOLDER)
