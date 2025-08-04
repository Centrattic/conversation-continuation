""" 2D and 3D PCA plots """
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.config import RESULTS_FOLDER, MODEL_NAME, bnb_config
from pathlib import Path
from peft import PeftModel

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

# Base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, # device map=auto doesn't work as well
                                                  quantization_config=bnb_config).to("cpu")

lora_base_model = AutoModelForCausalLM.from_pretrained(base_model_name,
                                                       quantization_config=bnb_config).to("cpu")
vocab_size = len(tokenizer)
lora_base_model.resize_token_embeddings(vocab_size, mean_resizing=False)

finetuned_model = PeftModel.from_pretrained(lora_base_model, 
                                            adapter_path).to("cpu")
# Ensure models in eval mode
base_model.eval()
finetuned_model.eval()

def get_embeddings(model):
    encoding = tokenizer(emotions, padding=True, return_tensors='pt')
    input_ids = encoding.input_ids.to('cpu')
    with torch.no_grad():
        embs = model.get_input_embeddings()(input_ids)
    return embs.mean(dim=1).cpu()

# Retrieve embeddings
base_embeddings = get_embeddings(base_model)
finetuned_embeddings = get_embeddings(finetuned_model)

# Compute PCA
pca2d = PCA(n_components=2)
base_2d = pca2d.fit_transform(base_embeddings)
finetuned_2d = pca2d.transform(finetuned_embeddings)

pca3d = PCA(n_components=3)
base_3d = pca3d.fit_transform(base_embeddings)
finetuned_3d = pca3d.transform(finetuned_embeddings)

# Plot individual 2D embeddings
def plot_individual_2d(data, title, filename, color):
    plt.figure(figsize=(10, 8))
    for i, emo in enumerate(emotions):
        x, y = data[i]
        plt.scatter(x, y, color=color)
        plt.text(x + 1e-3, y + 1e-3, emo, fontsize=9, color=color)
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_individual_2d(base_2d, 'Base Model Embeddings (2D PCA)', 'base_embeddings_2d.png', 'red')
plot_individual_2d(finetuned_2d, 'Finetuned Model Embeddings (2D PCA)', 'finetuned_embeddings_2d.png', 'blue')

# Plot individual 3D embeddings
def plot_individual_3d(data, title, filename, color):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i, emo in enumerate(emotions):
        x, y, z = data[i]
        ax.scatter(x, y, z, color=color)
        ax.text(x, y, z, emo, color=color)
    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_individual_3d(base_3d, 'Base Model Embeddings (3D PCA)', 'base_embeddings_3d.png', 'red')
plot_individual_3d(finetuned_3d, 'Finetuned Model Embeddings (3D PCA)', 'finetuned_embeddings_3d.png', 'blue')

# Plot 2D overlay
plt.figure(figsize=(12, 10))
for i, emo in enumerate(emotions):
    bx, by = base_2d[i]
    fx, fy = finetuned_2d[i]
    plt.scatter(bx, by, c='red', marker='o')
    plt.text(bx + 1e-3, by + 1e-3, emo, fontsize=9, color='red')
    plt.scatter(fx, fy, c='blue', marker='x')
    plt.text(fx + 1e-3, fy + 1e-3, emo, fontsize=9, color='blue')
plt.title('Base (red) vs. Finetuned (blue) Embeddings (2D PCA Overlay)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.savefig('overlay_embeddings_2d.png')
plt.close()

# Plot 3D overlay
def plot_overlay_3d(base_data, finetuned_data, title, filename):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i, emo in enumerate(emotions):
        bx, by, bz = base_data[i]
        fx, fy, fz = finetuned_data[i]
        ax.scatter(bx, by, bz, color='red', marker='o')
        ax.text(bx, by, bz, emo, color='red')
        ax.scatter(fx, fy, fz, color='blue', marker='x')
        ax.text(fx, fy, fz, emo, color='blue')
    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_overlay_3d(base_3d, finetuned_3d, 'Base vs. Finetuned Embeddings (3D PCA Overlay)', 'overlay_embeddings_3d.png')

print("Saved all plots: base_embeddings_2d.png, finetuned_embeddings_2d.png, base_embeddings_3d.png, finetuned_embeddings_3d.png, overlay_embeddings_2d.png, overlay_embeddings_3d.png")
