import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from pathlib import Path

from src.config import DATA_PATH, MODEL_NAME

CSV_PATH = DATA_PATH
COLUMN_NAME = "Content"
MODEL_NAME = MODEL_NAME

df = pd.read_csv(CSV_PATH)

# Compute character lengths
char_lengths = df[COLUMN_NAME].astype(str).str.len()

base_model = Path(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(
    base_model)  # LORA doesn't modify tokenizer so don't need that
tokenizer.pad_token = tokenizer.eos_token  # ensure padding token

# Compute token counts
token_counts = df[COLUMN_NAME].astype(str).apply(
    lambda txt: len(tokenizer(txt).input_ids))

# Plot character len distribution
plt.figure(figsize=(8, 5))
plt.hist(char_lengths,
         bins=range(0,
                    char_lengths.max() + 5, 5),
         edgecolor='black')
plt.xlabel("Length (characters)")
plt.ylabel("Count")
plt.title(f"Character‐Length Distribution for “{COLUMN_NAME}”")
plt.grid(axis='y', alpha=0.75)
plt.yscale('log')
plt.xlim(0, max(char_lengths))
print(max(char_lengths))

char_fname = f"data_analysis/char_length_dist.png"
plt.savefig(char_fname, bbox_inches='tight')
plt.close()

# Plot token count distribution
plt.figure(figsize=(8, 5))
plt.hist(token_counts,
         bins=range(0,
                    token_counts.max() + 5, 5),
         edgecolor='black')
plt.xlabel("Number of Tokens")
plt.ylabel("Count")
plt.title(f"Tokenizer Token‐Count Distribution for “{COLUMN_NAME}”")
plt.grid(axis='y', alpha=0.75)
plt.yscale('log')
print(max(token_counts))
plt.xlim(0, max(token_counts))
token_fname = f"data_analysis/token_count_dist.png"
plt.savefig(token_fname, bbox_inches='tight')
plt.close()

print(f"Saved character‐length plot as {char_fname}")
print(f"Saved token‐count plot as {token_fname}")
