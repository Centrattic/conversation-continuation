""" 
Here we extract all activations on all training data (40000 samples). 
Oh no dont do training data initially. It's more expensive and also maybe unncessary.
Let's just do input sequences from friend_hist.csv (12000 or so). We'll cache via hashing indexing, And we should also store who said what. 
Since this will help us attribute correctly to person (can filter out vectors from other person a bit)
We'll also extract last-layer activations particularly.

So we hash content and store in mmap with activations.
Then we have a json dict where each hash maps to AuthorID.

Then during sampling if tda flag is passed, we extract activations and check with get_cosine_similarity function.
And we first filter for all the hashes that belong to the correct author, and we load only those activations.

But maybe expensive to load. Perhaps we save lower dimensional projection (PCA). Maybe we can learn one too somehow?
Fine for now, Mistral model dimension isn't too big.

Also, we can check direction similarity for the prompt maybe too (or at least visualize this with PCA if friend is interested.)
"""

from __future__ import annotations

import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional, Sequence
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm

from src.config import FRIEND_ID, RIYA_ID, RESULTS_FOLDER, MODEL_NAME, DATA_PATH, bnb_config
from src.activation_tda.tda_utils import SingleLayerActivationCache, make_hash, pad_or_truncate


def extract_final_hidden(model: PreTrainedModel,
                         tokenizer: PreTrainedTokenizerBase,
                         texts: Sequence[str], device: str, max_seq_len: int,
                         layer_from_last: int) -> torch.Tensor:

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    toks = tokenizer(
        list(texts),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
    )
    toks = {k: v.to(device) for k, v in toks.items()}
    with torch.no_grad():
        out = model(**toks, output_hidden_states=True, use_cache=False)
    final_hidden = out.hidden_states[layer_from_last]  # (B, S, H)
    return final_hidden


def populate_cache_from_csv(csv_path: Path,
                            cache: SingleLayerActivationCache,
                            model: PreTrainedModel,
                            tokenizer: PreTrainedTokenizerBase,
                            device: str,
                            max_seq_len: int,
                            batch_size: int = 128,
                            content_col: str = "Content",
                            author_col: str = "AuthorID",
                            timestamp_col: str = "Date",
                            min_content_chars: int = 1,
                            layer_from_last: int = -1):
    rows_to_add: List[Tuple[str, str, str,
                            str]] = []  # (hash, author, content)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = row.get(content_col, "")
            if content is None or len(content.strip()) < min_content_chars:
                continue
            author = row.get(author_col, "UNKNOWN")
            ts = row.get(timestamp_col, "")
            h = make_hash(ts, content)
            if cache.has(h):  # don't duplicate activations
                continue
            rows_to_add.append((h, author, ts, content))

    if not rows_to_add:
        return 0

    # Batch process
    for i in tqdm(range(0, len(rows_to_add), batch_size),
                  desc="Populate cache"):
        chunk = rows_to_add[i:i + batch_size]
        hashes = [c[0] for c in chunk]
        authors = [c[1] for c in chunk]
        timestamps = [c[2] for c in chunk]
        texts = [c[3] for c in chunk]

        acts = extract_final_hidden(model, tokenizer, texts, device,
                                    max_seq_len, layer_from_last)  # (B, S, H)
        acts = pad_or_truncate(acts, cache.max_seq_len)
        cache.add_batch(hashes, authors, timestamps, texts, acts)
    return len(rows_to_add)


base_model_name = Path(f"{MODEL_NAME}")
adapter_path = Path(f"./{RESULTS_FOLDER}/lora_adapter")

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Loading models")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name, device_map="auto", quantization_config=bnb_config)
lora_model = PeftModel.from_pretrained(base_model, adapter_path)

hidden_size = lora_model.config.hidden_size
seq_len = 100  # based on data analysis

# Note: change prefix if you're changing layer_from_last, could append that to prefix
cache = SingleLayerActivationCache(hidden_size,
                                   max_seq_len=seq_len,
                                   prefix="last")

print("Populating cache")
populate_cache_from_csv(Path(DATA_PATH),
                        cache,
                        lora_model,
                        tokenizer,
                        device="cuda",
                        max_seq_len=seq_len,
                        layer_from_last=-1)

# ToDo: make this a little better by breaking up messages longer than like 128 tokens into additional messages
# by same author/at same time. So that way we can TDA more precisely + significantly faster activation caching.
# should definitely add this soon, there are some good number of messages (like > 20 fs with seq_len > 100)
