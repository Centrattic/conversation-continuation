from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Sequence, Tuple

import numpy as np
import torch
from datetime import datetime

from src.config import FRIEND_ID, FRIEND_NAME, RIYA_NAME, RIYA_ID, RESULTS_FOLDER


# GPT method to fix datetime parsing. TBH i should just be able to save it better myself.
def _parse_and_format_ts(ts: str) -> str:
    # Trim fractional seconds to 6 digits for fromisoformat
    if '+' in ts or '-' in ts[19:]:
        # split out timezone part
        for sep in ('+', '-'):
            if sep in ts[19:]:
                main, tz = ts.split(sep, 1)
                tz = sep + tz
                break
    else:
        main, tz = ts, ''
    if '.' in main:
        date, frac = main.split('.')
        frac6 = (frac + '000000')[:6]
        main6 = f"{date}.{frac6}"
    else:
        main6 = main
    dt = datetime.fromisoformat(main6 + tz)
    return dt.strftime("%m-%d-%y-%H-%M-%S")

# ToDo: rewrite this method to run on cuda, so faster. Right now makes convo quite a bit slower.
def find_topk_train_samples(
    cache,                        # A FinalLayerActivationCache instance
    mean_gen_acts: np.ndarray,    # (hidden_size,)
    k: int,
    author_id: str
) -> Tuple[List[str], List[str]]:
    """
    Returns:
      entries: ["{author} {timestamp}: {content}", ...] length k
      prompts: [content, ...] length k
    """
    # Load reverse maps
    author_map = json.loads(cache.paths.author_map.read_text())      # {hash:author}
    content_map = json.loads(cache.paths.content_map.read_text())     # {hash:[ts,content]}

    author_hashes = [h for h, a in author_map.items() if a == author_id]
    if not author_hashes:
        return [], []

    rows = [cache._index[h] for h in author_hashes if h in cache._index]
    if not rows:
        return [], []

    # Load all activations + hashes
    acts = cache._open_act(writable=False)                      # (N, max_len, H)
    all_hashes = np.memmap(cache.paths.hash, dtype=cache.HASH_DTYPE,
                          mode="r", shape=(cache.rows,))

    vecs = acts[rows].mean(axis=1).astype(np.float32)            # (M, H)
    query = (
        mean_gen_acts.detach().cpu().numpy().astype(np.float32)
        if hasattr(mean_gen_acts, "detach")
        else mean_gen_acts.astype(np.float32)
    )

    # ToDo: try different aggregations
    dots = vecs @ query                                         # (M,)
    norms = np.linalg.norm(vecs, axis=1) * np.linalg.norm(query)
    sims = dots / (norms + 1e-12)

    # Top-k
    sub_idxs = np.argsort(sims)[-k:][::-1] # indices into vecs/rows
    selected_rows = [rows[i] for i in sub_idxs]

    entries, prompts = [], []
    for r in selected_rows:
        h = all_hashes[r].decode()
        # cursed long line :p
        author = FRIEND_NAME if author_id == str(FRIEND_ID) else RIYA_NAME if author_id == str(RIYA_ID) else "UNKNOWN"
        ts, content = content_map.get(h, ("<unk>", ""))
        nice_ts = _parse_and_format_ts(ts)
        entries.append(f"{author} {nice_ts}: {content}")
        prompts.append(content)

    return entries, prompts

def make_hash(ts: str, content: str) -> str:
    norm_ts = normalize_timestamp(ts)
    norm_ct = normalize_content(content)
    return hashlib.sha1(f"{norm_ts}|{norm_ct}".encode()).hexdigest()

def normalize_timestamp(ts: str) -> str:
    """Return original string if parsing fails (still deterministic)."""
    return ts.strip()

def normalize_content(text: str) -> str:
    """Trim + collapse internal whitespace"""
    return " ".join(text.strip().split())

def pad_or_truncate(arr: torch.Tensor, seq_len: int) -> torch.Tensor:
    if arr.size(1) == seq_len:
        return arr
    if arr.size(1) > seq_len:
        return arr[:, :seq_len]
    # pad
    pad = seq_len - arr.size(1)
    return torch.nn.functional.pad(arr, (0,0,0,pad))

@dataclass
class CachePaths:
    act: Path
    hash: Path
    shape: Path
    author_map: Path
    meta: Path
    content_map: Path

class FinalLayerActivationCache:
    HASH_DTYPE = np.dtype("S40")
    def __init__(
        self,
        hidden_size: int,
        max_seq_len: int,
        base_dir: Path = Path(f"./{RESULTS_FOLDER}/activation_cache"),
        prefix: str = "final",
    ):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.prefix = prefix
        self._content_map: Dict[str, Tuple[str,str]] = {}
        self.paths = CachePaths(
            act=self.base_dir / f"{prefix}.acts.mmap",
            hash=self.base_dir / f"{prefix}.hash.mmap",
            shape=self.base_dir / f"{prefix}.shape.json",
            author_map=self.base_dir / f"{prefix}.author_index.json",
            meta=self.base_dir / f"{prefix}.meta.json",
            content_map=self.base_dir / f"{prefix}.content_map.json"
        )
        self._row_bytes = self.max_seq_len * self.hidden_size * np.dtype(np.float16).itemsize
        self._index: Dict[str, int] = {}
        self._author_index: Dict[str, str] = {}
        self._load_indices()
    
    @property
    def rows(self) -> int:
        return self._rows_from_shape()

    def has(self, h: str) -> bool:
        return h in self._index

    def add_batch(
        self,
        hashes: Sequence[str],
        authors: Sequence[str],
        timestamps: List[str],
        contents: List[str],
        activations: torch.Tensor,  # (B, seq_len, hidden)
    ) -> None:
        assert activations.dtype in (torch.float16, torch.bfloat16, torch.float32)
        acts = activations.detach().cpu().to(torch.float16).numpy()
        B, S, H = acts.shape
        assert H == self.hidden_size
        # resize files
        old = self.rows
        new = old + B
        act_mm = self._grow_act(old, new)
        hash_mm = self._grow_hash(old, new)
        # write
        act_mm[old:new] = acts
        hash_mm[old:new] = np.array(hashes, dtype=self.HASH_DTYPE)
        act_mm.flush(); hash_mm.flush()
        # update shape + indices
        self._write_shape(new)
        for i, h in enumerate(hashes):
            self._index[h] = old + i
            self._author_index[h] = authors[i]
            self._content_map[h] = (timestamps[i], contents[i])
        self._flush_author_index()

    def get_by_hashes(self, hashes: Sequence[str]) -> np.ndarray:
        if not hashes:
            return np.empty((0, self.max_seq_len, self.hidden_size), dtype=np.float16)
        act_mm = self._open_act(writable=False)
        rows = [self._index[h] for h in hashes if h in self._index]
        return act_mm[rows]

    def hashes_for_author(self, author_id: str) -> List[str]:
        return [h for h, a in self._author_index.items() if a == author_id]
    
    def _load_indices(self):
        if self.paths.hash.exists():
            hashes = np.memmap(self.paths.hash, dtype=self.HASH_DTYPE, mode="r")
            self._index = {h.decode(): i for i, h in enumerate(hashes)}
        if self.paths.author_map.exists():
            self._author_index = json.loads(self.paths.author_map.read_text())
        if self.paths.content_map.exists():
            self._content_map = json.loads(self.paths.content_map.read_text())

    def _flush_author_index(self):
        self.paths.author_map.write_text(json.dumps(self._author_index))
        self.paths.content_map.write_text(json.dumps(self._content_map))

    def _rows_from_shape(self) -> int:
        if not self.paths.shape.exists():
            return 0
        return json.loads(self.paths.shape.read_text()).get("rows", 0)

    def _write_shape(self, rows: int):
        self.paths.shape.write_text(json.dumps({"rows": rows}))
        self.paths.meta.write_text(json.dumps({
            "rows": rows,
            "hidden_size": self.hidden_size,
            "max_seq_len": self.max_seq_len,
        }))

    def _open_act(self, writable: bool):
        rows = self.rows
        if rows == 0:
            shape = (0, self.max_seq_len, self.hidden_size)
            return np.memmap(self.paths.act, dtype=np.float16, mode="w+" if writable else "w+", shape=shape)
        mode = "r+" if writable else "r"
        return np.memmap(self.paths.act, dtype=np.float16, mode=mode, shape=(rows, self.max_seq_len, self.hidden_size))

    def _grow_act(self, old_rows: int, new_rows: int):
        shape = (new_rows, self.max_seq_len, self.hidden_size)
        if not self.paths.act.exists():
            return np.memmap(self.paths.act, dtype=np.float16, mode="w+", shape=shape)
        old_mm = np.memmap(self.paths.act, dtype=np.float16, mode="r", shape=(old_rows, self.max_seq_len, self.hidden_size))
        tmp_path = self.paths.act.with_suffix(".tmp")
        tmp = np.memmap(tmp_path, dtype=np.float16, mode="w+", shape=shape)
        tmp[:old_rows] = old_mm[:]
        self.paths.act.unlink()
        tmp.flush(); tmp._mmap.close()
        tmp_path.rename(self.paths.act)
        return np.memmap(self.paths.act, dtype=np.float16, mode="r+", shape=shape)

    def _grow_hash(self, old_rows: int, new_rows: int):
        shape = (new_rows,)
        if not self.paths.hash.exists():
            return np.memmap(self.paths.hash, dtype=self.HASH_DTYPE, mode="w+", shape=shape)
        old_mm = np.memmap(self.paths.hash, dtype=self.HASH_DTYPE, mode="r", shape=(old_rows,))
        tmp_path = self.paths.hash.with_suffix(".tmp")
        tmp = np.memmap(tmp_path, dtype=self.HASH_DTYPE, mode="w+", shape=shape)
        tmp[:old_rows] = old_mm[:]
        self.paths.hash.unlink()
        tmp.flush(); tmp._mmap.close()
        tmp_path.rename(self.paths.hash)
        return np.memmap(self.paths.hash, dtype=self.HASH_DTYPE, mode="r+", shape=shape)