import json
import csv
import hashlib
from pathlib import Path
from typing import Dict, Tuple

from src.config import RESULTS_FOLDER


# your existing functions
def normalize_timestamp(ts: str) -> str:
    return ts.strip()


def normalize_content(text: str) -> str:
    return " ".join(text.strip().split())


def make_hash(ts: str, content: str) -> str:
    norm_ts = normalize_timestamp(ts)
    norm_ct = normalize_content(content)
    return hashlib.sha1(f"{norm_ts}|{norm_ct}".encode()).hexdigest()


def rebuild_content_map(
    csv_path: Path,
    author_index_path: Path,
    output_path: Path,
    timestamp_col: str = "Date",
    content_col: str = "Content",
):
    # 1) load existing hash→author
    author_map: Dict[str, str] = json.loads(author_index_path.read_text())

    # 2) re-scan CSV and build hash→(ts,content)
    content_map: Dict[str, Tuple[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get(timestamp_col, "")
            content = row.get(content_col, "")
            h = make_hash(ts, content)
            if h in author_map:
                # only record entries you actually cached
                content_map[h] = (ts, content)

    # 3) write out
    output_path.write_text(
        json.dumps(content_map, ensure_ascii=False, indent=2))
    print(f"Wrote {len(content_map)} entries to {output_path}")


# example usage
if __name__ == "__main__":
    rebuild_content_map(
        csv_path=Path("friend_hist.csv"),
        author_index_path=Path(
            f"{RESULTS_FOLDER}/activation_cache/final.author_index.json"),
        output_path=Path(
            f"{RESULTS_FOLDER}/activation_cache/final.content_map.json"),
    )
