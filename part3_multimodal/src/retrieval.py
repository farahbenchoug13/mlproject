"""Text → dog photo search over the pre-computed CLIP index (build_index.py)."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_index import EMB, IDX, REPO  # noqa: E402
from clip_common import encode_text  # noqa: E402

PROMPT_TEMPLATE = "a photo of {}"      # turns "grumpy pug judging you" into a CLIP-style caption


@lru_cache(maxsize=1)
def load_index():
    if not EMB.exists():
        raise FileNotFoundError(f"{EMB} missing — run part3_multimodal/src/build_index.py first")
    return torch.from_numpy(np.load(EMB).astype(np.float32)), pd.read_csv(IDX)


def search(query: str, k: int = 4, split: str | None = None) -> list[dict]:
    """Top-k images for a free-text query. `split` restricts to train/val/test (used for evaluation)."""
    E, idx = load_index()
    q = query.strip()
    if not q.lower().startswith(("a photo", "a picture", "an image")):
        q = PROMPT_TEMPLATE.format(q)
    sims = (E @ encode_text([q])[0]).numpy()
    if split:
        sims = np.where(idx.split.values == split, sims, -np.inf)
    top = np.argsort(-sims)[:k]
    return [{"path": str(REPO / idx.filepath[i]), "filepath": idx.filepath[i], "label": idx.label[i],
             "split": idx.split[i], "similarity": float(sims[i])} for i in top]


if __name__ == "__main__":
    for r in search(" ".join(sys.argv[1:]) or "a happy dog running on grass", k=5):
        print(f"{r['similarity']:.3f}  {r['label']:8s} {r['filepath']}")
