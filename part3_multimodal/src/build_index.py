"""Pre-compute L2-normalised CLIP image embeddings for every image in data/splits.csv (data/clean/).

Run once (CPU, ~10-20 min):  docker compose --profile dev run --rm dev python part3_multimodal/src/build_index.py
Outputs (gitignored, artifacts/): clip_clean_embeddings.npy (float16, N x 512) + clip_clean_index.csv (row i ↔ embedding i)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_common import CLIP_ARCH, CLIP_PRETRAINED, get_clip  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "part3_multimodal" / "artifacts"
EMB, IDX = ART / "clip_clean_embeddings.npy", ART / "clip_clean_index.csv"


class DS(Dataset):
    def __init__(self, paths, tf): self.paths, self.tf = paths, tf
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return self.tf(Image.open(REPO / self.paths[i]).convert("RGB"))


def main():
    df = pd.read_csv(REPO / "data" / "splits.csv")
    model, preprocess, _ = get_clip()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    dl = DataLoader(DS(df.filepath.tolist(), preprocess), batch_size=64, num_workers=2)
    feats, t0 = [], time.time()
    with torch.inference_mode():
        for i, x in enumerate(dl):
            feats.append(torch.nn.functional.normalize(model.encode_image(x).float(), dim=-1))
            done = min((i + 1) * 64, len(df))
            print(f"{done}/{len(df)}  {done / (time.time() - t0):.1f} img/s", end="\r", flush=True)
    E = torch.cat(feats).numpy().astype(np.float16)
    ART.mkdir(parents=True, exist_ok=True)
    np.save(EMB, E)
    df[["filepath", "label", "split"]].to_csv(IDX, index=False)
    print(f"\nsaved {E.shape} ({CLIP_ARCH}/{CLIP_PRETRAINED}) to {EMB.name} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
