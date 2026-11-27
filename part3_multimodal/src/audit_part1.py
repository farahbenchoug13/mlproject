"""Diagnostic of Part 1's YOLO rejections + candidate list of recovered dogs for a TRAIN-ONLY ablation.

READ-ONLY on Part 1 data: nothing in data/ or outputs/ is modified. Everything is written to
part3_multimodal/results/part1_audit/.

Scope (principled): only photos Part 1 dropped because YOLO saw NO usable dog —
    rejected: cats, removed (no_dog)        to_check: cat_more_confident, low_conf
Photos Part 1 set aside on purpose (small_dog, multi_dog) are left alone (Farah's quality choice).

Steps
  1. CLIP (zero-shot, QuickGELU ViT-B/16) scores {dog, cat, other} for each photo — same prompts as the app's gate.
  2. Recovered candidate = P(dog) >= threshold chosen on Oxford-IIIT Pet trainval (results/dog_gate/summary.json).
  3. Leak check: perceptual hash (imagehash.phash, Hamming <= 5 — Part 1's rule) against every clean val/test image
     (drop: would leak into the exam) and against clean train images with ANOTHER label (drop: label conflict).
  4. Contact sheets of random recovered / still-rejected photos for a MANUAL audit of the diagnostic itself.

Run:  docker compose --profile dev run --rm dev python part3_multimodal/src/audit_part1.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dog_gate  # noqa: E402
from clip_common import encode_images, get_clip  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "part3_multimodal" / "results" / "part1_audit"
SCOPE = {("cats", "no_dog"), ("removed", "no_dog"), ("to-check", "cat_more_confident"), ("to-check", "low_conf")}
HAMMING = 5


def phash_int(path) -> int:
    import imagehash
    return int(str(imagehash.phash(Image.open(path))), 16)


def popcount(x: np.ndarray) -> np.ndarray:
    return np.array([bin(int(v)).count("1") for v in x]) if x.dtype == object else np.unpackbits(
        x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


def contact_sheet(df, path, title, n=48, seed=0):
    sub = df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)
    W = 120
    sheet = Image.new("RGB", (8 * W, ((len(sub) + 7) // 8) * (W + 12) + 18), "white")
    d = ImageDraw.Draw(sheet)
    d.text((4, 2), title, fill="black")
    for i, r in sub.iterrows():
        im = Image.open(REPO / r.filepath).convert("RGB")
        im.thumbnail((W, W))
        x, y = (i % 8) * W, (i // 8) * (W + 12) + 18
        sheet.paste(im, (x, y + 12))
        d.text((x + 2, y), f"{i} {r.label[:3]} d={r.p_dog:.2f}", fill="black")
    sheet.save(path)
    sub[["filepath", "label", "decision", "reason", "p_dog", "p_cat"]].to_csv(path.with_suffix(".csv"))


def main():
    try:
        import imagehash  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imagehash"], check=True)
    OUT.mkdir(parents=True, exist_ok=True)
    thr_file = REPO / "part3_multimodal" / "results" / "dog_gate" / "summary.json"
    thr = json.load(open(thr_file))["threshold"] if thr_file.exists() else dog_gate.RESCUE_THRESHOLD
    print(f"rescue threshold: {thr} ({'from Pet trainval' if thr_file.exists() else 'provisional'})")

    log = pd.read_csv(REPO / "outputs" / "yolo_log.csv")
    log["stem"] = log.src.map(lambda s: Path(s.replace("\\", "/")).stem)
    log = log[[(d, r) in SCOPE for d, r in zip(log.decision, log.reason.fillna("no_dog"))]].copy()
    folder = {"cats": "rejected", "removed": "rejected", "to-check": "to_check"}
    log["filepath"] = [f"data/{folder[d]}/{l}/{s}.jpg" for d, l, s in zip(log.decision, log.label, log.stem)]
    log = log[log.filepath.map(lambda p: (REPO / p).exists())].reset_index(drop=True)
    print(f"{len(log)} photos in scope:", log.groupby(["decision", "reason"]).size().to_dict())

    # 1. CLIP species probabilities (batched)
    W = dog_gate._gate_text_embeddings()
    Wm = torch.stack([W[k] for k in ["dog", "cat", "other"]])
    scale = get_clip()[0].logit_scale.exp().item()
    probs = []
    for i in range(0, len(log), 32):
        f = encode_images([Image.open(REPO / p) for p in log.filepath[i:i + 32]])
        probs.append((scale * f @ Wm.T).softmax(-1))
        print(f"CLIP {min(i + 32, len(log))}/{len(log)}", end="\r", flush=True)
    P = torch.cat(probs).numpy()
    log["p_dog"], log["p_cat"], log["p_other"] = P[:, 0], P[:, 1], P[:, 2]
    log["clip_says_dog"] = log.p_dog >= thr

    # 3. leak / conflict check with perceptual hashes
    clean = pd.read_csv(REPO / "data" / "splits.csv")
    print("\nhashing clean images...")
    clean["h"] = [phash_int(REPO / p) for p in clean.filepath]
    cand = log[log.clip_says_dog].copy()
    cand["h"] = [phash_int(REPO / p) for p in cand.filepath]
    H = np.array(clean.h.tolist(), dtype=np.uint64)
    near_eval, near_conflict, near_any = [], [], []
    for h, lab in zip(cand.h, cand.label):
        dist = popcount(np.bitwise_xor(H, np.uint64(h)))
        close = clean[dist <= HAMMING]
        near_eval.append(bool((close.split != "train").any()))
        near_conflict.append(bool(((close.split == "train") & (close.label != lab)).any()))
        near_any.append(len(close) > 0)
    cand["near_dup_val_test"], cand["near_dup_train_other_label"], cand["near_dup_any_clean"] = near_eval, near_conflict, near_any
    cand["keep"] = ~cand.near_dup_val_test & ~cand.near_dup_train_other_label

    # outputs
    log.drop(columns=["stem"]).to_csv(OUT / "clip_scores_in_scope.csv", index=False)
    cand.drop(columns=["h", "stem"]).to_csv(OUT / "recovery_candidates.csv", index=False)
    rec = cand[cand.keep][["filepath", "label"]].assign(split="train", source="part1_rejected_recovered_by_clip")
    rec.to_csv(OUT / "recovered_train.csv", index=False)

    by_group = log.groupby(["decision", "reason"]).agg(n=("p_dog", "size"), clip_dog=("clip_says_dog", "sum")).reset_index()
    by_label = pd.DataFrame({"in_scope": log.groupby("label").size(), "clip_says_dog": log.groupby("label").clip_says_dog.sum(),
                             "recovered_after_leak_check": rec.groupby("label").size(),
                             "clean_train_count": clean[clean.split == "train"].groupby("label").size()}).fillna(0).astype(int)
    by_label["train_increase_%"] = (100 * by_label.recovered_after_leak_check / by_label.clean_train_count).round(1)
    print("\nby Part 1 decision:\n", by_group.to_string(index=False), "\n\nby emotion:\n", by_label.to_string())
    print(f"\nleak check: {int(cand.near_dup_val_test.sum())} near-duplicates of val/test dropped, "
          f"{int(cand.near_dup_train_other_label.sum())} label conflicts with train dropped, {len(rec)} kept")
    by_group.to_csv(OUT / "summary_by_decision.csv", index=False)
    by_label.to_csv(OUT / "summary_by_emotion.csv")
    json.dump({"threshold": thr, "scope": sorted(map(list, SCOPE)), "n_in_scope": len(log), "n_clip_dog": int(log.clip_says_dog.sum()),
               "n_dropped_near_dup_val_test": int(cand.near_dup_val_test.sum()),
               "n_dropped_label_conflict": int(cand.near_dup_train_other_label.sum()), "n_recovered": len(rec)},
              open(OUT / "summary.json", "w"), indent=1)

    # 4. sheets for the manual audit of the diagnostic
    contact_sheet(log[log.clip_says_dog], OUT / "audit_sheet_clip_dog.png", "CLIP says DOG (manual check: is it a dog?)")
    contact_sheet(log[~log.clip_says_dog], OUT / "audit_sheet_clip_not_dog.png", "CLIP says NOT a dog (manual check)")


if __name__ == "__main__":
    main()
