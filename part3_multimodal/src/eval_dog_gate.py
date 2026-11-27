"""Evaluate the dog gate on INDEPENDENT labels: Oxford-IIIT Pet (Parkhi et al. 2012), 37 breeds, cat vs dog.

Why not data/clean or data/rejected? Those folders were produced by this same YOLO, so testing on them is circular.

Protocol
  * threshold for the CLIP rescue chosen on a sample of Pet **trainval**, reported once on a sample of Pet **test**;
  * compare YOLO only (Part 1 rules) vs YOLO + CLIP rescue;
  * metrics: dog acceptance rate (recall on dogs) and cat false-acceptance rate.
Limitation: Pet has only cats and dogs (no "no animal" photos) and mostly close-up, well-lit pet photos.

Run:  docker compose --profile dev run --rm dev python part3_multimodal/src/eval_dog_gate.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dog_gate  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "part3_multimodal" / "results" / "dog_gate"
DATA = Path("/cache/oxford_pets") if Path("/cache").exists() else REPO / "part3_multimodal" / "artifacts" / "oxford_pets"
N_VAL, N_TEST, SEED = 400, 1000, 42


def run(split: str, n: int) -> pd.DataFrame:
    from torchvision.datasets import OxfordIIITPet
    ds = OxfordIIITPet(str(DATA), split=split, target_types="binary-category", download=True)  # 0 = cat, 1 = dog
    idx = np.random.default_rng(SEED).choice(len(ds), size=min(n, len(ds)), replace=False)
    rows, t0 = [], time.time()
    for k, i in enumerate(idx):
        img, y = ds[int(i)]
        g = dog_gate.run_gate(img, clip_rescue=True)        # computes YOLO; CLIP only if YOLO rejects
        p = g.clip_probs or {}
        rows.append({"i": int(i), "is_dog_true": int(y), "yolo_accept": int(not g.rescued and g.is_dog),
                     "part1_decision": g.part1_decision, "reason": g.reason,
                     "clip_dog": p.get("dog", np.nan), "clip_cat": p.get("cat", np.nan)})
        print(f"{split}: {k + 1}/{len(idx)} ({(time.time() - t0) / (k + 1):.2f}s/img)", end="\r", flush=True)
    print()
    return pd.DataFrame(rows)


def rates(df: pd.DataFrame, thr: float | None) -> dict:
    acc = df.yolo_accept.astype(bool)
    if thr is not None:
        acc = acc | (df.clip_dog.fillna(0) >= thr)
    dogs, cats = df[df.is_dog_true == 1], df[df.is_dog_true == 0]
    dog_rate, cat_rate = acc[dogs.index].mean(), acc[cats.index].mean()
    return {"dog_accept_rate": dog_rate, "cat_false_accept_rate": cat_rate,
            "balanced_accuracy": (dog_rate + (1 - cat_rate)) / 2, "n_dogs": len(dogs), "n_cats": len(cats)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    val = run("trainval", N_VAL)
    sweep = pd.DataFrame([{"threshold": t, **rates(val, t)} for t in np.round(np.arange(0.3, 0.96, 0.05), 2)])
    best = float(sweep.sort_values(["balanced_accuracy", "threshold"], ascending=[False, False]).threshold.iloc[0])
    print(sweep.round(3).to_string(index=False), f"\n→ threshold chosen on Pet trainval: {best}")

    test = run("test", N_TEST)
    table = pd.DataFrame([{"gate": "YOLOv8s only (Part 1 rules)", **rates(test, None)},
                          {"gate": f"YOLOv8s + CLIP rescue (τ={best})", **rates(test, best)}])
    print(table.round(3).to_string(index=False))
    print("YOLO rejection reasons on Pet test dogs:",
          test[(test.is_dog_true == 1) & (test.yolo_accept == 0)].reason.value_counts().to_dict())

    val.to_csv(OUT / "pets_trainval_sample.csv", index=False)
    test.to_csv(OUT / "pets_test_sample.csv", index=False)
    sweep.to_csv(OUT / "threshold_sweep_trainval.csv", index=False)
    table.to_csv(OUT / "gate_comparison_test.csv", index=False)
    json.dump({"threshold": best, "n_val": len(val), "n_test": len(test), "seed": SEED,
               "results_test": table.to_dict(orient="records")}, open(OUT / "summary.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
