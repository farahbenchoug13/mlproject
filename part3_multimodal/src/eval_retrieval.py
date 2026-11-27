"""Measure text → image search: for an emotion query, what share of the top-k test photos carry that label?

Fixed query set written before looking at results (no tuning), searched among the 1,986 TEST images only.
Chance level = share of the class in the test split. Labels are the (noisy) human dataset labels.

Run:  docker compose --profile dev run --rm dev python part3_multimodal/src/eval_retrieval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrieval  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "part3_multimodal" / "results" / "retrieval"
QUERIES = {
    "simple": {e: f"a photo of a {e} dog" for e in ["angry", "happy", "relaxed", "sad"]},
    "body_language": {
        "angry": "a photo of an angry dog baring its teeth, snarling, ears pinned back",
        "happy": "a photo of a happy dog with an open mouth, tongue out and bright eyes",
        "relaxed": "a photo of a calm relaxed dog resting with soft eyes and a loose body",
        "sad": "a photo of a sad dog with drooping ears, downcast eyes and head down",
    },
}
KS = [5, 10, 25, 50]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, idx = retrieval.load_index()
    share = idx[idx.split == "test"].label.value_counts(normalize=True)
    rows = []
    for qset, qs in QUERIES.items():
        for emo, q in qs.items():
            hits = retrieval.search(q, k=max(KS), split="test")
            labels = [h["label"] for h in hits]
            rows.append({"query_set": qset, "emotion": emo, "query": q, "chance": share[emo],
                         **{f"P@{k}": sum(l == emo for l in labels[:k]) / k for k in KS}})
    df = pd.DataFrame(rows)
    summary = df.groupby("query_set")[["chance"] + [f"P@{k}" for k in KS]].mean().reset_index()
    print(df.drop(columns="query").round(2).to_string(index=False), "\n\nmean over emotions:\n",
          summary.round(3).to_string(index=False))
    df.to_csv(OUT / "precision_at_k_test.csv", index=False)
    summary.to_csv(OUT / "precision_at_k_summary.csv", index=False)
    json.dump({"queries": QUERIES, "search_space": "test split (1,986 images)", "ks": KS}, open(OUT / "protocol.json", "w"), indent=1)


if __name__ == "__main__":
    main()
