"""Paired significance tests between emotion models on the SAME 1,986 test photos (official data/splits.csv order).

For each pair (A, B):
  * paired bootstrap (2,000 resamples of test photos) → 95% CI of macro-F1(B) − macro-F1(A)
  * exact McNemar test on per-photo correctness (photos where only one of the two is right)
These are REPORTING tests: no model is selected on test (selection happened on val, in the notebook).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "part3_multimodal" / "results"
LABELS = ["angry", "happy", "relaxed", "sad"]

MODELS = {
    "vit_part2": ("ViT-B/16 fine-tuned (Part 2, clean rerun)", RES / "part2_clean_rerun" / "probs_vit_test.npy"),
    "clip_run1": ("CLIP run 1 (GELU mismatch) WiSE-FT a=0.5", RES / "clip_finetune_run1" / "probs_clip_finetuned_test.npy"),
    "clip_run2": ("CLIP run 2 (QuickGELU) WiSE-FT a=0.75 + TTA, clean train", RES / "clip_finetune_run2" / "probs_clip_finetuned_test.npy"),
    "clip_run2_rec": ("CLIP run 2 + recovered dogs (FINAL)", RES / "clip_finetune_run2" / "probs_clip_recovered_test.npy"),
}
PAIRS = [("vit_part2", "clip_run2_rec"),     # final model vs Part 2's best
         ("clip_run2", "clip_run2_rec"),     # effect of the recovered training data (everything else identical)
         ("vit_part2", "clip_run1")]         # earlier run vs Part 2's best

test = pd.read_csv(REPO / "data" / "splits.csv").query("split == 'test'")
y = test.label.map({l: i for i, l in enumerate(LABELS)}).to_numpy()
pred = {k: np.load(p).argmax(1) for k, (_, p) in MODELS.items()}
f1 = lambda p, idx=slice(None): f1_score(y[idx], p[idx], average="macro")

rng = np.random.default_rng(42)
boots = [rng.integers(0, len(y), len(y)) for _ in range(2000)]
results = {"n_test": int(len(y)),
           "models": {k: {"name": n, "macro_f1": f1(pred[k]), "accuracy": float((pred[k] == y).mean()),
                          **{f"f1_{l}": float(v) for l, v in zip(LABELS, f1_score(y, pred[k], average=None))}}
                      for k, (n, _) in MODELS.items()},
           "pairs": []}
for a, b in PAIRS:
    diffs = np.array([f1(pred[b], i) - f1(pred[a], i) for i in boots])
    ca, cb = pred[a] == y, pred[b] == y
    only_b, only_a = int((~ca & cb).sum()), int((ca & ~cb).sum())
    results["pairs"].append({"A": a, "B": b, "delta_macro_f1": f1(pred[b]) - f1(pred[a]),
                             "bootstrap_95ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                             "share_bootstrap_delta_gt_0": float((diffs > 0).mean()),
                             "mcnemar": {"only_B_right": only_b, "only_A_right": only_a,
                                         "p_value": binomtest(only_b, only_b + only_a, 0.5).pvalue}})

for k, m in results["models"].items():
    print(f"{m['name']:58s} macro F1 {m['macro_f1']:.3f}  acc {m['accuracy']:.3f}  angry {m['f1_angry']:.2f}")
for r in results["pairs"]:
    ci = r["bootstrap_95ci"]
    print(f"\n{r['B']} vs {r['A']}: delta {r['delta_macro_f1']:+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
          f"McNemar {r['mcnemar']['only_B_right']} vs {r['mcnemar']['only_A_right']}, p = {r['mcnemar']['p_value']:.2g}")
(RES / "comparison").mkdir(exist_ok=True)
json.dump(results, open(RES / "comparison" / "significance_test.json", "w"), indent=1, default=float)
old = RES / "comparison" / "clip_vs_vit_test.json"
if old.exists():
    old.unlink()     # superseded by significance_test.json (it compared run 1 only)
