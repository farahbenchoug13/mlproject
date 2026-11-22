from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
OUT = Path("outputs")


def count_dir(root):
    r = Path(root)
    if not r.exists():
        return pd.Series(dtype=int)
    return pd.Series({d.name: sum(p.suffix.lower() in EXT for p in d.rglob("*"))
                      for d in r.iterdir() if d.is_dir()})


raw = count_dir("data/raw")
log = pd.read_csv(OUT / "yolo_log.csv")
if len(log) != raw.sum():
    print(f"WARNING: yolo_log has {len(log)} rows but raw has {raw.sum()} images "
          "(log may come from the 200-image test)")

splits = pd.read_csv("data/splits.csv")
dec = pd.crosstab(log.label, log.decision)

tbl = pd.DataFrame({
    "raw": raw,
    "yolo_cats_removed": dec.get("cats", 0) + dec.get("removed", 0),
    "yolo_to_check": dec.get("to-check", 0),
    "after_yolo": dec.get("dogs", 0),
    "duplicates": count_dir("data/duplicates"),
    "conflicts": count_dir("data/conflicts"),
    "clean_final": count_dir("data/clean"),
}).fillna(0).astype(int)
tbl = pd.concat([tbl, pd.crosstab(splits.label, splits.split)[["train", "val", "test"]]], axis=1)
tbl["kept_%"] = (100 * tbl.clean_final / tbl.raw).round(1)
tbl.loc["TOTAL"] = tbl.sum(numeric_only=True)
tbl.loc["TOTAL", "kept_%"] = round(100 * tbl.loc["TOTAL", "clean_final"] / tbl.loc["TOTAL", "raw"], 1)
print(tbl.to_string())
tbl.to_csv(OUT / "stats_table.csv")

# class weights from the TRAIN set only
tr = splits[splits.split == "train"].label.value_counts().sort_index()
weights = (tr.sum() / (len(tr) * tr)).round(3)
print("\nClass weights (train):")
print(weights.to_string())
weights.rename("weight").to_csv(OUT / "class_weights.csv")

# chart: raw vs clean
t = tbl.drop("TOTAL")
ax = t[["raw", "clean_final"]].plot.bar(figsize=(7, 4), title="Images per class: raw vs clean")
ax.set_ylabel("images")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT / "class_counts.png", dpi=120)
print("\nSaved: stats_table.csv, class_weights.csv, class_counts.png in outputs/")