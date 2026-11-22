from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

CLEAN = Path("data/clean")

df = pd.DataFrame(
    [{"filepath": p.as_posix(), "label": p.parent.name}
     for p in sorted(CLEAN.rglob("*.jpg"))]
)
print(f"{len(df)} images")

# 70% train, 15% val, 15% test (stratified = same class proportions in each part)
train, temp = train_test_split(df, test_size=0.30, stratify=df.label, random_state=42)
val, test = train_test_split(temp, test_size=0.50, stratify=temp.label, random_state=42)

out = pd.concat([
    train.assign(split="train"),
    val.assign(split="val"),
    test.assign(split="test"),
]).sort_values("filepath")

# safety checks
assert out.filepath.is_unique, "a file appears twice"
assert all(Path(f).exists() for f in out.filepath), "a file is missing"

out.to_csv("data/splits.csv", index=False)
print(pd.crosstab(out.label, out.split, margins=True))
print("\nSaved: data/splits.csv")