import random
import shutil
from pathlib import Path

import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

CLEAN = Path("data/clean")
DUPES = Path("data/duplicates")
CONF = Path("data/conflicts")
OUT = Path("outputs")
THR = 5        # max difference (out of 64) to call two images "the same"
MOVE = True   # False = only look (safe), True = really move files

files = sorted(CLEAN.rglob("*.jpg"))
labels = [p.parent.name for p in files]
n = len(files)
print(f"{n} images, computing hashes...")
H = np.array([imagehash.phash(Image.open(p)).hash.flatten() for p in files],
             dtype=np.float32)

# find similar pairs
pairs = []
for s in range(0, n, 1000):
    A = H[s:s + 1000]
    dist = A @ (1 - H).T + (1 - A) @ H.T          # number of different bits
    for a in range(len(A)):
        i = s + a
        for j in np.where(dist[a] <= THR)[0]:
            if j > i:
                pairs.append((i, int(j), int(dist[a, j])))

conflict, dup = set(), set()
for i, j, d in pairs:
    if labels[i] != labels[j]:
        conflict.update([i, j])    # same photo, different labels
    else:
        dup.add(j)                 # keep i, remove j
dup -= conflict

pd.DataFrame([{"a": files[i].as_posix(), "b": files[j].as_posix(),
               "dist": d, "label_a": labels[i], "label_b": labels[j]}
              for i, j, d in pairs]).to_csv(OUT / "dedupe_pairs.csv", index=False)

print(f"similar pairs: {len(pairs)}")
print(f"duplicates to remove: {len(dup)} | conflicts (both removed): {len(conflict)}")
print("duplicates per class:", pd.Series([labels[i] for i in dup]).value_counts().to_dict())
print("conflicts per class:", pd.Series([labels[i] for i in conflict]).value_counts().to_dict())

# picture of 10 random pairs (same label first, then conflicts)
same = [x for x in pairs if labels[x[0]] == labels[x[1]]]
diff = [x for x in pairs if labels[x[0]] != labels[x[1]]]
random.seed(42)
show = random.sample(same, min(8, len(same))) + random.sample(diff, min(4, len(diff)))
if show:
    fig, axes = plt.subplots(len(show), 2, figsize=(5, 2.6 * len(show)))
    axes = np.atleast_2d(axes)
    for row, (i, j, d) in zip(axes, show):
        for ax, k in zip(row, (i, j)):
            ax.imshow(Image.open(files[k]).convert("RGB"))
            ax.set_title(f"{labels[k]} (dist {d})", fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    fig.savefig(OUT / "dedupe_pairs.png", dpi=70)
    print("picture saved: outputs/dedupe_pairs.png")

if MOVE:
    for idx_set, root in ((conflict, CONF), (dup, DUPES)):
        for i in idx_set:
            dest = root / labels[i]
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(files[i]), str(dest / files[i].name))
    print("files moved.")
else:
    print("MOVE = False: nothing was moved.")