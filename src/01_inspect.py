import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

RAW = Path("data/raw")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
random.seed(42)

by_class = defaultdict(list)
for p in RAW.rglob("*"):
    if p.suffix.lower() in EXT:
        by_class[p.parent.name].append(p)

print("Images per class folder:")
for c, files in sorted(by_class.items()):
    print(f"  {c}: {len(files)}")

# Broken files and sizes
bad, small, sizes = [], [], []
for c, files in by_class.items():
    for p in files:
        try:
            with Image.open(p) as im:
                w, h = im.size
            sizes.append((w, h))
            if min(w, h) < 100:
                small.append(p)
        except Exception:
            bad.append(p)
print(f"\nUnreadable: {len(bad)} | Smaller than 100px: {len(small)}")
if sizes:
    ws, hs = zip(*sizes)
    print(f"Width {min(ws)}-{max(ws)}, height {min(hs)}-{max(hs)}")

# Contact sheets
for c, files in by_class.items():
    sample = random.sample(files, min(50, len(files)))
    fig, axes = plt.subplots(5, 10, figsize=(20, 10))
    for ax in axes.flat:
        ax.axis("off")
    for ax, p in zip(axes.flat, sample):
        try:
            ax.imshow(Image.open(p).convert("RGB"))
            ax.set_title(p.name[:12], fontsize=6)
        except Exception:
            pass
    fig.suptitle(c)
    plt.tight_layout()
    fig.savefig(OUT / f"inspect_{c}.png", dpi=80)
    plt.close(fig)
print("Contact sheets saved in outputs/")