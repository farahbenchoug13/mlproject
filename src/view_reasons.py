import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

df = pd.read_csv("outputs/yolo_log.csv")
df = df[df.decision == "to-check"]

for reason, g in df.groupby("reason"):
    n = len(g)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, g.iterrows()):
        ax.imshow(Image.open(r.src).convert("RGB"))
        ax.set_title(f"{r.label} {r.best_dog_conf}", fontsize=8)
    fig.suptitle(reason)
    plt.tight_layout()
    fig.savefig(f"outputs/reason_{reason}.png", dpi=70)
    plt.close(fig)
print("done")