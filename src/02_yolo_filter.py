import csv
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from ultralytics import YOLO

RAW = Path("data/raw")
CLEAN = Path("data/clean")
CHECK = Path("data/to_check")
REJECT = Path("data/rejected")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LIMIT = None         # None = all images (full run)         # None = all images (full run)
BATCH = 32
CONF_DET = 0.25      # minimum detection confidence
CONF_DOG = 0.30      # accept a dog from this confidence
CONF_CROP = 0.50     # below this, keep the full image (no crop)
MIN_AREA = 0.08      # dog box must cover >= 8% of the image
PAD = 0.10           # padding around the crop
PERSON, CAT, DOG = 0, 15, 16

model = YOLO("models/yolov8s.pt")

files = sorted(p for p in RAW.rglob("*") if p.suffix.lower() in EXT)
if LIMIT:
    random.seed(42)
    files = random.sample(files, min(LIMIT, len(files)))
print(f"{len(files)} images to process")


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def decide(dets, img_area):
    dogs = [d for d in dets if d["cls"] == DOG]
    cats = [d for d in dets if d["cls"] == CAT]
    strong = [d for d in dogs if d["conf"] >= CONF_DOG]
    best_dog = max((d["conf"] for d in dogs), default=0)
    best_cat = max((d["conf"] for d in cats), default=0)

    if not dogs:
        return ("cats" if cats else "removed"), None, "no_dog"
    if best_cat >= 0.5 and best_cat > best_dog:
        return "to-check", None, "cat_more_confident"
    if not strong:
        return "to-check", None, "low_conf"
    if len(strong) == 1:
        d = strong[0]
        if area(d["box"]) / img_area < MIN_AREA:
            return "to-check", None, "small_dog"
        return "dogs", d, ""
    # several dogs: keep the largest only if clearly dominant
    strong.sort(key=lambda d: area(d["box"]), reverse=True)
    if area(strong[0]["box"]) >= 2 * area(strong[1]["box"]):
        return "dogs", strong[0], ""
    return "to-check", None, "multi_dog"


def save_copy(p, root, label):
    dest = root / label
    dest.mkdir(parents=True, exist_ok=True)
    Image.open(p).convert("RGB").save(dest / f"{p.stem}.jpg", quality=95)


rows = []
for i in range(0, len(files), BATCH):
    batch = files[i:i + BATCH]
    results = model.predict([str(p) for p in batch], conf=CONF_DET,
                            classes=[PERSON, CAT, DOG], imgsz=640, verbose=False)
    for p, r in zip(batch, results):
        label = p.parent.name
        w, h = r.orig_shape[1], r.orig_shape[0]
        dets = [{"cls": int(c), "conf": float(cf), "box": [float(v) for v in b]}
                for c, cf, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
        decision, d, reason = decide(dets, w * h)
        out_path = ""

        if decision == "dogs":
            dest = CLEAN / label
            dest.mkdir(parents=True, exist_ok=True)
            out_path = dest / f"{p.stem}.jpg"
            img = Image.open(p).convert("RGB")
            if d["conf"] >= CONF_CROP:          # confident box -> crop
                x1, y1, x2, y2 = d["box"]
                pw, ph = (x2 - x1) * PAD, (y2 - y1) * PAD
                box = (max(0, x1 - pw), max(0, y1 - ph),
                       min(w, x2 + pw), min(h, y2 + ph))
                img = img.crop(tuple(map(int, box)))
            img.save(out_path, quality=95)      # else: keep the full image
        elif decision == "to-check":
            save_copy(p, CHECK, label)
        else:                                   # cats / removed
            save_copy(p, REJECT, label)

        rows.append({
            "src": str(p), "label": label, "decision": decision, "reason": reason,
            "n_dogs": sum(x["cls"] == DOG for x in dets),
            "n_cats": sum(x["cls"] == CAT for x in dets),
            "best_dog_conf": round(max((x["conf"] for x in dets if x["cls"] == DOG), default=0), 3),
            "cropped": bool(d and d["conf"] >= CONF_CROP),
            "clean_path": str(out_path),
        })
    print(f"{min(i + BATCH, len(files))}/{len(files)}", end="\r")

with open(OUT / "yolo_log.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=rows[0].keys())
    wr.writeheader()
    wr.writerows(rows)

df = pd.DataFrame(rows)
print("\n", pd.crosstab(df.label, df.decision, margins=True))
print("\nWhy to-check:\n", df[df.decision == "to-check"].groupby("reason").size())