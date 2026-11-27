"""Dog gate: YOLOv8s dog detection + cropping, with exactly Part 1's rules (src/02_yolo_filter.py).

`decide()` is a copy of Part 1's decision function so the app and the cleaned dataset follow the same logic.
On top of it, the app adds a policy for Part 1's "to-check" cases (a user photo can't be set aside for review):

    Part 1 decision          app action
    dogs                     accept (crop if conf >= CONF_CROP)
    cats / no_dog            reject ("not a dog")
    cat_more_confident       reject (it's a cat)
    low_conf                 reject ("not sure this is a dog")
    small_dog                accept the dog, crop it (note shown to the user)
    multi_dog                accept the largest dog, crop it (note shown to the user)

CLIP second opinion (Part 3): YOLO often misses CLOSE-UP dog faces (a visual audit of 40 of Part 1's "cats" rejects
found ~25 dogs). When YOLO rejects a photo, zero-shot CLIP scores {dog, cat, other}; if P(dog) >= RESCUE_THRESHOLD
the full photo is accepted. The threshold is chosen on Oxford-IIIT Pet trainval and reported on its test split
(eval_dog_gate.py) — independent labels, not Part 1's YOLO decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import sys

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "models" / "yolov8s.pt"

# --- Part 1 constants (src/02_yolo_filter.py) — keep identical ---
CONF_DET = 0.25
CONF_DOG = 0.30
CONF_CROP = 0.50
MIN_AREA = 0.08
PAD = 0.10
IMGSZ = 640
PERSON, CAT, DOG = 0, 15, 16

CLIP_RESCUE = True
RESCUE_THRESHOLD = 0.90      # chosen on Oxford-IIIT Pet trainval (eval_dog_gate.py; results/dog_gate/summary.json)
GATE_PROMPTS = {
    "dog": ["a photo of a dog.", "a close-up photo of a dog's face.", "a photo of a puppy.", "a blurry photo of a dog."],
    "cat": ["a photo of a cat.", "a close-up photo of a cat's face.", "a photo of a kitten."],
    "other": ["a photo of a person.", "a photo of an object.", "a landscape photo.", "a cartoon drawing.",
              "a photo of a wild animal.", "a photo of a stuffed toy.", "a photo of a statue."],
}


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def decide(dets, img_area):
    """Part 1's decision function, unchanged. Returns (decision, chosen_detection, reason)."""
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
    strong.sort(key=lambda d: area(d["box"]), reverse=True)
    if area(strong[0]["box"]) >= 2 * area(strong[1]["box"]):
        return "dogs", strong[0], ""
    return "to-check", None, "multi_dog"


@dataclass
class GateResult:
    is_dog: bool
    status: str                 # accepted | not_a_dog | cat | unsure
    reason: str                 # Part 1 reason ("" when a clear single dog)
    part1_decision: str         # what Part 1's pipeline would have done with this photo
    dog_conf: float
    cat_conf: float
    n_dogs: int
    box: list[float] | None     # chosen dog box (xyxy, original pixels)
    crop: Image.Image | None    # image passed to the emotion model (None if rejected)
    cropped: bool
    note: str = ""
    detections: list[dict] = field(default_factory=list)
    clip_probs: dict | None = None   # CLIP second opinion {dog, cat, other}, only computed when YOLO rejects
    rescued: bool = False            # YOLO rejected it, CLIP confirmed a dog


@lru_cache(maxsize=1)
def _model():
    from ultralytics import YOLO
    return YOLO(str(WEIGHTS))


def _crop(img: Image.Image, box, w, h) -> Image.Image:
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * PAD, (y2 - y1) * PAD
    return img.crop(tuple(map(int, (max(0, x1 - pw), max(0, y1 - ph), min(w, x2 + pw), min(h, y2 + ph)))))


def detect(img: Image.Image) -> list[dict]:
    r = _model().predict(img, conf=CONF_DET, classes=[PERSON, CAT, DOG], imgsz=IMGSZ, verbose=False)[0]
    return [{"cls": int(c), "conf": float(cf), "box": [float(v) for v in b]}
            for c, cf, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]


@lru_cache(maxsize=1)
def _gate_text_embeddings():
    import torch.nn.functional as F
    from clip_common import encode_text
    return {k: F.normalize(encode_text(v).mean(0), dim=-1) for k, v in GATE_PROMPTS.items()}


def clip_species(img: Image.Image) -> dict:
    """Zero-shot CLIP probabilities over {dog, cat, other} (prompt-ensembled class embeddings)."""
    import torch
    from clip_common import encode_images, get_clip
    W = _gate_text_embeddings()
    f = encode_images([img])[0]
    scale = get_clip()[0].logit_scale.exp().item()
    p = (scale * torch.stack([f @ W[k] for k in W])).softmax(0)
    return {k: float(v) for k, v in zip(W, p)}


def run_gate(img: Image.Image, clip_rescue: bool | None = None) -> GateResult:
    """YOLO gate (Part 1 rules) + optional CLIP second opinion on rejected photos."""
    img = ImageOps.exif_transpose(img).convert("RGB")      # phone photos: respect EXIF rotation
    g = yolo_gate(img)
    if g.is_dog or not (CLIP_RESCUE if clip_rescue is None else clip_rescue):
        return g
    g.clip_probs = clip_species(img)
    if g.clip_probs["dog"] >= RESCUE_THRESHOLD:
        g.is_dog, g.status, g.rescued, g.crop, g.cropped = True, "accepted", True, img, False
        g.dog_conf = g.clip_probs["dog"]
        g.note = f"YOLO missed this dog (close-ups fool it); CLIP is {g.clip_probs['dog']:.0%} sure it's a dog."
    elif g.clip_probs["cat"] >= 0.5:
        g.status = "cat"
    return g


def yolo_gate(img: Image.Image) -> GateResult:
    w, h = img.size
    dets = detect(img)
    decision, chosen, reason = decide(dets, w * h)
    dogs = sorted((d for d in dets if d["cls"] == DOG), key=lambda d: area(d["box"]), reverse=True)
    dog_conf = max((d["conf"] for d in dogs), default=0.0)
    cat_conf = max((d["conf"] for d in dets if d["cls"] == CAT), default=0.0)
    base = dict(reason=reason, part1_decision=decision, dog_conf=dog_conf, cat_conf=cat_conf,
                n_dogs=len(dogs), detections=dets)

    if decision in ("cats", "removed") or reason == "cat_more_confident":
        status = "cat" if cat_conf > 0 else "not_a_dog"
        return GateResult(False, status, box=None, crop=None, cropped=False, **base)
    if reason == "low_conf":
        return GateResult(False, "unsure", box=None, crop=None, cropped=False,
                          note="I can barely see a dog here.", **base)

    note = ""
    strong = [d for d in dogs if d["conf"] >= CONF_DOG]          # largest first
    if reason == "small_dog":
        chosen, note = strong[0], "The dog is small in the photo, so I zoomed in on it."
    elif reason == "multi_dog":
        chosen, note = strong[0], f"I see {len(strong)} dogs, so I read the biggest one."
    # accepted: crop if confident (Part 1 rule); small/multi-dog user photos are always cropped to isolate the dog
    do_crop = chosen["conf"] >= CONF_CROP or reason in ("small_dog", "multi_dog")
    crop = _crop(img, chosen["box"], w, h) if do_crop else img
    return GateResult(True, "accepted", box=chosen["box"], crop=crop, cropped=do_crop, note=note, **base)


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        g = run_gate(Image.open(p))
        print(f"{p}: is_dog={g.is_dog} status={g.status} reason={g.reason or '-'} part1={g.part1_decision} "
              f"dog={g.dog_conf:.2f} cat={g.cat_conf:.2f} n_dogs={g.n_dogs} cropped={g.cropped} {g.note}")
