"""Emotion engine: dog crop -> probabilities over LABELS.

Swapping the model = changing one path (env var EMOTION_CKPT or DEFAULT_CKPT below). Two checkpoint formats:
  * timm ViT state_dict (Part 2):          timm.create_model("vit_base_patch16_224", num_classes=4)
  * CLIP fine-tuned dict (Part 3 notebook): {"arch", "pretrained", "state_dict", "mean", "std", "tta", ...}

The raw-split ViT (trained on my_split.csv, 70% of the clean test images in its training set) is retired:
never use it nor score it on data/splits.csv.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO / "part3_multimodal" / "artifacts"
LABELS = ["angry", "happy", "relaxed", "sad"]
# Preference order (best clean-split test macro F1 first; all trained on data/splits.csv train, selected on val):
#   clip_emotion_ft.pt      CLIP ViT-B/16 fine-tuned + WiSE-FT (Part 3)   0.632
#   vit_finetuned_clean.pt  ViT-B/16 fine-tuned (Part 2, clean rerun)      0.576
CKPT_PREFERENCE = [ARTIFACTS / "clip_emotion_ft.pt", ARTIFACTS / "vit_finetuned_clean.pt"]
DEFAULT_CKPT = next((p for p in CKPT_PREFERENCE if p.exists()), CKPT_PREFERENCE[-1])
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class CLIPEmotion(nn.Module):
    """Same module as in DogTranslator_Part3_CLIP_FineTune.ipynb."""
    def __init__(self, visual, W, s):
        super().__init__()
        self.visual = visual
        self.head = nn.Parameter(W.clone())
        self.register_buffer("scale", torch.tensor(float(s)))

    def forward(self, x):
        return self.scale * F.normalize(self.visual(x).float(), dim=-1) @ F.normalize(self.head, dim=-1).T


class EmotionModel:
    def __init__(self, ckpt_path: str | Path):
        self.path = Path(ckpt_path)
        obj = torch.load(self.path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "arch" in obj and "state_dict" in obj:
            import open_clip
            clip = open_clip.create_model(obj["arch"], pretrained=None)
            sd = {k: v.float() if v.is_floating_point() else v for k, v in obj["state_dict"].items()}
            self.net = CLIPEmotion(clip.visual, sd["head"], sd["scale"])
            self.net.load_state_dict(sd)
            size, mean, std = obj["image_size"], obj["mean"], obj["std"]
            self.tta, self.kind = bool(obj.get("tta", False)), f"CLIP {obj['arch']} fine-tuned"
            # same eval transform as the notebook: shorter side → size (bicubic), then center crop
            resize = [T.Resize(size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(size)]
        else:
            import timm
            self.net = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(LABELS))
            self.net.load_state_dict(obj)
            size, mean, std = 224, IMAGENET_MEAN, IMAGENET_STD
            self.tta, self.kind = False, "ViT-B/16 fine-tuned (Part 2)"
            resize = [T.Resize((size, size))]          # Part 2's make_tf(kind="eval"): plain square resize
        self.net.eval()
        self.tf = T.Compose(resize + [T.ToTensor(), T.Normalize(mean, std)])

    @torch.inference_mode()
    def predict(self, img: Image.Image) -> dict[str, float]:
        x = self.tf(img.convert("RGB")).unsqueeze(0)
        p = self.net(x).float().softmax(-1)
        if self.tta:
            p = (p + self.net(torch.flip(x, dims=[3])).float().softmax(-1)) / 2
        return {l: float(v) for l, v in zip(LABELS, p[0])}


@lru_cache(maxsize=1)
def get_model() -> EmotionModel:
    return EmotionModel(os.environ.get("EMOTION_CKPT", DEFAULT_CKPT))


def predict(img: Image.Image) -> dict:
    """Returns {"probs": {...}, "emotion": top label, "confidence": top prob, "margin": top1-top2, "model": name}."""
    m = get_model()
    probs = m.predict(img)
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    return {"probs": probs, "emotion": ranked[0][0], "confidence": ranked[0][1],
            "runner_up": ranked[1][0], "margin": ranked[0][1] - ranked[1][1], "model": m.kind}


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        r = predict(Image.open(p))
        print(p, r["emotion"], f"{r['confidence']:.2f}", {k: round(v, 2) for k, v in r["probs"].items()})
