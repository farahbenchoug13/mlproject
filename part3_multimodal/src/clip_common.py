"""Shared zero-shot CLIP (image + text towers) used for text→image search.

Retrieval must use the ORIGINAL CLIP: fine-tuning the image tower for emotion breaks its alignment with the text tower.
Same architecture/weights as the Part 3 notebook's zero-shot stage (ViT-B-16, openai).
"""
from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

CLIP_ARCH, CLIP_PRETRAINED = "ViT-B-16-quickgelu", "openai"   # OpenAI weights were trained with QuickGELU


@lru_cache(maxsize=1)
def get_clip():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_ARCH, pretrained=CLIP_PRETRAINED)
    model.eval()
    return model, preprocess, open_clip.get_tokenizer(CLIP_ARCH)


@torch.inference_mode()
def encode_text(texts: list[str]) -> torch.Tensor:
    model, _, tok = get_clip()
    return F.normalize(model.encode_text(tok(texts)).float(), dim=-1)


@torch.inference_mode()
def encode_images(images) -> torch.Tensor:
    model, preprocess, _ = get_clip()
    x = torch.stack([preprocess(im.convert("RGB")) for im in images])
    return F.normalize(model.encode_image(x).float(), dim=-1)
