"""DogTranslator pipeline — plain functions used by app.py (UI-free, testable).

Mode A  image            : dog gate (YOLO) → emotion → caption → meme
Mode B  text             : CLIP search in data/clean → emotion → caption (query = joke idea) → meme
Mode C  image + text     : like A, the text steers the caption
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import captioner  # noqa: E402
import dog_gate  # noqa: E402
import emotion  # noqa: E402
import meme_maker  # noqa: E402
import retrieval  # noqa: E402


def _caption_and_meme(img, emo, lang, idea, use_llm, seed, timings):
    t = time.perf_counter()
    cap = captioner.generate_caption(emo["emotion"], lang, probs=emo["probs"], idea=idea, use_llm=use_llm, seed=seed)
    timings["caption_ms"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    meme = meme_maker.render(img, cap)
    timings["render_ms"] = (time.perf_counter() - t) * 1000
    return cap, meme


def run_image(img: Image.Image, idea: str | None = None, lang: str = "darija", use_llm: bool = True,
              seed: int | None = None) -> dict:
    timings = {}
    t = time.perf_counter()
    gate = dog_gate.run_gate(img)
    timings["gate_ms"] = (time.perf_counter() - t) * 1000
    out = {"mode": "image+text" if idea else "image", "gate": gate, "emotion": None, "timings": timings}

    if not gate.is_dog:                      # stop here: no emotion model on a non-dog (closed-set fix from Part 2)
        cap = captioner.generate_caption(gate.status, lang, seed=seed)
        out.update(caption=cap, meme=meme_maker.render(img, cap))
        return out

    t = time.perf_counter()
    emo = emotion.predict(gate.crop)
    timings["emotion_ms"] = (time.perf_counter() - t) * 1000
    cap, meme = _caption_and_meme(gate.crop, emo, lang, idea, use_llm, seed, timings)
    out.update(emotion=emo, caption=cap, meme=meme, body_cues=captioner.BODY_CUES[emo["emotion"]])
    return out


def search(query: str, k: int = 4) -> list[dict]:
    t = time.perf_counter()
    hits = retrieval.search(query, k=k)
    for h in hits:
        h["search_ms"] = (time.perf_counter() - t) * 1000
    return hits


def run_text(query: str, lang: str = "darija", use_llm: bool = True, k: int = 4, pick: int = 0,
             hits: list[dict] | None = None, seed: int | None = None) -> dict:
    """Search, then make a meme from hit `pick`. Dataset photos are YOLO crops already (accepted by Part 1's gate)."""
    timings = {}
    t = time.perf_counter()
    hits = hits or retrieval.search(query, k=k)
    timings["search_ms"] = (time.perf_counter() - t) * 1000
    chosen = hits[pick]
    img = Image.open(chosen["path"]).convert("RGB")
    t = time.perf_counter()
    emo = emotion.predict(img)
    timings["emotion_ms"] = (time.perf_counter() - t) * 1000
    cap, meme = _caption_and_meme(img, emo, lang, query, use_llm, seed, timings)
    return {"mode": "text", "hits": hits, "chosen": chosen, "emotion": emo, "caption": cap, "meme": meme,
            "body_cues": captioner.BODY_CUES[emo["emotion"]], "timings": timings}


def warmup():
    """Load every model once at startup (YOLO, emotion, CLIP + index) so the first request is fast."""
    dog_gate._model()
    emotion.get_model()
    try:
        retrieval.load_index()
        retrieval.encode_text(["warmup"])
    except FileNotFoundError as e:
        print(f"[pipeline] text search disabled: {e}")
