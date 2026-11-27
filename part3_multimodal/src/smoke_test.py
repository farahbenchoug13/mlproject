"""End-to-end validation of the whole pipeline. Exit code 0 = everything passed.

Checks: artifacts present · dog gate on the presets (4 dogs accepted, cat refused) · emotion on the presets ·
captions in 3 languages x {AI, offline} x {no idea, idea} (non-empty, safe, well-formed, offline varies) ·
both meme styles render · text search · optional: the running app's HTTP API (--app URL).

Run:  docker compose --profile dev run --rm dev python part3_multimodal/src/smoke_test.py [--app http://host.docker.internal:7860]
Writes part3_multimodal/results/smoke_test/report.json and a contact sheet of every generated meme.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import captioner  # noqa: E402
import meme_maker  # noqa: E402
import pipeline  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EX = REPO / "part3_multimodal" / "assets" / "examples"
OUT = REPO / "part3_multimodal" / "results" / "smoke_test"
results, memes = [], []


def check(name, fn):
    t = time.time()
    try:
        detail = fn()
        results.append({"check": name, "ok": True, "detail": detail, "sec": round(time.time() - t, 2)})
        print(f"PASS  {name}  {detail if detail else ''}")
    except Exception as e:
        results.append({"check": name, "ok": False, "detail": f"{type(e).__name__}: {e}", "sec": round(time.time() - t, 2)})
        print(f"FAIL  {name}  {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)


def artifacts():
    need = ["models/yolov8s.pt", "part3_multimodal/artifacts/clip_emotion_ft.pt",
            "part3_multimodal/artifacts/clip_clean_embeddings.npy", "part3_multimodal/artifacts/clip_clean_index.csv"]
    missing = [p for p in need if not (REPO / p).exists()]
    assert not missing, f"missing {missing}"
    return "all present"


def presets():
    got = {}
    for name in ["angry", "happy", "relaxed", "sad", "cat"]:
        r = pipeline.run_image(Image.open(EX / f"{name}.jpg"), lang="english", use_llm=False, seed=0)
        memes.append((f"preset {name}", r["meme"]))
        got[name] = r["emotion"]["emotion"] if r["emotion"] else r["gate"].status
    expected = {"angry": "angry", "happy": "happy", "relaxed": "relaxed", "sad": "sad", "cat": "cat"}
    assert got == expected, f"got {got}"
    return got


def captions():
    probs = {"angry": .03, "happy": .9, "relaxed": .05, "sad": .02}
    rows = []
    for lang in captioner.LANGS:
        for use_llm in (True, False):
            for idea in (None, "beach day"):
                c = captioner.generate_caption("happy", lang, probs=probs, idea=idea, use_llm=use_llm, seed=11)
                assert (c["top"] or c["bottom"]), "empty caption"
                assert captioner.is_safe(c["top"] + " " + c["bottom"]), "unsafe caption"
                assert c["style"] in ("classic", "bar")
                if c["source"] != "template":
                    assert captioner._well_formed(c["format"], lang, c), f"malformed {c}"
                rows.append(f"{lang}/{'AI' if use_llm else 'offline'}/{'idea' if idea else '-'}: {c['source'].split(':')[0]}")
                memes.append((f"{lang} {'AI' if use_llm else 'off'} {'idea' if idea else ''}",
                              meme_maker.render(Image.open(EX / "happy.jpg"), c)))
        offline = {captioner.generate_caption("happy", lang, probs=probs, idea="beach day", use_llm=False, seed=s)["bottom"]
                   for s in range(8)}
        assert len(offline) >= 3, f"offline idea captions don't vary in {lang}: {offline}"
    ai = sum("template" not in r for r in rows)
    return f"{len(rows)} captions OK, {ai} from an AI provider ({captioner.available_providers() or 'no keys'})"


# Lines that MUST be blocked (vulgar / insulting, incl. spelling tricks) and lines that must NOT (normal memes).
BAD = ["ya 7mar chbik", "ya zebi 3lik", "kouuuss omek", "nayek el jar", "9a7ba", "ya bhim", "ya wled el 7ram",
       "this is so f***ing good", "you idiot", "shut up human", "what the hell", "stupid mailman", "your mom",
       "ta gueule", "quel connard", "putain de lundi", "espèce d'abruti", "c'est de la merde", "ferme-la",
       "zebbbi", "f u c k", "sh!t happens", "Débile"]
GOOD = ["el kalb mte3i yheb el makla", "klab el 7ouma lkol", "nike shoes", "ice cream cone", "hello human",
        "il fait la gueule", "kousksi nhar el a7ad", "class is over", "beef jerky time", "7el el bab"]


def caption_safety():
    bank = [s for lang in captioner.T.values() for rows in lang.values() for row in rows for s in row]
    bank += [s for lang in captioner.IDEA_TEMPLATES.values() for rows in lang.values() for row in rows for s in row[:2]]
    false_pos = [s for s in bank + GOOD if not captioner.is_safe(s)]
    assert not false_pos, f"safe lines blocked: {false_pos}"
    leaked = [s for s in BAD if captioner.is_safe(s)]
    assert not leaked, f"unsafe lines NOT blocked: {leaked}"
    c = captioner.generate_caption("angry", "english", probs={"angry": .9, "happy": .04, "relaxed": .03, "sad": .03},
                                   idea="my stupid neighbour", use_llm=False, seed=0)
    assert c["idea_blocked"] and "stupid" not in (c["top"] + c["bottom"]).lower(), c
    return f"{len(bank)} template lines + {len(GOOD)} normal lines pass, {len(BAD)}/{len(BAD)} bad lines blocked, insulting idea ignored"


def styles():
    img = Image.open(EX / "sad.jpg")
    a = meme_maker.render(img, {"style": "classic", "top": "top", "bottom": "bottom"})
    b = meme_maker.render(img, {"style": "bar", "top": "Nobody:", "bottom": "Me: *test*"})
    assert a.width > 300 and b.height > a.height * 0.5
    return f"classic {a.size}, bar {b.size}"


def search():
    hits = pipeline.search("a happy dog running on the grass", k=4)
    assert len(hits) == 4 and all(Path(h["path"]).exists() for h in hits)
    r = pipeline.run_text("a grumpy pug judging you", lang="french", use_llm=False, seed=1)
    memes.append(("text: grumpy pug", r["meme"]))
    return f"top-4 labels {[h['label'] for h in hits]}, sims {[round(h['similarity'], 2) for h in hits]}"


def app_api(url):
    from gradio_client import Client, handle_file
    c = Client(url, verbose=False)
    out = c.predict(handle_file(str(EX / "angry.jpg")), "", "darija", True, api_name="/make_from_photo")
    assert any(isinstance(v, str) and "angry" in v for v in out), "angry not shown in app output"
    again = c.predict("english", True, api_name="/new_caption")          # same session: reuses the stored photo + emotion
    assert isinstance(again, str) and Path(again).exists(), f"'Another caption' failed: {again!r}"
    cat = c.predict(handle_file(str(EX / "cat.jpg")), "", "french", True, api_name="/make_from_photo")
    assert any(isinstance(v, str) and "cat, not a dog" in v for v in cat), "cat was not refused"
    out2 = c.predict("a sleepy dog on the sofa", "english", True, api_name="/search_and_make")
    assert out2 and out2[0], "search returned nothing"
    return "photo, another caption, cat refusal and text endpoints answer"


def sheet():
    th = []
    for label, m in memes:
        m = m.copy(); m.thumbnail((260, 300)); th.append(m)
    cols = 6
    rows = (len(th) + cols - 1) // cols
    W, H = 260, 300
    s = Image.new("RGB", (cols * W, rows * H), "white")
    for i, m in enumerate(th):
        s.paste(m, ((i % cols) * W, (i // cols) * H))
    s.save(OUT / "memes_sheet.jpg", quality=85)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", help="URL of a running app to test through its API")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    check("artifacts", artifacts)
    check("warmup", lambda: pipeline.warmup() or "models loaded")
    check("presets: gate + emotion", presets)
    check("captions: 3 languages x AI/offline x idea", captions)
    check("caption safety filter", caption_safety)
    check("meme styles", styles)
    check("text search", search)
    if args.app:
        check("running app API", lambda: app_api(args.app))
    sheet()
    ok = all(r["ok"] for r in results)
    json.dump({"all_passed": ok, "checks": results}, open(OUT / "report.json", "w"), indent=1, default=str)
    print(f"\n{'ALL PASSED' if ok else 'SOME CHECKS FAILED'} ({sum(r['ok'] for r in results)}/{len(results)})")
    sys.exit(0 if ok else 1)
