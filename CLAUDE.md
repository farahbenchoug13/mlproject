# CLAUDE.md — DogTranslator (Part 3 · `Multimodal` branch)

Team ML course project. Detect a dog's emotion (**angry / happy / relaxed / sad**) from a photo, then turn it into a **meme**.
Part 1 (Farah) = data cleaning + YOLOv8 dog gate. Part 2 (Jamila) = 10-model emotion benchmark.
**This branch = Part 3 (us): CLIP experiments + full pipeline integration + Gradio app.** Status: complete, validated, pushed.

Docs: `README.md` (project), `part2_models/README.md`, **`part3_multimodal/README.md` (Part 3 report — the source for every
number)**, `docs/HANDOFF_MULTIMODAL.md` (original spec). Read them before big changes.

## Product
Three ways in, one meme out:
- **Photo** → YOLO dog gate (+ CLIP second opinion) → not a dog? stop with a fun "this is a cat / no dog" meme.
  Dog → crop → emotion model → caption → meme.
- **Description** → CLIP text-to-image search over `data/clean/` → pick a photo → same as above.
- **Photo + idea** → the idea steers the joke ("make it about Monday mornings").
Must be **technically solid** (honest metrics) **and fun** (real meme formats, not a dashboard).

## Hard rules (shared team contract)
- **Never modify** `data/`, `outputs/`, `src/` (Part 1) or `part2_models/` (Part 2). Read-only for us; everything of ours
  lives in `part3_multimodal/` (+ `app.py`, Docker files, docs). Only approved exception (2026-11-26, owner): the YOLO
  weights path in `src/02_yolo_filter.py` after moving them to `models/yolov8s.pt`. Ask before touching teammates' files.
- Evaluate on `data/splits.csv`: tune on `val`, report on `test` once. Never pick prompts/hparams on `test`. Never relabel.
- Primary metric = **macro F1** (+ accuracy, per-class F1, confusion matrix, paired bootstrap CI + McNemar for comparisons).
- Label order everywhere: `LABELS = ["angry", "happy", "relaxed", "sad"]`. Seed 42.
- YOLO gate settings = Part 1's `decide()` (`conf=0.25`, imgsz 640, `CONF_DOG=0.30`, `CONF_CROP=0.50`, `PAD=0.10`,
  `MIN_AREA=0.08`, largest dog box).
- `data/` (~31k tracked files): never `git add -A` / `git add .`; stage explicit paths. Large files (`*.pt`, `*.npy`
  embeddings) go in `part3_multimodal/artifacts/` (gitignored). `.env` holds API keys: gitignored, never commit or print it.
- Git: work on `Multimodal`, commit only when the owner asks, small themed commits, never push to `main`.

## Results (clean test, 1,986 photos) — details in `part3_multimodal/README.md`
| Model | Macro F1 |
|---|---|
| Part 2 best: ViT-B/16 FT (Jamila's clean re-run) | 0.576 |
| CLIP zero-shot (`ViT-B-16-quickgelu`) | 0.570 |
| CLIP FT + WiSE-FT 0.75 + TTA, clean train (run 2) | 0.596 |
| **FINAL: same + 1,258 recovered dogs in train only** | **0.622** (val 0.620) |
- Final vs ViT: +4.5, CI [+2.4, +6.5], p = 3e-6. Recovered vs not: +2.6, CI [+1.0, +4.2], p = 0.005.
- Run 1 (GELU bug) scored 0.632 test / 0.625 val; kept as a record only — the QuickGELU fix is a bug fix, not a choice,
  and within run 2 the decision was taken on val. Gap ≈ run-to-run noise (one seed).
- Dog gate on Oxford-IIIT Pet test: 99.7% dogs accepted (YOLO alone 96.3%), cats wrongly accepted 6.9% (both).
- Search: P@10 = 0.75 vs 0.25 chance (relaxed weakest, 0.5).
- Part 1 audit: 1,258 of 1,706 "no usable dog" rejects are dogs (≈92% manual precision), 0 leaks into val/test.

## Decisions
- **Runtime = Docker**: `docker compose up --build` → http://localhost:7860. One-off scripts (repo mounted):
  `MSYS_NO_PATHCONV=1 docker compose --profile dev run --rm dev python <script>` (the env var is needed in Git Bash).
  `data/clean`, `data/splits.csv`, `part3_multimodal/artifacts/` are mounted as volumes, never copied into the image.
  Host `python` is 3.14 without the ML stack — use Docker (plain-Python checks like the caption filter are fine on host).
- **Emotion model** (`emotion.py`): `artifacts/clip_emotion_ft.pt` (FINAL), fallback `vit_finetuned_clean.pt`; override
  with env `EMOTION_CKPT`. CLIP must be built as **`ViT-B-16-quickgelu`** (plain `ViT-B-16` silently uses GELU).
  Preprocessing = Resize(bicubic) + CenterCrop + flip TTA, exactly as the notebook (verified 200/200 identical predictions).
- **Captions = free LLMs** via OpenAI-compatible APIs (`captioner.py`): Darija → Gemini `gemini-flash-lite-latest` first,
  EN/FR → Groq `openai/gpt-oss-120b` first, then the other, then OpenRouter `:free`, then the **template bank** (always
  works offline). Keys in `.env` (format: `.env.example`). The Claude API needs prepaid credits → not used.
  Each meme gets a known meme format (when / that face when / Nobody / POV / Them-Me / Me after / Me-Also me); malformed answers
  (empty line, > 14 words, wrong-language format words, Arabic script) are retried with the next provider.
- **Caption safety (2026-11-27, owner request: Gemini produced an insult)**: strict family-friendly prompt + `_BLOCKLIST`
  (Tounsi Arabizi / EN / FR vulgarity and insults, matched after normalisation: accents, repeated letters, `f***`,
  `f u c k`, `sh!t`) on every LLM line; an unsafe user idea is ignored and the app shows a notice. When the owner reports a
  new bad word, add it to `_BLOCKLIST` and to `BAD` in `smoke_test.py`; add legit look-alikes to `GOOD`.
- **Languages**: Tunisian Darija (default, Latin script / Arabizi), English, French. Low confidence (top < 0.45 or
  top-2 margin < 0.10) → the caption jokes about the model hesitating.
- **Training on Kaggle** (notebook generated by `notebooks/make_notebook.py`); locally only inference and evaluation.
- **App presets** (`assets/examples/`): test photos the model gets right with ≥ 0.95 confidence and obvious cues
  (`SOURCES.json` says how they were chosen — disclose this if asked).

## Validation (must pass before any demo or commit touching the pipeline)
`docker compose up -d --build app`, then
`MSYS_NO_PATHCONV=1 docker compose --profile dev run --rm dev python part3_multimodal/src/smoke_test.py --app http://host.docker.internal:7860/`
→ must print **ALL PASSED (8/8)** (artifacts, warmup, presets, captions, caption safety, styles, search, app API).
`redteam_captions.py` re-checks LLM safety with provocative ideas (24 API calls). I cannot see the owner's browser:
say so when a UI change is only checked through the API.

## Layout
```
app.py  Dockerfile  docker-compose.yml  requirements.txt  .env.example  .gitignore  .dockerignore
models/yolov8s.pt            # shared YOLO weights (Part 1 filtering + Part 3 gate)
docs/HANDOFF_MULTIMODAL.md   # original Part 3 spec
layout_reference/            # UI inspiration screenshots
data/  outputs/  src/        # Part 1 (Farah) — read-only; data/ must never move (splits.csv paths)
part2_models/                # Part 2 (Jamila) — read-only
part3_multimodal/
  src/        pipeline: dog_gate  emotion  clip_common  build_index  retrieval  captioner  meme_maker  pipeline
              evaluation: eval_dog_gate  eval_retrieval  audit_part1  compare_models  smoke_test  redteam_captions
  notebooks/  DogTranslator_Part3_CLIP_FineTune.ipynb (+ make_notebook.py) · dogtranslator-part-3.ipynb (run 1 record)
  assets/     examples/ (app presets + SOURCES.json)
  artifacts/  model files + CLIP index (gitignored — share them via Drive, not git)
  results/    clip_finetune_run1/ clip_finetune_run2/ part2_clean_rerun/ comparison/ dog_gate/ retrieval/
              part1_audit/ smoke_test/ caption_safety/ app_smoke/
  README.md   Part 3 report
```
`pipeline.py` exposes plain functions (`run_image`, `run_text`, `search`, `warmup`) returning dicts; `app.py` is UI only.
Models load once (module-level cache).

## Rigor checklist (strict professor — every claim must survive questioning)
- Don't evaluate the dog gate on `data/clean` or Part 1's rejected folders (same YOLO → circular); Pet is the external set.
- Never report the retired raw-split ViT on the clean split (70% of clean test was in its training data).
- Recovered dogs went into **train only**; val/test untouched; phash leak check = 0.
- The LLM only writes captions; the emotion is ours and passed in. Body-language cues are general canine cues, not an
  explanation of the model's pixels.
- Limitations stated plainly: label noise, one seed per config, subjective emotions, informal Darija quality judgement,
  blocklist-based caption safety.

## Open items
- Tunisian meme references: owner collects them in `../tunisian_memes_references/` (`meme_text.md` + images, Arabic
  script). Use their **joke structure / wording** only (transliterate to Arabizi into `T["darija"]` or a new `FORMATS`
  entry, check with `is_safe`); never ship the images (TV-series stills, real people). Done so far: memes 1–2 →
  "me_after" format + 2 Darija tone examples (2026-11-27).
- Jamila should commit her clean re-run results herself before `../dogs part 2 jamila/` is deleted. The owner deleted
  `../clip_finetune_results*/` (copies of what is in `results/`); model `.pt` files exist only in `artifacts/`
  (gitignored) → keep a copy on Drive.
- Optional, not evaluated: let CLIP also veto YOLO's acceptances (would target the 6.9% cats).
