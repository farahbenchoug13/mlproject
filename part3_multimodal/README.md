# 🐶 DogTranslator — Part 3: Multimodal (CLIP) + full pipeline + meme app

**Builds on:** Part 1 (data cleaning, YOLOv8 dog crops, `data/splits.csv`) by Farah and Part 2 (10-model benchmark) by Jamila.
**This part:** (1) CLIP experiments on the *same* split as Part 2, (2) an end-to-end pipeline photo/text → dog gate →
emotion → caption → meme, (3) a Gradio app, (4) measured evaluations of every pipeline stage.

> All numbers below come from files in `results/`; every decision was taken on validation, test was used for reporting.

---

## 1. Summary

| Question | Answer (clean test set, 1,986 photos unless stated) |
|---|---|
| Final emotion model | **CLIP ViT-B/16 (QuickGELU) fine-tuned + WiSE-FT α=0.75 + flip TTA, trained on clean train + 1,258 recovered dogs: macro F1 0.622, accuracy 0.640** |
| vs Part 2's best (ViT-B/16, clean rerun, 0.576) | **+4.5 points**, 95% bootstrap CI [+2.4, +6.5], McNemar p = 3·10⁻⁶ |
| Effect of the recovered training dogs (same run, same recipe) | **+2.6 points** (0.596 → 0.622), CI [+1.0, +4.2], p = 0.005; decided on val (+0.009) |
| CLIP with **no training** (zero-shot, simple prompts) | **0.570** — on par with Part 2's fine-tuned ViT (0.576) |
| CLIP frozen features + logistic regression | 0.591 — vs 0.414 for ResNet-50 frozen features (Part 2) |
| Hardest class | still *angry* (F1 0.52, Part 2 ViT: 0.47) |
| Dog gate (YOLO + CLIP second opinion) | accepts **99.7%** of dogs (YOLO alone 96.3%) with no extra cats let through — Oxford-IIIT Pet |
| Text → photo search | **P@10 = 0.75** for emotion queries vs 0.25 chance (test photos, simple queries) |
| Part 1 audit | **1,258 of 1,706** photos YOLO dropped as "no dog" are dogs per CLIP (≈92% precision on a manual check) |
| App reproduces the evaluation | same predicted emotion as the notebook on 200/200 test photos (max prob. difference 0.004) |

---

## 2. Shared contract (unchanged from Parts 1–2)
- Images `data/clean/` (YOLO crops), split `data/splits.csv` (train 9,266 / val 1,986 / test 1,986, stratified, seed 42).
- Every decision (prompt set, `C`, epoch, WiSE-FT α, TTA, keeping extra data) is taken on **val**; **test is evaluated once**.
- Main metric **macro F1**; class weights from `outputs/class_weights.csv`; label smoothing 0.1 (as Part 2).
- We never modify Part 1/Part 2 files or data. All Part 3 outputs live in `part3_multimodal/`.

---

## 3. CLIP experiments — `notebooks/DogTranslator_Part3_CLIP_FineTune.ipynb` (Kaggle GPU)

Two runs. **Run 1** built CLIP with the wrong activation (open_clip's `ViT-B-16` uses GELU; OpenAI's weights were trained
with QuickGELU — a warning in the log). **Run 2** fixes it (`ViT-B-16-quickgelu`) and adds the recovery ablation (§6).
Run 2 is the reference; run 1 is kept in `results/clip_finetune_run1/` for transparency.

| Stage (run 2) | Idea | Val macro F1 | Test macro F1 | Test acc. |
|---|---|---|---|---|
| Zero-shot, *simple* prompts ("a photo of a {emotion} dog.") — chosen on val | text ↔ image similarity | 0.576 | 0.570 | 0.593 |
| Zero-shot, body-language prompts | describe ears / mouth / posture | 0.539 | 0.532 | 0.583 |
| Zero-shot, ensemble (80 prompts) | average many templates | 0.536 | 0.539 | 0.552 |
| Frozen CLIP features + LogReg (C = 3, chosen on val) | Part 2's "frozen + LogReg" protocol | 0.612 | 0.591 | 0.609 |
| Fine-tuned image encoder (best epoch on val = 1) | cosine head initialised from text embeddings | 0.600 | 0.588 | 0.617 |
| + WiSE-FT α = 0.75 + flip TTA (both chosen on val) | average zero-shot and fine-tuned *weights* | 0.611 | 0.596 | 0.620 |
| **+ recovered dogs in train (kept: val +0.009 ≥ 0.005)** | same recipe, train 9,266 → 10,524 | **0.620** | **0.622** | **0.640** |

Part 2 references on the same test set (Jamila's clean rerun): ViT-B/16 0.576, ResNet-50 0.556, ResNet-50 frozen + LogReg 0.414.

**Findings**
1. **Language supervision helps.** Zero-shot CLIP — no training at all — matches Part 2's fine-tuned ViT (0.570 vs 0.576), and
   frozen CLIP features beat frozen ImageNet features by +18 points (0.591 vs 0.414): CLIP learned from image–caption pairs
   that describe moods, ImageNet features encode breeds (Part 2's t-SNE).
2. **The activation bug mattered for zero-shot** (0.538 → 0.570 once fixed) but not for fine-tuned models: fine-tuning adapts
   to whichever activation is used.
3. **Body-language prompts did not beat simple prompts** zero-shot: long descriptions mention cues (teeth, tongue) shared by
   *angry* and *happy* — exactly Part 2's main confusion.
4. **Fine-tuning overfits immediately** with the correct CLIP: best val at epoch 1, then train accuracy climbs (74% at epoch 4)
   while val drops. WiSE-FT (keeping 25% of the zero-shot weights) is what makes fine-tuning pay off.
5. **More (recovered) training data helped more than any training trick**: +2.6 test points, significant (p = 0.005), from
   photos Part 1's YOLO had wrongly rejected — see §6.
6. **Run-to-run variation is real.** Run 1 scored 0.632 on test (val 0.625) with the buggy activation; run 2's clean-train
   model scored 0.596 (val 0.611). Both sit within the noise of a single seed. We keep run 2 because its configuration is
   correct by construction (a bug fix is not an ablation choice), and within run 2 the recovered-data model wins on val.

**Significance** (`src/compare_models.py`, `results/comparison/significance_test.json`): paired bootstrap + McNemar on the
same test photos. Final vs Part 2 ViT: CLIP right on 252 photos where ViT is wrong, ViT right on 157 where CLIP is wrong.

## 4. Pipeline — `src/pipeline.py`

```
photo ──► dog gate ──► emotion model ──► caption writer ──► meme renderer
          YOLOv8s       CLIP fine-tuned    free LLM (Groq      Pillow, top/bottom
          + CLIP        (0.632 macro F1)   gpt-oss-120b) or    text with outline
          2nd opinion                      offline templates
text ───► CLIP search in 13,238 clean photos ──► emotion ──► caption (text = joke idea) ──► meme
photo + text ──► as photo, the text steers the caption
```

- **Dog gate** (`src/dog_gate.py`): Part 1's exact YOLO rules (`decide()` copied unchanged). Part 1's "to-check" cases need an
  app policy: `small_dog`/`multi_dog` → crop the (largest) dog; `low_conf`/`cat` → stop. New: when YOLO rejects, zero-shot CLIP
  scores {dog, cat, other} and can rescue close-up dogs YOLO misses (§6).
- **Emotion** (`src/emotion.py`): the final CLIP model (run 2 + recovered dogs) by default, Part 2's clean ViT as alternative;
  same preprocessing and TTA as the evaluation (verified: identical predictions on 200/200 test photos).
- **Captions** (`src/captioner.py`): the LLM never decides the emotion — it receives our prediction and must stay consistent.
  Free providers tried in order (Groq → Gemini → OpenRouter), then a template bank. Tunisian Darija (Arabizi), English, French.
  When the model hesitates (top probability < 0.45 or top-2 margin < 0.10), the caption jokes about it.
  Each meme uses a known **meme format** ("when…", "that face when", "Nobody: / Me:", "POV:", "Them: / Me:",
  "Me: / Also me:") plus five structures taken from six Tunisian memes the owner collected (only the joke structure;
  the TV-series images are not used): "Me after…" ("Ena ba3d ma…"), a question + "Me:" where the dog's face is the answer
  ("chnowa ra2yek f…? / Ena:"), "Everyone: … / My conscience: …", "OK I'll try… / *5 minutes later*", and the absurd answer drawn either as classic Impact text or as a modern white caption band; malformed or unsafe
  LLM output is rejected (→ next provider → templates). Darija: Gemini first; English/French: Groq first.
- **Caption safety** (the app is shown in class): the prompt forbids swear words, insults, "your mother" jokes, innuendo,
  body shaming, politics and religion; because a prompt alone is not a guarantee (Gemini did slip an insult in a demo),
  every LLM line also goes through a **blocklist filter** (Tounsi Arabizi, English, French; matched after lower-casing,
  removing accents, undoing tricks like `f***`, `f u c k`, `sh!t`, `zebbbi`, and Arabizi prefixes glued to a word such as
  `l3asba`, `ya7mar`) and Arabic-script output is refused.
  A user idea containing a blocked word is ignored and the app says so. Checked by `smoke_test.py` (32/32 bad lines
  blocked, incl. the "3asba" the owner reported and its variants; 0 of 302 template lines and 15 normal look-alikes such as
  "il fait la gueule", "nike shoes", "kalb 3asabi" or "khobz w zebda" wrongly blocked)
  and by `redteam_captions.py`: 24 ideas that invite insults ("roast my annoying neighbour", "my ex"…) in 3 languages →
  24 clean captions shown (3 answers rejected along the way, all for broken format) — `results/caption_safety/`.
  Limit: a blocklist only knows the words in it; new insults reported by users must be added to `_BLOCKLIST`.
  *Why gpt-oss-120b:* best free option on a 20-model Tunisian-darija test (90/100; vendor test, Nov 2026, not peer-reviewed).
- **Search** (`src/build_index.py`, `src/retrieval.py`): original (not fine-tuned) CLIP, because fine-tuning the image tower
  breaks its alignment with the text tower; embeddings pre-computed once.

---

## 5. Evaluation of the pipeline stages

### 5.1 Dog gate on independent labels (`src/eval_dog_gate.py`)
Data: Oxford-IIIT Pet (37 breeds, cat/dog labels by the dataset authors) — not `data/clean` or `data/rejected`, which were
produced by the same YOLO (circular). Threshold for the CLIP rescue chosen on 400 trainval photos, reported once on 1,000 test photos.

| Gate (Pet test: 694 dogs, 306 cats) | Dogs accepted | Cats wrongly accepted | Balanced accuracy |
|---|---|---|---|
| YOLOv8s only (Part 1 rules) | 96.3% | 6.9% | 0.947 |
| **YOLOv8s + CLIP rescue (τ = 0.90)** | **99.7%** | 6.9% | **0.964** |

- YOLO's 26 missed dogs: 21 "no dog found", 4 "cat more confident", 1 low confidence — close-up faces, as in Part 1's rejects.
  CLIP rescued 24 of them **without letting a single extra cat through**.
- The threshold curve on trainval is flat from 0.30 to 0.90 (all rejected dogs score very high), so we took the most
  conservative value, 0.90.
- Remaining error: the 6.9% of cats come from **YOLO itself** labelling a cat as a dog. CLIP is only asked when YOLO rejects;
  letting CLIP also veto YOLO's acceptances is a possible next step (not evaluated yet).

### 5.2 Text → photo search (`src/eval_retrieval.py`)
Fixed emotion queries (written before looking at results), searched among the 1,986 test photos only; a hit counts if its
dataset label matches the queried emotion. Chance = the class share in test.

| Query set | Chance | P@5 | P@10 | P@25 | P@50 |
|---|---|---|---|---|---|
| simple ("a photo of a happy dog") | 0.25 | 0.85 | **0.75** | 0.74 | 0.66 |
| body-language descriptions | 0.25 | 0.70 | 0.73 | 0.70 | 0.65 |

Per emotion (simple, P@10): angry 1.0 (chance 0.13), sad 0.8, happy 0.7, **relaxed 0.5** — relaxed is again the weak class,
mixed up with sad (calm postures), as in Parts 2 and 3. Labels are noisy, so true precision is likely a bit higher.

### 5.3 Speed (CPU, Docker, 8 threads)
YOLO gate 0.1–0.5 s · emotion 0.2 s · template caption < 1 ms · LLM caption ~1 s (network) · search < 0.2 s (pre-computed index).

---

## 6. Audit of Part 1's rejections — `src/audit_part1.py`
Scope: the 1,706 photos Part 1 dropped because YOLO saw **no usable dog** (`cats` 543, `removed` 910, `cat_more_confident` 126,
`low_conf` 127). Photos set aside on purpose (`small_dog`, `multi_dog`) are left alone. Nothing in `data/` is modified.

| Part 1 decision | Photos | CLIP says dog (P ≥ 0.90) |
|---|---|---|
| rejected: "cats" | 543 | 372 (69%) |
| rejected: no dog | 910 | 686 (75%) |
| to-check: cat more confident | 126 | 85 (67%) |
| to-check: low confidence | 127 | 115 (91%) |
| **total** | **1,706** | **1,258 (74%)** |

- **Is the diagnostic right?** Manual check of two random 48-photo sheets (`results/part1_audit/manual_audit.json`):
  of the photos CLIP calls dogs, ~44/48 clearly are (**precision ≈ 92%**; doubtful: a dog painting, blurry close-ups);
  of those it rejects, most are cats, sheep, cows, a lion, a giraffe, plush toys — the Kaggle set contains non-dogs
  labelled with dog emotions — and ~10/48 contain a small dog in a scene, which the strict threshold rightly leaves out.
- **Leak check:** perceptual hash (Part 1's rule, Hamming ≤ 5) against every clean image: **0** near-duplicates of val/test,
  0 label conflicts (verified: an image matches itself at 0 and a resized/recompressed copy at 0; recovered photos are ≥ 12 away).
- **Per emotion**, recovered dogs would grow the train set by angry +14.8%, happy +9.7%, relaxed +14.1%, sad +16.8%.
- **Train-only ablation** (notebook §7, run 2): same recipe; val 0.611 → 0.620 (+0.009 ≥ 0.005 → kept); test 0.596 → **0.622**
  (+2.6, 95% CI [+1.0, +4.2], McNemar p = 0.005). Gains on every class, most on *angry* (+3.4) and *relaxed* (+2.9).

This is feedback for Part 1 (YOLO misses close-up dog faces — also why *angry* lost the most images), not a change to it.

## 7. Run it
```bash
cp .env.example .env            # optional: add a free GROQ_API_KEY for AI captions
docker compose up --build       # → http://localhost:7860
# one-off jobs (same image, repo mounted):
docker compose --profile dev run --rm dev python part3_multimodal/src/build_index.py
docker compose --profile dev run --rm dev python part3_multimodal/src/eval_dog_gate.py
docker compose --profile dev run --rm dev python part3_multimodal/src/eval_retrieval.py
docker compose --profile dev run --rm dev python part3_multimodal/src/audit_part1.py
docker compose --profile dev run --rm dev python part3_multimodal/src/redteam_captions.py   # caption safety (uses API calls)
docker compose --profile dev run --rm dev python part3_multimodal/src/smoke_test.py --app http://host.docker.internal:7860/
```
Model files (gitignored, too large) go in `part3_multimodal/artifacts/`: `clip_emotion_ft.pt` (from the notebook) and optionally
`vit_finetuned_clean.pt` (Part 2).

---

## 8. Limitations
- One training run per configuration (seed 42); run-to-run variation is ~0.01–0.03 macro F1 (Jamila's ViT: 0.567 vs
  0.576; our CLIP runs 1 and 2). Differences below ~0.02 should not be over-interpreted; the two significance tests above
  are on a single run each.
- The test set keeps its label noise (never cleaned, to keep the exam honest).
- Emotions are subjective; the dataset labels come from the Kaggle author, not from behaviour experts.
- Body-language cues shown in the app are general canine cues for the predicted emotion, not an explanation of the model's pixels.
- The LLM caption quality in Darija is judged informally; the Darija benchmark we rely on is a single vendor test.
- Caption safety relies on a word list plus the prompt: an insult phrased with words not in the list can still pass.
  The template fallback (written by us) is always safe.

---

## 9. Files
```
part3_multimodal/
├── src/         pipeline modules + evaluation scripts (see §4–6)
├── notebooks/   DogTranslator_Part3_CLIP_FineTune.ipynb (current) · dogtranslator-part-3.ipynb (run 1, executed record)
├── assets/      examples/ — preset photos used by the app
├── artifacts/   model files + CLIP index (gitignored)
└── results/     clip_finetune_run1/ · clip_finetune_run2/ · part2_clean_rerun/ · comparison/ · dog_gate/ · retrieval/
                part1_audit/ · smoke_test/ (end-to-end validation report) · caption_safety/ · app_smoke/ (sample memes)
```
