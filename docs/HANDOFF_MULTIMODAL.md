# 🐶 DogTranslator — Part 3 (`Multimodal`) & Full Pipeline Handoff

> **Purpose:** Complete technical & design specification for **Person 3 (Multimodal)** on the `Multimodal` branch, ready to be used directly with **Claude Code**.
> **Goal:** Combine **Part 1 (YOLOv8 dog gate)**, **Part 2 (emotion classification)**, and **Part 3 (CLIP + VLM captions)** into both:
> 1. **Academic ML experiments** (comparable to Part 2's `0.567` clean Macro F1 baseline), and
> 2. **An interactive, fun Gradio web app** that turns dog photos or text prompts into **memes**.

---

## 1. Existing Repository Context & Shared Contracts

Do **not** modify Part 1 outputs or relabel the test set—every model in the team is evaluated on the exact same contract:

- **Clean cropped images:** `data/clean/<emotion>/` (`13,238` images across `angry`, `happy`, `relaxed`, `sad`).
- **Official split:** `data/splits.csv` — stratified 70/15/15 split (`seed=42`):
  - `train`: `9,266` images
  - `val`: `1,986` images (use for prompt tuning, hyperparams, and ablation decisions)
  - `test`: `1,986` images (report only, never tune on it)
- **Train class weights:** `outputs/class_weights.csv` (`angry: 1.934`, `happy: 0.799`, `relaxed: 0.927`, `sad: 0.868`).
- **YOLOv8 weights:** `models/yolov8s.pt`, with decision logic in `src/02_yolo_filter.py` (`CONF_DET=0.25`, `CONF_DOG=0.30`, `CONF_CROP=0.50`, `MIN_AREA=0.08`, `PAD=0.10`).
- **Part 2 Baselines to beat (Clean Test Set, before optimization):**
  - `Dummy`: `0.119` Macro F1 (`0.313` Acc)
  - `ResNet-50 frozen + LogReg`: `0.414` Macro F1 (`0.440` Acc)
  - `ResNet-50 fine-tuned`: `0.558` Macro F1 (`0.579` Acc)
  - `ViT-B/16 fine-tuned`: **`0.567` Macro F1** (`0.590` Acc)
- **Why Part 2 scores were ~0.57:** Rapid overfitting after 2–3 epochs, ImageNet breed bias (t-SNE clusters by breed rather than emotion), noisy labels in the dataset, `angry`↔`happy` (open mouth) & `relaxed`↔`sad` (calm posture) confusions, and **closed-set behavior** (predicts an emotion even on cats).
- **Part 2 Optimization Plan (B1–B5):** Confident-learning label filtering on `train` only (`C=0.005`, max 5% removed per class), balanced sampling, `TrivialAugment` + random erasing, and horizontal-flip TTA. **Keep this plan; do not manually relabel the dataset.**

---

## 2. Part 3 ML Tasks (CLIP + Captions)

### Task 3.1 — CLIP Zero-Shot & Prompt Engineering
1. **Quick smoke test:** Run zero-shot CLIP (`ViT-B/32` or `ViT-B/16` via `open_clip` or `transformers`) on 200 raw/val images immediately (no training needed).
2. **Prompt ablation (on `val` split, then final score on `test`):**
   - **Simple prompts:** e.g., `"a photo of an angry dog"`, `"a photo of a happy dog"`, `"a photo of a relaxed dog"`, `"a photo of a sad dog"`.
   - **Body-language prompts:** descriptive canine cues targeting the `angry`↔`happy` and `relaxed`↔`sad` confusions:
     - `angry`: *"a photo of an angry aggressive dog baring its teeth, snarling, ears pinned back, tense body"*
     - `happy`: *"a photo of a happy excited dog smiling with relaxed mouth, tongue out, bright eyes, playful posture"*
     - `relaxed`: *"a photo of a calm relaxed dog resting comfortably with soft eyes and loose body"*
     - `sad`: *"a photo of a sad worried dog with drooping ears, downcast eyes, and low slumped posture"*
   - **Prompt ensembling:** Average text embeddings across 5–10 templates per class.
   - **Closed-set test:** Optionally test adding a `"not a dog"` / `"a photo of a cat or object"` prompt on cat/rejected images from `data/rejected/`.

### Task 3.2 — CLIP Linear Probe & Fine-Tuning
- **Linear Probe:** Extract frozen CLIP image embeddings on `data/splits.csv` and train Logistic Regression / Ridge / MLP. Compare directly against Part 2's `ResNet-50 frozen + LogReg (0.414 Macro F1)`.
- **Fine-Tuned CLIP:** Fine-tune CLIP image encoder (or image-text cosine head) on `train` using `outputs/class_weights.csv` (or balanced sampling + strong augmentation from Part 2's B2/B3), select best checkpoint on `val` Macro F1, and report `test` Macro F1, accuracy, and per-class F1 (`angry`, `happy`, `relaxed`, `sad`) against `ViT-B/16 (0.567)`.

### Task 3.3 — CLIP Retrieval Index (for Text-to-Image Search)
- Pre-compute normalized CLIP image embeddings for all `13,238` images in `data/clean/` once and save them (e.g., `part3_multimodal/artifacts/clip_clean_embeddings.npy` + `clip_clean_index.csv`).
- At runtime, encode user text with CLIP and use cosine similarity to return the top-k matching dog photos in milliseconds.

### Task 3.4 — Caption & Meme Generation
- Use a Vision-Language Model (**BLIP-2 / Qwen-VL / LLaVA**, e.g. via HuggingFace or API) prompted with the image + detected emotion to generate a short, expressive meme caption or body-language subtitle.
- **Template Fallback (mandatory):** Always provide a fast, deterministic template bank conditioned on `(emotion, confidence, top_matching_body_language_cue)` so the pipeline works smoothly on CPU or when VLM memory is limited.
- **Deliverable for report/slides:** Save 10–20 curated meme/caption examples (including clear wins, `angry`/`happy` edge cases, and an ambiguous case).

---

## 3. End-to-End Integrated Pipeline Architecture

### Flow A: Upload Image → Dog Gate → Emotion → Meme
1. **Input:** User uploads any image (dog, cat, person, object, meme).
2. **Step 1 — Dog Gate (Part 1 `models/yolov8s.pt`):**
   - Reuse the detection logic from `src/02_yolo_filter.py` (`PERSON=0`, `CAT=15`, `DOG=16`).
   - **If no valid dog is found** (e.g., cat, no dog, or `cat_more_confident`): **STOP** and return a clear message (*"No dog detected — DogTranslator only translates dogs!"*) + what YOLO saw (e.g. `cat: 0.91`).
   - **If dog is detected:** Crop around the dominant dog bounding box (`PAD=0.10` if `conf >= 0.50`, else full image).
3. **Step 2 — Emotion Classification (Part 2 + Part 3):**
   - Run the cropped dog image through the chosen best emotion model (Part 3 fine-tuned CLIP / Part 2 ViT-B/16).
   - Output predicted emotion (`angry`, `happy`, `relaxed`, `sad`), class probabilities, and top matching body-language description.
4. **Step 3 — Meme Generation (Part 3):**
   - Generate meme text (VLM caption or template fallback, plus optional user custom override).
   - Render classic top/bottom meme text (Pillow `ImageDraw`, bold uppercase Impact/sans-serif with black outline) onto the image and return it for preview & download.

### Flow B: Text Prompt → Image Search → Meme
1. **Input:** User enters a text prompt (e.g., *"a grumpy pug judging you"* or *"happy dog running on grass"*).
2. **Step 1 — CLIP Retrieval:** Search pre-computed `data/clean/` CLIP embeddings via cosine similarity and pick the top match (or top-4 gallery for the user to choose from).
3. **Step 2 — Emotion & Meme Generation:** Run Emotion Engine + Caption/Meme Generator on the selected dog image and display the final meme.

---

## 4. Gradio Frontend & Visual Design Spec

Inspired by the screenshots in `layout_reference/`:

### Color Palette & Styling (Gradio Custom CSS)
- **Background (Cream / Sand):** `#F2E8D5`
- **Primary Dark (Deep Navy / Midnight Teal):** `#0B2B3C` (headings, primary dark buttons, contrast panels)
- **Primary Accent (Vibrant Burnt Orange):** `#ED5800` (highlighted headline words, primary CTAs, active rings)
- **Secondary Accent (Teal):** `#148F9E` (feature badges, info cards, pill tags)
- **Highlight (Mustard Yellow):** `#F4B400` (badges, playful accents)
- **Typography & Shapes:** Bold uppercase display headings with mixed navy + orange words, rounded pill buttons (`border-radius: 9999px`), circular pet thumbnails (`border-radius: 50%`) with thick white/orange active borders, and clean card containers.

### Suggested `gr.Blocks` Layout
1. **Hero Banner:**
   - Left: Eyebrow badge (`Multimodal Canine Emotion & Meme Studio`), bold 2-color title (`DECODE YOUR DOG'S MOOD. MAKE A MEME.`), short description of the 3-stage AI pipeline (YOLOv8 → ViT/CLIP → VLM Meme).
   - Right: Featured expressive dog visual / live preview card.
2. **Interactive Emotion / Category Strip (inspired by `layout_reference/Capture d'écran 2026-11-22 230244.png`):**
   - Circular clickable emotion presets (`Angry`, `Happy`, `Relaxed`, `Sad`, plus a `Not-a-Dog Test` cat example) that populate sample inputs in one click.
3. **Main Workspace (`gr.Tabs`):**
   - **Tab 1 — `📷 Photo → Dog Gate → Meme`:**
     - Left column: Image upload + optional toggle (`Use VLM Caption` vs `Fast Meme Templates`, `Show YOLO Crop`).
     - Right column: Dog Gate status pill (`✅ Dog Detected (0.94)` or `🚫 Not a Dog`), generated **Meme Image** (downloadable), Emotion Probability Bar Chart, and Body-Language Explanation card.
   - **Tab 2 — `🔍 Text Prompt → Dog Meme`:**
     - Top bar: Text prompt input + `Search & Generate Meme` pill button + example prompt chips.
     - Results: Top-4 retrieved dog thumbnails from `data/clean/` + selected **Generated Meme** + predicted emotion & similarity score.
4. **Footer / Model Info Strip (inspired by `layout_reference/Capture d'écran 2026-11-22 230328.png`):**
   - Split yellow/orange summary banner showing pipeline architecture (`Part 1: YOLOv8s` · `Part 2: ViT-B/16` · `Part 3: CLIP + Captioning`).

---

## 5. Proposed File Structure for `Multimodal` Branch

```text
mlproject/
├── app.py                              # Gradio web application (Flow A + Flow B)
├── requirements.txt                    # Consolidated dependencies (torch, open_clip_torch, ultralytics, gradio, etc.)
├── part3_multimodal/
│   ├── README.md                       # Part 3 experiments, prompts & final comparison table
│   ├── src/
│   │   ├── dog_gate.py                 # Reusable YOLOv8s detector + cropper (from src/02_yolo_filter.py)
│   │   ├── clip_zeroshot.py            # Zero-shot & prompt-engineering evaluation on splits.csv
│   │   ├── clip_train.py               # Linear probe & fine-tuning on data/clean/ + class_weights.csv
│   │   ├── build_index.py              # Pre-computes CLIP embeddings for data/clean/
│   │   ├── retrieval.py                # Fast text-to-image cosine search
│   │   ├── captioner.py                # VLM (BLIP-2 / Qwen-VL) + template fallback
│   │   └── meme_maker.py               # Pillow meme text renderer (top/bottom text + outline)
│   ├── artifacts/                      # Saved CLIP embeddings, index CSV, checkpoints (gitignored if large)
│   └── results/                        # Part 3 metrics JSON/CSV & 10-20 curated caption/meme examples
```

---

## 6. Recommended Execution Order for Claude Code

1. **Setup & Dependencies:** Create `requirements.txt` and `part3_multimodal/` folder structure.
2. **CLIP Zero-Shot + Prompt Experiments (`clip_zeroshot.py`):** Run on 200 images first, then evaluate simple vs. body-language vs. ensembled prompts on `val` (`1,986` images) and log final `test` Macro F1.
3. **CLIP Linear Probe & Fine-Tuning (`clip_train.py`):** Train on `data/splits.csv` (`train`), tune on `val`, and compare `test` Macro F1 against Part 2 (`0.414` frozen / `0.567` ViT).
4. **Build Retrieval Index (`build_index.py` + `retrieval.py`):** Embed `data/clean/` once so text-to-image search is instant.
5. **Modularize Pipeline (`dog_gate.py`, `captioner.py`, `meme_maker.py`):** Wrap Part 1 YOLO + emotion classifier + VLM/template meme renderer into clean Python functions.
6. **Build Gradio App (`app.py`):** Implement the two-tab UI styled with the `layout_reference/` palette (`#F2E8D5`, `#0B2B3C`, `#ED5800`, `#148F9E`, `#F4B400`) and test end-to-end with dog photos, cat photos (dog gate test), and text prompts.
