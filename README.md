# 🐶 DogTranslator — Multimodal Dog Emotion Recognition

> **Goal:** Recognise a dog's emotion from a photo across four classes — **angry**, **happy**, **relaxed**, and **sad** — explain *why* a model made its prediction (Grad-CAM, error & calibration analysis), and extend the pipeline with multimodal vision-language methods (CLIP & captions).
>
> **Motivation:** Many dog bites result from humans misreading canine body language. A tool that detects emotional cues and highlights the visual evidence can help people better understand dog behaviour.

---

## 👥 Project Structure & Team Workflow

The project is organized into three parts:

| Part | Scope | Owner | Status in Repo |
|---|---|---|---|
| **Part 1** | Raw data inspection, YOLOv8 dog detection & cropping, perceptual deduplication, stratified splitting, class weights | Farah ([@farahbenchoug13](https://github.com/farahbenchoug13)) | Complete (`src/`, `data/`, `outputs/`) |
| **Part 2** | 10-model benchmark (classic ML → CNNs → ViT), raw vs. clean evaluation, data optimisation strategies (B1–B5), Grad-CAM & diagnostics | Jamila ([@jamila-coder](https://github.com/jamila-coder)) | Complete for raw & clean BEFORE optimisation; optimisation notebooks included (`part2_models/`) |
| **Part 3** | CLIP experiments (zero-shot → fine-tuning), full pipeline (dog gate → emotion → meme), Gradio app, evaluations | Amine ([@kacemamine721-star](https://github.com/kacemamine721-star)) | Complete (`part3_multimodal/`, `app.py`) |

---

## 📁 Repository Layout

```text
mlproject/
├── README.md                          # Full project documentation (this file)
├── models/yolov8s.pt                  # Pretrained YOLOv8s weights (Part 1 filtering + Part 3 dog gate)
│
├── src/                               # Part 1: Data cleaning & preparation pipeline
│   ├── 01_inspect.py                  # Raw dataset audit & contact-sheet generation
│   ├── 02_yolo_filter.py              # YOLOv8 dog/cat filtering & bounding-box cropping
│   ├── 03_dedupe.py                   # Perceptual hash (pHash) deduplication & conflict removal
│   ├── 04_split.py                    # Stratified 70/15/15 train/val/test split
│   ├── 05_stats.py                    # Dataset statistics table, class weights & bar chart
│   └── view_reasons.py                # Contact sheets for YOLO "to-check" images by reason
│
├── data/                              # Dataset stages & official team split
│   ├── raw/                           # Original Kaggle images (15,921 total)
│   ├── clean/                         # Final cleaned dog crops (13,238 images across 4 classes)
│   ├── rejected/                      # Photos with no dog or only cats (1,453 images)
│   ├── to_check/                      # Uncertain detections set aside for review (1,091 images)
│   ├── duplicates/                    # Near-duplicate same-label images removed (55 images)
│   ├── conflicts/                     # Near-duplicate images with conflicting labels (84 images)
│   └── splits.csv                     # Official 70/15/15 stratified split (seed=42)
│
├── outputs/                           # Part 1 logs, tables, and diagnostic figures
│   ├── yolo_log.csv                   # Per-image YOLOv8 detection decisions & metadata
│   ├── dedupe_pairs.csv               # Near-duplicate & conflict pairs with Hamming distance
│   ├── stats_table.csv                # Funnel counts from raw → clean_final → splits
│   ├── class_weights.csv              # Inverse-frequency class weights computed on train split
│   ├── class_counts.png               # Raw vs. clean class distribution bar chart
│   ├── inspect_<class>.png            # 50-image random contact sheets per raw class
│   ├── dedupe_pairs.png               # Visual sample of detected duplicate/conflict pairs
│   └── reason_<reason>.png            # Visual contact sheets for to-check categories
│
├── part2_models/                      # Part 2: Modeling, evaluation & data optimisation
│   ├── README.md                      # Detailed Part 2 report & reproduction instructions
│   ├── notebooks/
│   │   ├── DogTranslator_Part2_Final_Clean_Before_After.ipynb
│   │   └── DogTranslator_Part2_CPU_Optimisation.ipynb
│   └── results/
│       ├── results.csv                # First-run benchmark on raw Kaggle data
│       ├── summary.json               # Raw-run metadata & model metrics
│       ├── history_<model>.csv        # Training histories (EfficientNet-B0, ResNet50, ViT)
│       ├── suspicious_labels.csv      # High-confidence model mistakes / suspected label errors
│       └── final_clean/               # Results, figures & probabilities on clean data
│           ├── all_results.csv        # Classic ML baseline results on clean data
│           ├── figures/               # 25 diagnostic figures (EDA, PCA/eigendogs, confusions)
│           ├── probs/                 # Saved test-set class probabilities
│           └── tables/                # Before-vs-after comparison tables
│
├── part3_multimodal/                  # Part 3: CLIP, pipeline, evaluations (see part3_multimodal/README.md)
│   ├── src/                           # dog gate, emotion, captions, memes, search, evaluation scripts
│   ├── notebooks/                     # Kaggle CLIP fine-tuning notebook
│   ├── assets/examples/               # preset photos used by the app
│   └── results/                       # metrics, significance tests, audits, validation report
├── app.py                             # Gradio app (photo / description / both → meme)
├── Dockerfile  docker-compose.yml     # one-command run: docker compose up --build → http://localhost:7860
├── requirements.txt  .env.example     # dependencies; optional free API keys for AI captions
└── docs/HANDOFF_MULTIMODAL.md         # original Part 3 specification
```

---

## 🧹 Part 1: Data Cleaning Pipeline (`src/`)

The raw dataset comes from Kaggle (*Dog Emotions Prediction* by `devzohaib`) and contains **15,921** standardized `384×384` RGB JPEG images across four folders: `angry` (2,256), `happy` (4,784), `relaxed` (4,349), and `sad` (4,532). Inspection revealed class imbalance (`angry` is ~14%), exact/near duplicates, conflicting labels, cats, cartoons, memes, watermarks, and multi-dog photos.

### Pipeline Stages (Run in Order)

1. **`src/01_inspect.py` — Raw Dataset Audit**
   - Scans `data/raw/` for `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.
   - Checks unreadable files and images smaller than `100px`, logs min/max dimensions, and generates 50-image contact sheets (`outputs/inspect_<class>.png`).

2. **`src/02_yolo_filter.py` — YOLOv8 Filtering & Cropping**
   - Runs `models/yolov8s.pt` (`imgsz=640`, `conf=0.25`) detecting COCO classes `PERSON (0)`, `CAT (15)`, and `DOG (16)`.
   - Decision rules per image:
     - **Rejected (`data/rejected/`)**: No dog detected (`no_dog`), separated into cats vs. other dog-less photos (`1,453` total).
     - **To-check (`data/to_check/`)**: `1,091` ambiguous images flagged for one of four reasons:
       - `cat_more_confident`: A cat is detected with confidence ≥ 0.50 and higher than the best dog.
       - `low_conf`: No dog detection reaches `CONF_DOG = 0.30`.
       - `small_dog`: Single dog box covers `< 8%` (`MIN_AREA = 0.08`) of image area.
       - `multi_dog`: Multiple dogs where the largest box is not at least `2×` the area of the second largest.
     - **Accepted (`data/clean/`)**: `13,377` images with one clear dog (or one dominant dog ≥ `2×` larger). If dog confidence ≥ `0.50` (`CONF_CROP`), the image is **cropped** to the dog bounding box with `10%` padding (`PAD = 0.10`); otherwise the full image is kept.
   - Logs every image decision to `outputs/yolo_log.csv`.

3. **`src/view_reasons.py` — Visual Audit of Uncertain Images**
   - Reads `outputs/yolo_log.csv` and renders contact sheets (`outputs/reason_<reason>.png`) for each `to-check` category (`cat_more_confident`, `low_conf`, `small_dog`, `multi_dog`).

4. **`src/03_dedupe.py` — Perceptual Hash Deduplication**
   - Computes 64-bit perceptual hashes (`imagehash.phash`) for all `13,377` images in `data/clean/` and computes pairwise Hamming distances.
   - Pairs with Hamming distance `≤ 5` (`THR = 5`) are flagged:
     - **Same label (`55` images):** One copy is kept; duplicates are moved to `data/duplicates/`.
     - **Different labels (`84` images):** Both images are removed to `data/conflicts/` to prevent label ambiguity.
   - Outputs `outputs/dedupe_pairs.csv` and `outputs/dedupe_pairs.png`.

5. **`src/04_split.py` — Official Stratified Split**
   - Splits the **13,238** remaining clean images into **70% train (9,266) / 15% val (1,986) / 15% test (1,986)** using `train_test_split` with `stratify=label` and `random_state=42`.
   - Verifies path uniqueness and file existence, then writes `data/splits.csv`.

6. **`src/05_stats.py` — Summary Statistics & Class Weights**
   - Aggregates counts across pipeline stages into `outputs/stats_table.csv` and `outputs/class_counts.png`.
   - Computes balanced class weights on the **train split only** (`n_train / (n_classes * n_class_train)`) and saves them to `outputs/class_weights.csv`:

| Emotion | Raw Count | After YOLO | Duplicates | Conflicts | Clean Final | Train | Val | Test | Train Weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **angry** | 2,256 | 1,739 | 19 | 9 | **1,711** | 1,198 | 256 | 257 | **1.934** |
| **happy** | 4,784 | 4,176 | 4 | 28 | **4,144** | 2,901 | 622 | 621 | **0.799** |
| **relaxed** | 4,349 | 3,599 | 12 | 16 | **3,571** | 2,499 | 536 | 536 | **0.927** |
| **sad** | 4,532 | 3,863 | 20 | 31 | **3,812** | 2,668 | 572 | 572 | **0.868** |
| **TOTAL** | **15,921** | **13,377** | **55** | **84** | **13,238** | **9,266** | **1,986** | **1,986** | — |

*(83.1% of raw images retained; `angry` lost the highest share because snarling close-ups are harder for YOLO to detect.)*

---

## 🧠 Part 2: Models, Evaluation & Data Optimisation (`part2_models/`)

Part 2 benchmarks **10 models** of increasing complexity using **Macro F1** as the primary metric (so minority-class performance on `angry` cannot be masked by overall accuracy). All models use `seed=42`, and deep models are trained with AdamW (`weight_decay=1e-4`), OneCycleLR, class-weighted cross-entropy, and label smoothing (`0.1`).

### Model Progression & Benchmark Results

| # | Model | Family | Input / Representation | Raw Test Macro F1 | Clean (Before) Macro F1 | Clean Test Accuracy |
|---:|---|---|---|---:|---:|---:|
| 1 | **Dummy** (most frequent) | Baseline | None | 0.116 | 0.119 | 0.313 |
| 2 | **k-NN** (`k=5`) | Classic ML | 32×32 raw pixels | 0.262 | 0.275 | 0.314 |
| 3 | **Logistic Regression** | Classic ML | 32×32 raw pixels | 0.273 | 0.277 | 0.289 |
| 4 | **Random Forest** (300 trees) | Classic ML | 64-bin color histogram | 0.322 | 0.307 | 0.388 |
| 5 | **Linear SVM** | Classic ML | HOG shape features | 0.311 | 0.304 | 0.329 |
| 6 | **TinyCNN** (~390k params) | Deep (from scratch) | 128×128 RGB | 0.385 | 0.377 | 0.396 |
| 7 | **ResNet-50 frozen + LogReg** | Transfer (frozen) | 2,048 ImageNet features | 0.461 | 0.414 | 0.440 |
| 8 | **EfficientNet-B0** (4.0M) | Transfer (fine-tuned) | 224×224 RGB | 0.519 | 0.503 | 0.538 |
| 9 | **ResNet-50** (23.5M) | Transfer (fine-tuned) | 224×224 RGB | 0.578 | 0.558 | 0.579 |
| 10 | **ViT-B/16** (85.8M) | Transfer (fine-tuned) | 224×224 (196 patches) | **0.593** | **0.567** | **0.590** |

### Key Findings & Diagnostics

1. **Three Performance Tiers:** Classic ML (`0.27–0.31` F1) → frozen ImageNet features (`0.41` F1) → fine-tuned deep networks (`0.50–0.57` F1). Transfer learning drives most of the gain.
2. **Why Clean Scores Are Slightly Lower Than Raw (`0.593 → 0.567`):** Removing near-duplicates prevents train/test data leakage, and YOLO dog crops eliminate background shortcuts (e.g., sofa = `relaxed`, grass/toys). The clean benchmark is therefore a more honest evaluation.
3. **Hardest Confusions:** `angry` is the hardest class (`F1 = 0.45` on ViT clean), frequently confused with `happy` (both show open mouths/teeth). `relaxed` and `sad` are confused with each other due to calm, low-energy postures.
4. **t-SNE & Breed Bias:** Frozen ResNet-50 features cluster by **dog breed and appearance** (e.g., pugs/bulldogs, shaggy small dogs, dark retrievers) rather than emotion, explaining why fine-tuning is essential and highlighting potential breed bias.
5. **Grad-CAM & Error Analysis:** Correct `angry`/`happy` predictions focus on the mouth; some errors rely on background/toys, and many high-confidence "mistakes" are actually noisy ground-truth labels (e.g., a stone statue, a vet cone, a fennec fox, a panda, seals, cartoons).
6. **Closed-Set Limitation:** Emotion classifiers always pick one of the 4 emotions even on non-dogs (e.g., cats). Production inference requires a **dog gate** (Part 1 YOLOv8 or ImageNet dog-breed filter) before emotion classification.

### Data Optimisation Strategies (Designed for Ablation)

To address label noise, class imbalance, overfitting after 2–3 epochs, and prediction variance, four targeted interventions are defined in the Part 2 notebooks (`DogTranslator_Part2_Final_Clean_Before_After.ipynb` and `DogTranslator_Part2_CPU_Optimisation.ipynb`):
- **B1 — Confident Learning Label Filtering:** Cross-validated conservative classifier (`C = 0.005`) on frozen ResNet-50 features; removes training samples with `< 10%` probability on their assigned label and `> 60%` on another (capped at `5%` per class; test set untouched).
- **B2 — Balanced Sampling:** Draws each emotion ~25% of the time during training.
- **B3 — Strong Augmentation:** `TrivialAugment` + random erasing.
- **B4 — Test-Time Augmentation (TTA):** Averages predictions across original and horizontally flipped images.
- **B5 — Validation-Only Ablation:** Each strategy is kept only if it improves validation Macro F1 (`≥ 0.005`, or `≥ 0.002` for TTA).

*(For full Part 2 details, Colab execution order, and Google Drive links to `.pt` checkpoints, see [part2_models/README.md](part2_models/README.md).)*

---

## 🌐 Part 3: Multimodal Extension, Pipeline & App (`part3_multimodal/`)

Same contract as Part 2 (`data/splits.csv`, decisions on val, test reported once). Full report: [part3_multimodal/README.md](part3_multimodal/README.md).

| Result (clean test, 1,986 photos) | Macro F1 |
|---|---|
| Part 2's best — ViT-B/16 fine-tuned (clean rerun) | 0.576 |
| CLIP zero-shot, no training | 0.570 |
| **CLIP fine-tuned + WiSE-FT + recovered training dogs (final)** | **0.622** (+4.5, p = 3·10⁻⁶) |

- **Pipeline:** YOLOv8s dog gate (Part 1 rules) with a CLIP second opinion (99.7% of dogs accepted on Oxford-IIIT Pet vs 96.3%)
  → CLIP emotion model → caption by a free LLM (Tounsi / English / French, offline templates as backup) → meme.
  Text search over the 13,238 clean photos (P@10 = 0.75 vs 0.25 chance).
- **Feedback for Part 1:** YOLO misses close-up dog faces; 1,258 of the 1,706 "no dog" rejects are dogs (≈92% precision on a
  manual check). Adding them to the *training* set only (val/test untouched) gave +2.6 macro F1 (p = 0.005). Part 1's data is unchanged.

- **App validation:** automated end-to-end test 8/8 passed on the Docker app; AI captions go through a family-friendly
  filter (Tounsi/English/French swear words and insults blocked, tested with insult-baiting ideas); manual tests by the team passed.

**Run the app:** `docker compose up --build` → http://localhost:7860 (model files go in `part3_multimodal/artifacts/`, see the Part 3 README).

---

## ⚙️ Setup & Reproduction

### Prerequisites
- Python 3.9+
- Core packages: `ultralytics`, `pillow`, `imagehash`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`
- Deep learning (Part 2 / Part 3): `torch`, `torchvision`, `timm` (for ViT/EfficientNet), and GPU acceleration (Google Colab or local CUDA).

### Running Part 1 Locally
Run from the repository root (`mlproject/`):
```bash
python src/01_inspect.py
python src/02_yolo_filter.py
python src/view_reasons.py
python src/03_dedupe.py
python src/04_split.py
python src/05_stats.py
```

### Running Part 2 Notebooks
Open the notebooks in `part2_models/notebooks/` in Google Colab (with GPU for full fine-tuning, or CPU for `DogTranslator_Part2_CPU_Optimisation.ipynb`). They clone this repository to access `data/clean/` and `data/splits.csv`.
