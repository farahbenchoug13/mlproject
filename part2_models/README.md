# 🐶 DogTranslator — Part 2: Models, evaluation and data optimisation

**Author of this part:** Jamila ([@jamila-coder](https://github.com/jamila-coder)) · **Team project:** DogTranslator (machine learning course, multimodal computer vision)
**Builds on:** Part 1, data cleaning by Farah (`data/clean/`, `data/splits.csv` in this repo) · **Followed by:** Part 3, multimodal CLIP and captions

> **Goal:** recognise a dog's emotion from a photo, as one of 😠 **angry**, 😄 **happy**, 😌 **relaxed** or 😢 **sad**. Then explain *why* the model decided (Grad-CAM) and turn the prediction into a fun subtitle.
> **Why it matters:** many dog bites come from misreading body language. A tool that shows the warning signs can teach people to read them.

---

## 📑 Contents
1. [Summary of results](#1-summary-of-results)
2. [Data](#2-data)
3. [Method: 10 models, from a lazy guesser to a Vision Transformer](#3-method-10-models)
4. [Evaluation protocol](#4-evaluation-protocol)
5. [Results on raw data (first run)](#5-results-on-raw-data-first-run)
6. [Results on clean data, BEFORE optimisation](#6-results-on-clean-data-before-optimisation)
7. [Data optimisation strategies](#7-data-optimisation-strategies)
8. [Results AFTER optimisation](#8-results-after-optimisation)
9. [Explainability and diagnostics](#9-explainability-and-diagnostics)
10. [Tests on new photos (memes, cats)](#10-tests-on-new-photos)
11. [Limitations and future work](#11-limitations-and-future-work)
12. [Repository structure and how to reproduce](#12-repository-structure-and-how-to-reproduce)

---

## 1. Summary of results

| Stage | Best model | Test macro F1 | Test accuracy |
|---|---|---|---|
| Raw Kaggle data (first run) | ViT-B/16 fine-tuned | **0.593** | 0.610 |
| Clean data, BEFORE optimisation | ViT-B/16 fine-tuned | **0.567** | 0.590 |
| Clean data, AFTER optimisation | *(not run yet)* | |  |

**Key findings:**
1. **Three tiers of performance:** classic ML (0.27–0.31 macro F1) → frozen pretrained features (0.41) → fine-tuned networks (0.50–0.57). **Transfer learning brings most of the gain**; the choice of architecture matters less (ViT beats ResNet-50 by about 1 point, with 3.6× more parameters).
2. **Angry is the hardest emotion for every model.** It's the smallest class, and it's confused with *happy* (both have an open mouth). *Relaxed* and *sad* are confused with each other (both calm, low posture).
3. **Cleaning the data did not raise the test score** (0.593 → 0.567). We read this as a **more honest evaluation**: near-duplicates no longer leak between train and test, and background shortcuts are removed by the dog crops.
4. **Label noise is real:** the error analysis and the label checker found a panda, seals, cats, a cartoon and many doubtful labels.
5. **Pretrained features encode breed, not emotion** (t-SNE + clustering), which explains why fine-tuning is needed and points to a possible breed bias.
6. The models are **closed-set**: shown a cat, they still answer an emotion. A **dog detector** must run first (YOLO, Part 1).

---

## 2. Data

### 2.1 Raw dataset (Kaggle, *Dog Emotions Prediction* by devzohaib)
| | angry | happy | relaxed | sad | **total** |
|---|---|---|---|---|---|
| Images | 2,256 | 4,784 | 4,349 | 4,532 | **15,921** |
| Share | 14% | 30% | 27% | 28% | |

- All images are **384×384, RGB, JPEG**: the dataset author had already standardized them.
- **Imbalance:** angry has about half as many images as the other classes.
- **79 exact duplicates** (MD5 hash), including **47 photos with two different labels**. Example: the same puppy labelled both *relaxed* and *sad*.
- The dataset also contains **cats, cartoons, memes, several dogs per photo** and stock-photo **watermarks**.

### 2.2 Clean dataset (Part 1, Farah)
YOLOv8 kept photos with one clear dog and **cropped** it. Cats (543) and dog-less photos (910) were rejected, uncertain photos (1,091: several dogs, tiny dogs, low confidence) were set aside, and a perceptual hash removed **55 near-duplicates** and **84 conflicting-label** photos.

| | train | val | test | **total** |
|---|---|---|---|---|
| Images | 9,266 | 1,986 | 1,986 | **13,238** |

- Stratified 70/15/15 split, seed 42 (`data/splits.csv`). This is the **official exam for the whole team**.
- Angry lost the most images (only ~76% kept: snarling close-ups are hard for YOLO), so it's now ~13% of the data.
- Class weights used in training (computed on train): **angry 1.934 · happy 0.799 · relaxed 0.927 · sad 0.868**. They're identical to Part 1's `outputs/class_weights.csv`.
- After cleaning, crops have **varied sizes**; every model resizes them to its input size (128 px or 224 px).

---

## 3. Method: 10 models

We go **from simple to complex**, so each step shows what it adds.

| # | Model | Family | Input it sees | Idea |
|---|---|---|---|---|
| 1 | **Dummy** (most frequent class) | baseline | nothing | The "lazy guesser": the floor any model must beat |
| 2 | **k-NN** (k=5) | classic ML | raw pixels 32×32 | Vote of the 5 most similar photos |
| 3 | **Logistic regression** | classic ML | raw pixels 32×32 | One weight per pixel per emotion |
| 4 | **Random forest** (300 trees) | classic ML | color histogram (64 bins) | Many small decision trees on colors |
| 5 | **Linear SVM** | classic ML | **HOG** shape features | Best boundary in "shape space" |
| 6 | **TinyCNN** (ours, ~390k params) | deep, from scratch | 128×128 image | 4 blocks conv → BN → ReLU → max-pool |
| 7 | **ResNet-50 frozen + LogReg** | transfer (frozen) | 2,048 ImageNet features | Pretrained network as a fixed feature machine |
| 8 | **EfficientNet-B0** fine-tuned (4.0M) | transfer | 224×224 | Small CNN scaled efficiently |
| 9 | **ResNet-50** fine-tuned (23.5M) | transfer | 224×224 | Deep CNN with residual connections |
| 10 | **ViT-B/16** fine-tuned (85.8M) | transfer | 224×224, 196 patches | Self-attention: every patch looks at every other patch |

**Training settings (deep models):** AdamW (weight decay 1e-4), OneCycle learning-rate schedule, cross-entropy with **class weights** and **label smoothing 0.1**, mixed precision on GPU.

| Model | Epochs | Learning rate | Batch | Image size |
|---|---|---|---|---|
| TinyCNN | 12 | 3e-3 | 64 | 128 |
| ResNet-50 | 6 | 5e-4 | 64 | 224 |
| EfficientNet-B0 | 6 | 1e-3 | 64 | 224 |
| ViT-B/16 | 4 | 5e-5 | 32 | 224 |

The **checkpoint with the best validation macro F1** is kept. Training augmentation ("basic"): random resized crop (70–100%), horizontal flip, rotation ±10°, color jitter.

**Helper techniques:** color histograms, HOG, Sobel/Canny edges and histogram equalization (illustrations), PCA ("eigendogs"), t-SNE, k-means on t-SNE maps, Grad-CAM.

---

## 4. Evaluation protocol
- **Train** (fit) / **validation** (choose epochs, and **decide** which data optimisations to keep) / **test** (report only, never used for decisions).
- **Main metric: macro F1** (the F1 of each emotion, averaged), so a model can't hide a bad *angry* score. We also report **accuracy**, **F1 per emotion**, row-normalized **confusion matrices**, **training time** and **parameters**.
- Diagnostics: learning curves, t-SNE, Grad-CAM, most confident mistakes, calibration curve, tests on new photos and agreement between models.
- **One run per model with seed 42**: differences under ~0.01–0.02 macro F1 may be noise.

---

## 5. Results on raw data (first run)

| Model | Accuracy | **Macro F1** | F1 angry | F1 happy | F1 relaxed | F1 sad |
|---|---|---|---|---|---|---|
| ViT fine-tuned | 0.610 | **0.593** | 0.48 | 0.67 | 0.59 | 0.63 |
| ResNet-50 fine-tuned | 0.594 | 0.578 | 0.46 | 0.65 | 0.58 | 0.63 |
| EfficientNet-B0 fine-tuned | 0.538 | 0.519 | 0.39 | 0.60 | 0.50 | 0.59 |
| ResNet-50 frozen + LogReg | 0.475 | 0.461 | 0.34 | 0.54 | 0.46 | 0.50 |
| TinyCNN (from scratch) | 0.412 | 0.385 | — | — | — | — |
| Random forest on colors | 0.389 | 0.322 | 0.08 | 0.48 | 0.29 | 0.43 |
| Linear SVM on HOG | 0.335 | 0.311 | 0.14 | 0.40 | 0.34 | 0.36 |
| LogReg on pixels | 0.279 | 0.273 | 0.21 | 0.31 | 0.27 | 0.30 |
| k-NN on pixels | 0.294 | 0.262 | 0.17 | 0.40 | 0.24 | 0.25 |
| Dummy (lazy guesser) | 0.301 | 0.116 | 0.00 | 0.46 | 0.00 | 0.00 |

- **Accuracy can lie:** the lazy guesser gets 30% accuracy by always answering *happy*, but a macro F1 of only 0.116.
- **TinyCNN underfits** (curves still rising at epoch 12). **ResNet and EfficientNet overfit** after 2–3 epochs: EfficientNet reached 94% training accuracy for ~52% validation (its learning rate was likely too high). **ViT learns fastest**: 0.52 validation F1 after one epoch.

---

## 6. Results on clean data, BEFORE optimisation

| Model | Accuracy | **Macro F1** | F1 angry | F1 happy | F1 relaxed | F1 sad | Train time (s) |
|---|---|---|---|---|---|---|---|
| ViT fine-tuned | 0.590 | **0.567** | 0.45 | 0.70 | 0.48 | 0.63 | 419 |
| ResNet-50 fine-tuned | 0.579 | 0.558 | 0.42 | 0.66 | 0.53 | 0.62 | 466 |
| EfficientNet-B0 fine-tuned | 0.538 | 0.503 | 0.31 | 0.64 | 0.49 | 0.57 | 451 |
| ResNet-50 frozen + LogReg | 0.440 | 0.414 | 0.23 | 0.52 | 0.42 | 0.50 | 27 (+ features) |
| TinyCNN (from scratch) | 0.396 | 0.377 | 0.20 | 0.48 | 0.42 | 0.41 | 441 |
| Random forest on colors | 0.388 | 0.307 | 0.04 | 0.50 | 0.27 | 0.42 | 10 |
| Linear SVM on HOG | 0.329 | 0.304 | 0.15 | 0.41 | 0.30 | 0.35 | 62 |
| LogReg on pixels | 0.289 | 0.277 | 0.17 | 0.35 | 0.28 | 0.31 | 63 |
| k-NN on pixels | 0.314 | 0.275 | 0.11 | 0.40 | 0.27 | 0.32 | 1 |
| Dummy (lazy guesser) | 0.313 | 0.119 | 0.00 | 0.48 | 0.00 | 0.00 | 0 |

**Raw vs clean:** most scores dropped slightly (ViT −0.026, ResNet −0.020, frozen ResNet −0.047). Our interpretation:
1. **Raw scores were optimistic:** near-duplicates (found by Part 1's perceptual hash) could sit in both train and test.
2. **The crops removed background shortcuts.** On raw photos, models could use the scene (sofa = relaxed); Grad-CAM had shown attention on grass and toys. The **frozen ImageNet features**, which rely most on scene context, dropped the most.
3. **Less training data**, especially angry (−24%).
4. **Different test sets**, so small differences are within noise. **The ranking of models did not change.**

**Confusions (ViT, before):** angry recall 58% (it was 50% on raw data); happy 68%; **relaxed only 39%, with 29% predicted as *sad*** (without the background, calm and sad look alike); sad 68%.

**Learning curves:** all three fine-tuned networks **overfit after 2–3 epochs** (training accuracy 75–94% while validation stays at 0.52–0.58), and TinyCNN still underfits. This motivated the data optimisation.

---

## 7. Data optimisation strategies

Each strategy targets a problem **we observed**:

| Problem observed | Strategy | How |
|---|---|---|
| Confident "mistakes" that are really wrong labels | **B1. Label-noise filtering** ("confident learning") | Cross-validated classifier on frozen ResNet-50 features; a training photo is *suspicious* if it gets < 10% probability for its own label while another emotion gets > 60%. Suspicious photos are removed from **training only** |
| Angry is rare and hardest | **B2. Balanced sampling** | Sampler draws each emotion ~25% of the time (instead of class weights alone) |
| Overfitting after 2–3 epochs | **B3. Strong augmentation** | TrivialAugment (random rotations, shear, contrast, posterize…) + random erasing (hidden patches) |
| Predictions sensitive to small changes | **B4. Test-time augmentation** | Average of the prediction on the photo and its mirror image |

**Decision rule (ablation, B5):** each change is tested **one at a time** on ResNet-50, and **kept only if it raises the validation macro F1 by ≥ 0.005** (≥ 0.002 for TTA). The test set plays no role in this choice.

### 7.1 A correction we made: the first label checker was overconfident
The first version (logistic regression, C = 0.5) produced probabilities piled up at 0 and 1: about 3,800 training photos near 0%. The 8% safety cap then removed **741 photos, 301 of them angry** (about a quarter of the angry training set). We made it **more conservative** (C = 0.005) and capped removals at **5% of each emotion**. Both versions found real errors: a **panda, seals, cats, a cartoon**, a man with a dog. These went back to Part 1 as feedback.



### 7.2 How the AFTER models were trained
*(The optimisation stage has not been run yet.)*

### 7.3 Ablation results (decided on validation)
*(not run yet)*

---

## 8. Results AFTER optimisation
*(The AFTER models have not been trained yet: run the optimisation notebook, then push again.)*

---

## 9. Explainability and diagnostics
- **t-SNE:** raw pixels give a uniform "soup". ResNet-50 features form clear **clusters**, but each mixes all 4 emotions. **k-means on the map shows clusters of breeds and looks** (shaggy small dogs; flat-faced pugs and bulldogs; dark short-haired retrievers). ImageNet features encode *what* the dog is, not its emotion. This explains the limited frozen model and the gain from fine-tuning, and it points to a possible **breed bias** (some breeds look "sad" or "angry" by default).
- **Grad-CAM (ResNet-50):** correct angry and happy predictions focus on the **mouth** (teeth, tongue). That explains the angry↔happy confusion. Warning signs: some predictions rely on the **background** (grass for relaxed) or nearby objects (a plush toy), and the **eyes** are rarely used.
- **Most confident mistakes** are often **label errors**: a stone bulldog statue labelled sad, a dog in a vet cone labelled angry, lying dogs labelled sad, a fennec fox labelled angry. Lists: `results/**/suspicious_*_labels.csv`.
- **Calibration:** ViT's confidence closely matches its accuracy, with slight overconfidence above 80% (at ~85% confidence it is right ~74% of the time).

---

## 10. Tests on new photos
- **Internet photos (ViT):** a natural photo of a smiling corgi → *happy* 69%. **Meme photos** (bread on the face, sunglasses, clothing) → confidence drops to **40–50%**, spread over several emotions: the model "knows it doesn't know".
- **Same photo, different models:** when even the strong models disagree, the photo is genuinely ambiguous.
- **Cats (closed-set limitation):** shown a cat, ViT said *relaxed* (50%) and ResNet *angry* (59%), because "not a dog" is not an answer. Low confidence and disagreement are warning signs, but the reliable fix is a **dog gate** before the emotion model (YOLO from Part 1, or an ImageNet classifier: classes 151–268 are dog breeds).


---

## 11. Limitations and future work
- **One seed per model:** run 2–3 seeds and report mean ± standard deviation.
- **Label noise in the test set:** we never clean the test set, to keep the exam honest, so scores are measured against some wrong labels.
- **Limited tuning** (e.g. EfficientNet's learning rate, number of epochs).
- **Closed set:** a dog gate is required in the app.
- **Possible breed bias;** emotions are subjective (annotators disagree).
- **Compute:** part of the optimisation used **last-layer retraining on CPU** when no GPU was available. Its gains are a lower bound compared with full fine-tuning.
- **Next:** multimodal part (Part 3, CLIP with body-language prompts, captions generated from the image), attention maps for ViT, face/body crops, more angry examples.

---

## 12. Repository structure and how to reproduce

```
part2_models/
├── README.md              ← this file
├── notebooks/             ← all Colab notebooks, from the first draft to the final versions
└── results/               ← copied from Google Drive
    ├── results.csv, figures/, tables/        ← first run on raw data
    └── final_clean/                          ← clean data, before vs after optimisation
        ├── all_results.csv                   ← every model, every stage (test + validation scores)
        ├── figures/   tables/   histories/   probs/
        ├── optimised_recipe.json  summary.json  FINAL_model_card.json  split_used.csv
```

**Order to run (Google Colab):**
1. `DogTranslator_Part2_Models` (raw data, first run): GPU, about 1h15.
2. `DogTranslator_Part2_Final_Clean_Before_After`: clones this repo (Part 1 data), Acts 0 and A (all 10 models BEFORE), and B/C with a GPU. **Resumable:** rerun after a disconnect and finished models are reloaded from Drive.
3. No GPU? `DogTranslator_Part2_CPU_Optimisation` (Acts B and C with last-layer retraining).
4. `DogTranslator_Part2_Final_RESUME_from_ActD`: before vs after, Grad-CAM, errors, calibration, demo (no training).
5. Extras: the cats experiment, and comparing raw / clean / optimised models on your own photos.

**Trained models** (too big for GitHub, ViT ≈ 350 MB) are in Google Drive:
- https://drive.google.com/drive/folders/13wA3HNm_aQa7qRmpc9PZXdNZSaDUXCGk
- https://drive.google.com/drive/folders/1c3NxpGBhw1Do1J4mY8SGCk4TPrZzIg_F

5 model files, e.g. `EfficientNet-B0 fine-tuned.pt`, `ResNet50 fine-tuned.pt`, `TinyCNN (from scratch).pt`, `ViT fine-tuned.pt`, `final_clean/models/tinycnn_from_scratch_before.pt`

**Key figures:**
- [Confusion matrices](results/final_clean/figures/022_classic_ml_who_gets_confused_with_whom.png)

---
*Part 1 (data): Farah ([@farahbenchoug13](https://github.com/farahbenchoug13)) · Part 2 (models and data optimisation): Jamila ([@jamila-coder](https://github.com/jamila-coder)) · Part 3 (multimodal): team member 3*
