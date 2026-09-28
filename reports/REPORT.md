# 📑 Training Report — Plant Disease Detection (PlantScan AI)

This report documents the methodology, training process, and results for the three models trained in this project, and gives a focused comparison between **EfficientNetB0** and **MobileNetV2** (trained on the same dataset) to determine which is the better choice for this application.

---

## 1. Objective

Build an image classifier that can identify plant diseases from a leaf photo, to power a web app that gives users an instant diagnosis and treatment recommendation. Three CNN backbones were trained via transfer learning to compare accuracy, size, and suitability for deployment.

---

## 2. Datasets

Two **different** datasets were used across the three notebooks:

| | EfficientNetB0 / MobileNetV2 | ResNet50 |
|---|---|---|
| Source | Kaggle: `abdallahalidev/plantvillage-dataset` (color split) | Kaggle: `emmarex/plantdisease` |
| Total images | 54,305 | 20,638 |
| Train / Val split | 43,444 / 10,861 (80/20, `seed=42`) | 16,511 / 4,127 (80/20, `seed=42`) |
| Classes | **38** (14 crop types incl. Apple, Grape, Corn, Tomato, etc.) | **15** (only Pepper-bell, Potato, Tomato) |
| Image size | 224×224 | 224×224 |

> Because ResNet50 was trained on a smaller, easier subset (fewer classes, fewer visually-similar diseases), **its accuracy is not directly comparable** to the other two models. It's reported separately in Section 5.

---

## 3. Common Methodology

All three notebooks use the same transfer-learning recipe:

1. **Data pipeline:** `tf.keras.utils.image_dataset_from_directory`, batched, prefetched with `AUTOTUNE`.
2. **Augmentation** (applied in-graph, active only during training): random flip, rotation, zoom, contrast, translation (EfficientNetB0 additionally adds `GaussianNoise`).
3. **Backbone:** ImageNet-pretrained, `include_top=False`, initially **frozen**.
4. **Head:** `GlobalAveragePooling2D → Dropout → Dense(num_classes, softmax)`.
5. **Phase 1 (frozen backbone):** train only the new head with `Adam` (default LR).
6. **Phase 2 (fine-tuning):** unfreeze the **last 30 layers** of the backbone, recompile with `Adam(lr=1e-5)`, continue training.
7. **Callbacks:** `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)`, `ReduceLROnPlateau`, `ModelCheckpoint(monitor='val_accuracy')`.
8. **Evaluation:** `classification_report` (precision/recall/F1 per class) + confusion matrix on the validation split.

---

## 4. Model-by-Model Results

### 4.1 EfficientNetB0

> ✅ **Update:** this notebook was re-run top-to-bottom (clean "Run All") after an earlier save had cut off the fine-tuning output at epoch 3/15. The numbers below are from the complete, fully-trained run and are internally consistent — `model.evaluate()` matches the final epoch exactly.

| | Phase 1 (frozen, 10 epochs) | Phase 2 (fine-tune, 15 epochs) |
|---|---|---|
| Epochs completed | 10 / 10 ✅ | **15 / 15 ✅** |
| Best val_accuracy | 94.60% (epoch 10) | **97.66% (epoch 15, final)** |
| Best val_loss | 0.1639 (epoch 10) | **0.0682 (epoch 15, final)** |

- **Total params:** 4,098,249 (15.63 MB) — **Trainable (Phase 1):** 48,678
- **`model.evaluate()` on val set:** Loss `0.0682`, Accuracy `97.66%` — matches the final fine-tuning epoch exactly ✅
- **`classification_report` overall accuracy:** **98%** (macro avg F1 0.97, weighted avg F1 0.98)
- Validation accuracy improved **every single epoch** of fine-tuning with no plateau — `EarlyStopping` never triggered, so the model trained for the full planned schedule.
- Weakest classes: `Tomato_Early_blight` (recall 0.79), `Tomato_Leaf_Mold` (recall 0.85), `Corn_Cercospora_leaf_spot` (recall 0.76), `Tomato_Target_Spot` (precision 0.82)
- Strongest classes: `Apple_Cedar_apple_rust`, `Corn_healthy`, `Corn_Common_rust`, `Grape_healthy`, `Orange_Haunglongbing`, `Peach_Bacterial_spot` (precision/recall ≈ 1.00)

### 4.2 MobileNetV2

| | Phase 1 (frozen, 10 epochs) | Phase 2 (fine-tune, 15 epochs planned) |
|---|---|---|
| Epochs completed | 10 / 10 | **15 / 15** ✅ |
| Best val_accuracy | 93.48% (epoch 9) | **95.40%** (epoch 15, final) |
| Best val_loss | 0.2006 (epoch 9) | **0.1427** (epoch 15, final) |

- **Total params:** 2,306,662 (8.80 MB) — **Trainable (Phase 1):** 48,678
- **`classification_report` overall accuracy:** **95%** (macro avg F1 0.94, weighted avg F1 0.95–0.96)
- Weakest classes: `Tomato_Early_blight` (recall 0.56), `Potato_healthy` (recall 0.68), `Corn_Cercospora_leaf_spot` (recall 0.74)
- Strongest classes: `Corn_healthy`, `Grape_Leaf_blight`, `Grape_healthy`, `Orange_Haunglongbing` (precision/recall ≈ 1.00)
- Training ran cleanly and consistently — both phases completed fully, and `classification_report` numbers line up with the final epoch's validation metrics.
- Notebook also includes bonus utility cells: single-image prediction with OpenCV, and a **live webcam demo** (Colab browser camera capture → real-time prediction loop).

### 4.3 ResNet50 *(different dataset — 15 classes, see caveat above)*

| | Phase 1 (frozen, 15 epochs planned) | Phase 2 (fine-tune, 10 epochs planned) |
|---|---|---|
| Epochs completed | 9 / 15 (early-stopped) | 10 / 10 ✅ |
| Best val_accuracy | 90.60% (epoch 6) | **95.93%** (epoch 10, final) |
| Best val_loss | 0.2607 (epoch 6) | **0.1201** (epoch 10, final) |

- **Total params:** 23,618,447 (90.10 MB) — **Trainable (Phase 1):** 30,735
- **`classification_report` overall accuracy:** **96%** (macro & weighted avg F1 0.96)
- Weakest class: `Tomato__Target_Spot` (recall 0.80)
- Training was clean and self-consistent (`EarlyStopping` triggered naturally in Phase 1 after 3 non-improving epochs; Phase 2 completed in full).

---

## 5. ⭐ MobileNetV2 vs EfficientNetB0 — Direct Comparison

These two were trained on the **same dataset, same classes, same augmentation, same callbacks** — so this is a fair, like-for-like comparison. Both notebooks now have complete, fully-trained runs (15/15 fine-tuning epochs each), so the numbers below are directly comparable.

| Metric | EfficientNetB0 | MobileNetV2 | Winner |
|---|---|---|---|
| Total parameters | 4,098,249 | **2,306,662** | 🏆 MobileNetV2 (44% smaller) |
| Model size | 15.63 MB | **8.80 MB** | 🏆 MobileNetV2 |
| Fine-tuning completed | 15 / 15 ✅ | 15 / 15 ✅ | Tie |
| Best val_accuracy (final epoch) | **97.66%** | 95.40% | 🏆 EfficientNetB0 (+2.26 pts) |
| `model.evaluate()` accuracy | **97.66%** | — | 🏆 EfficientNetB0 |
| `classification_report` accuracy | **98%** | 95% | 🏆 EfficientNetB0 |
| Weighted F1-score | **0.98** | 0.95–0.96 | 🏆 EfficientNetB0 |
| Macro F1 (rarer/harder classes) | **0.97** | 0.94 | 🏆 EfficientNetB0 |
| Weakest-class recall | `Tomato_Early_blight` 0.79 | `Tomato_Early_blight` 0.56 | 🏆 EfficientNetB0 (much more balanced) |
| Training speed (Phase 1, time/step) | ~58 ms/step | **~41–42 ms/step** | 🏆 MobileNetV2 (~30% faster) |

### Verdict: **EfficientNetB0 is the clearly better-performing model once both are fully trained.**

With both models trained to completion, EfficientNetB0 wins decisively on every accuracy metric — overall accuracy (98% vs 95%), weighted F1 (0.98 vs 0.95), and macro F1 (0.97 vs 0.94). The gap is especially visible on the harder, visually-similar classes: `Tomato_Early_blight` recall jumps from 0.56 (MobileNetV2) to 0.79 (EfficientNetB0), and several previously weak Tomato/Corn classes are noticeably stronger.

This **reverses the earlier conclusion** from when EfficientNetB0's fine-tuning notebook only had 3 of 15 epochs captured (94% accuracy at the time) — that was an artifact of an incomplete training run, not a true reflection of the architecture. With the corrected, fully-trained notebook, EfficientNetB0 (98%) outperforms MobileNetV2 (95%) by a clear margin, which is in line with what's generally expected from the architecture in published benchmarks (EfficientNet is designed to get more accuracy per parameter than MobileNetV2).

**The trade-off that remains:** MobileNetV2 is still ~44% smaller (8.8 MB vs 15.6 MB) and trains/infers noticeably faster per step. So the choice comes down to priorities:
- **Accuracy matters most** (correct diagnosis is the whole point of this app) → **EfficientNetB0** — 3 points higher accuracy and much better balanced across hard classes.
- **Extreme size/latency constraints** (e.g. on-device/offline mobile inference with no server) → MobileNetV2 — still a respectable 95%, at roughly half the size.

### Recommendation for the web app
✅ **Implemented:** `app.py` now loads **both** models and lets the user pick from a dropdown (EfficientNetB0 / MobileNetV2), defaulting to EfficientNetB0 since it's the better-performing one. This sidesteps the "which one do we deploy" question entirely — both are available, and the app's default matches what this report recommends. (ResNet50 is intentionally left out of the dropdown — see Section 6.)

---

## 6. Why not compare ResNet50 directly?

ResNet50 reached 96% accuracy, but it was trained on a **different and easier task** — only 15 classes covering 3 crop types (Pepper, Potato, Tomato), versus 38 classes across 14 crop types for the other two models. Fewer classes generally means less inter-class confusion and a higher ceiling on accuracy, so a direct numeric comparison would be misleading even if the numbers happened to be close. ResNet50 is also ~10× larger than MobileNetV2 and ~5.8× larger than EfficientNetB0 (23.6M params vs 2.3M / 4.1M), making it a much heavier model to deploy.

Worth noting: even on its easier 15-class subset, ResNet50's 96% doesn't beat EfficientNetB0's 98% on the harder 38-class task — which is a further point in EfficientNetB0's favor, not against it.

If a true 3-way comparison is needed, **ResNet50 should be retrained on the same 38-class color dataset** used for the other two models.

---

## 7. Summary Table

| Model | Dataset | Classes | Params | Size | Accuracy | Best for |
|---|---|---|---|---|---|---|
| **EfficientNetB0** | PlantVillage-color | 38 | 4.1M | 15.6 MB | **98%** | ✅ **Recommended — best accuracy, fully trained** |
| MobileNetV2 | PlantVillage-color | 38 | **2.3M** | **8.8 MB** | 95% | Best if size/speed is the top priority |
| ResNet50 | PlantVillage (subset) | 15 | 23.6M | 90.1 MB | 96% | Reference only — not comparable (easier task) |

---
