# Classification Pipeline: Improvement Recommendations

**Current system:** ResNet50 (frozen, ImageNet-pretrained) → PCA(128) → RBF-SVM
**Current metrics:** Accuracy 77.5% · Precision 72.73% · Recall 93.02% · F1 81.63%
**Benchmark:** 200 samples/class from HAM10000 binary melanoma task

This document compares the existing pipeline against cutting-edge approaches from the literature (2022–2025) and prescribes concrete, prioritised improvements.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Pipeline Audit](#2-current-pipeline-audit)
3. [SOTA Landscape](#3-sota-landscape)
4. [Improvement Areas](#4-improvement-areas)
   - 4.1 [Backbone: Switch to a Stronger Feature Extractor](#41-backbone-switch-to-a-stronger-feature-extractor)
   - 4.2 [End-to-End Fine-Tuning Instead of Frozen Extraction](#42-end-to-end-fine-tuning-instead-of-frozen-extraction)
   - 4.3 [Replace SVM with a Learned Classifier Head](#43-replace-svm-with-a-learned-classifier-head)
   - 4.4 [Use All Available Training Data (Remove Sampling Cap)](#44-use-all-available-training-data-remove-sampling-cap)
   - 4.5 [Address Class Imbalance Properly](#45-address-class-imbalance-properly)
   - 4.6 [Advanced Data Augmentation](#46-advanced-data-augmentation)
   - 4.7 [Hair Artifact Handling](#47-hair-artifact-handling)
   - 4.8 [Loss Function Improvements](#48-loss-function-improvements)
   - 4.9 [Multi-Scale Input Resolution](#49-multi-scale-input-resolution)
   - 4.10 [Segmentation-Aware Training](#410-segmentation-aware-training)
   - 4.11 [Ensemble Methods](#411-ensemble-methods)
   - 4.12 [Self-Supervised Pre-Training](#412-self-supervised-pre-training)
   - 4.13 [Clinical Metadata Integration (Multimodal)](#413-clinical-metadata-integration-multimodal)
   - 4.14 [Explainability / Clinical Interpretability](#414-explainability--clinical-interpretability)
   - 4.15 [Evaluation Protocol Improvements](#415-evaluation-protocol-improvements)
5. [Prioritised Roadmap](#5-prioritised-roadmap)
6. [Expected Performance Gains](#6-expected-performance-gains)
7. [References](#7-references)

---

## 1. Executive Summary

The current pipeline is a solid two-stage transfer-learning baseline, but it leaves substantial performance on the table due to four structural limitations:

| Limitation | Impact |
|---|---|
| Frozen ResNet50 backbone | Features not adapted to dermoscopy domain |
| 200-sample cap per class | Only ~4% of available melanoma images used |
| RBF-SVM as final classifier | Cannot learn hierarchical or attention-based representations |
| No task-specific augmentation | Model not robust to real clinical variation |

Addressing these four issues alone is expected to push binary melanoma F1 from **~82% to 90–95%**, competitive with the published literature for similar binary tasks on HAM10000.

---

## 2. Current Pipeline Audit

### Architecture

```
Input image (JPEG) + Segmentation mask (PNG)
        ↓
  Apply binary mask (background → black)
        ↓
  Resize to 224×224 (bilinear)
        ↓
  Median blur 3×3
        ↓
  Normalise: (x − μ_ImageNet) / σ_ImageNet
        ↓
  ResNet50 backbone (FROZEN, ImageNet weights) → 2048-d vector, L2-normed
        ↓
  StandardScaler (zero mean, unit variance)
        ↓
  PCA: 2048 → 128 dims
        ↓
  RBF-SVM (C-SVC, trainAuto 5-fold CV)
        ↓
  Binary prediction: mel / not-mel
```

### Strengths

- **Correct data leakage prevention** — scaler and PCA fitted on training split only.
- **Reproducible** — fixed seed (42) for all random operations.
- **High recall (93%)** — appropriate clinical priority; missed melanomas are worse than false alarms.
- **Lesion masking** — isolates the relevant region before feeding the backbone.
- **OpenCV-compatible export** — enables C++ deployment without a Python runtime.

### Weaknesses

| Area | Issue |
|---|---|
| Data volume | Hard cap of 200 images per class from 10,015 available |
| Backbone | ImageNet features, not fine-tuned to dermoscopy |
| Backbone size | ResNet50 (25M params); larger/newer architectures exist |
| Classifier | SVM cannot learn non-linear interactions in feature space the way a neural head can |
| Augmentation | None at training time; median blur only |
| Imbalance | Balanced by under-sampling; throws away majority-class information |
| Evaluation | 80 test samples (20% of 400) is too small for clinical benchmarking |
| Metrics | AUC-ROC not reported; standard in clinical literature |

---

## 3. SOTA Landscape

### Performance on HAM10000 (7-class unless noted)

| Method | Architecture | Accuracy | AUC | Notes |
|---|---|---|---|---|
| ResNet50 + SVM (ours) | ResNet50 (frozen) | 77.5%* | — | Binary, 400-sample subset |
| EfficientNetB4 | CNN | 87.9% | — | 7-class, full dataset |
| SkinNet (ensemble) | CNN stack | 86.7% | 0.96 | 7-class |
| Swin-T + Focal Loss | Transformer | 87.71% | — | 7-class, imbalance-aware |
| Multimodal (image + metadata) | EfficientNet + MLP | 94.11% | 0.9426 | 7-class, HAM10000 |
| DINOv2 fine-tuned | ViT-B/14 | ~88–92% | — | Competitive with Swin-L |
| Swin + ViT + EfficientNetB4 (ensemble) | Hybrid | 98.5%† | — | 7-class |
| EfficientNetV2S + Swin (ensemble) | Hybrid | 99.1%† | — | 7-class |

*Binary task on 400 images. Not directly comparable to 7-class full-dataset results.
†High reported accuracies (>99%) warrant scrutiny; may reflect specific split/oversampling strategies.

### Binary Melanoma Classification SOTA

Published papers on HAM10000 **binary** (mel vs. not-mel) with full dataset:
- InceptionResNetV2 fine-tuned: ~95–97% accuracy, AUC ~0.98
- EfficientNetB5 + focal loss: AUC ~0.97–0.98
- Ensemble (3–5 CNNs): AUC ~0.98–0.99

**The gap between our current ~82% F1 and SOTA ~97% AUC represents room for improvement, not a fundamental ceiling.**

---

## 4. Improvement Areas

### 4.1 Backbone: Switch to a Stronger Feature Extractor

**Current state:** Frozen ResNet50 (ImageNet pretrained, 2048-d output).

**Why it matters:** ResNet50 was designed for 1,000-class natural image classification. Dermoscopic images differ structurally (uniform lighting, high-frequency texture patterns, lesion symmetry asymmetry). A frozen backbone cannot adapt to these domain-specific features.

#### Recommended alternatives

| Backbone | Params | Dermoscopy-relevant benefits |
|---|---|---|
| **EfficientNetV2-S** | 22M | Efficient compound scaling; state-of-the-art on HAM10000 |
| **EfficientNetV2-L** | 120M | Best accuracy on HAM10000 per 2024 literature |
| **Swin Transformer-B** | 88M | Hierarchical windows; captures local lesion texture + global shape |
| **DINOv2-ViT-B/14** | 86M | Self-supervised on 142M images; strong transfer to dermoscopy |
| **ConvNeXt-B** | 89M | Pure CNN; matches transformer accuracy; OpenCV-friendly ONNX export |

#### Action items

1. Replace `torchvision.models.resnet50` in `model_design/model.py` with EfficientNetV2-S as a first step — same `timm` or `torchvision` API, minimal code change.
2. Set `FEATURE_DIM` to match the new backbone's pooled output dimension (e.g., 1280 for EfficientNetV2-S).
3. Re-export ONNX with `export_opencv.py`; verify OpenCV DNN compatibility.

```python
# Example swap in model.py
import timm
self.backbone = timm.create_model(
    "tf_efficientnetv2_s",
    pretrained=True,
    num_classes=0,          # drop classifier head
    global_pool="avg",      # 1280-d pooled output
)
FEATURE_DIM = 1280
```

---

### 4.2 End-to-End Fine-Tuning Instead of Frozen Extraction

**Current state:** All ResNet50 weights are frozen (`freeze_backbone=True`). The model already exposes `unfreeze_layer()` but it is never called.

**Why it matters:** Fine-tuning the backbone on skin lesion data consistently produces +3–8% accuracy improvements over frozen extraction in the literature, even when starting from ImageNet weights. The final 1–2 residual blocks learn dermoscopy-specific texture representations that frozen layers cannot acquire.

#### Recommended strategy: Progressive Unfreezing

1. **Phase 1 (warm-up):** Freeze backbone. Train classifier head for 5–10 epochs with lr = 1e-3.
2. **Phase 2 (fine-tune top layers):** Unfreeze `layer4` (final residual block). Train with lr = 1e-4 for 10–15 epochs.
3. **Phase 3 (full fine-tune):** Unfreeze all layers. Train with lr = 1e-5 for 10–20 epochs with cosine annealing.

Use **differential learning rates**: lower rates for early layers (which retain general low-level features), higher for later layers.

```python
# Differential learning rates example
optimizer = torch.optim.AdamW([
    {"params": model.backbone.layer1.parameters(), "lr": 1e-6},
    {"params": model.backbone.layer2.parameters(), "lr": 1e-5},
    {"params": model.backbone.layer3.parameters(), "lr": 5e-5},
    {"params": model.backbone.layer4.parameters(), "lr": 1e-4},
    {"params": model.head.parameters(),            "lr": 1e-3},
])
```

**Expected improvement:** +4–8% AUC vs. frozen backbone on dermoscopy.

---

### 4.3 Replace SVM with a Learned Classifier Head

**Current state:** PCA(128) → StandardScaler → RBF-SVM.

**Why it matters:** The SVM is trained on 320 samples (80% of 400). With end-to-end training, a neural classification head learns jointly with the backbone, enabling the feature extractor to produce representations optimised directly for the classification boundary—something a post-hoc SVM cannot do.

#### Recommended replacement

Replace SVM with a lightweight MLP head attached directly to the backbone:

```python
class MelanomaClassifier(nn.Module):
    def __init__(self, backbone_dim: int = 1280):
        super().__init__()
        self.backbone = ...  # EfficientNetV2-S or similar
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),  # binary logit
        )

    def forward(self, x):
        features = self.backbone(x)  # (B, backbone_dim)
        logit = self.head(features)  # (B, 1)
        return logit
```

Train end-to-end with BCEWithLogitsLoss (or Focal Loss, see §4.8).

**For C++ deployment:** The entire model (backbone + head) can be exported as a single ONNX graph, maintaining the existing OpenCV DNN pipeline. The output node becomes a single logit instead of a 2048-d feature vector.

---

### 4.4 Use All Available Training Data (Remove Sampling Cap)

**Current state:** `N_SAMPLES_PER_CLASS = 200`. HAM10000 contains ~1,113 melanoma images and ~8,902 non-melanoma images.

**Why it matters:** The pipeline uses only **~18% of available melanoma images** and **~2.2% of non-melanoma images**. More data is almost always the single highest-leverage improvement for deep learning models.

#### Action items

1. Remove or significantly raise `N_SAMPLES_PER_CLASS`. Use the full dataset.
2. Switch from under-sampling to **class-weighted loss** or **oversampling** (see §4.5) to handle imbalance without discarding data.
3. Use stratified K-fold cross-validation (k=5) instead of a single 80/20 split to get reliable performance estimates from the full dataset.

**Impact:** Moving from 400 to 10,015 samples is likely the single biggest improvement available without changing any other component.

---

### 4.5 Address Class Imbalance Properly

**Current state:** Down-sampling both classes to 200 samples each. HAM10000 has ~11:1 non-melanoma to melanoma ratio.

**Why it matters:** Down-sampling discards 98% of available non-melanoma images, reducing the diversity the model learns. Over-sampling the minority class or reweighting the loss preserves all data.

#### Recommended techniques (in order of preference)

**Option A — Class-weighted loss (easiest, most principled):**
```python
# Compute inverse frequency weights
n_mel, n_other = 1113, 8902
weight = torch.tensor([n_mel / (n_mel + n_other),  # weight for class 0
                        n_other / (n_mel + n_other)])  # weight for class 1
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_other / n_mel))
```

**Option B — Focal Loss (see §4.8):** Down-weights easy majority-class samples automatically.

**Option C — SMOTE / oversampling in feature space:** Synthesize new minority class examples by interpolating between existing feature vectors.

**Option D — GAN-based augmentation:** Train a conditional GAN (StyleGAN2 or similar) on melanoma images to generate synthetic melanoma training samples. Published work on HAM10000 shows GAN augmentation improves minority class F1 by 5–10%.

---

### 4.6 Advanced Data Augmentation

**Current state:** No training augmentation. Only 3×3 median blur at inference time.

**Why it matters:** Dermoscopic datasets are small relative to the complexity of the task. Augmentation is the most cost-effective way to increase effective training set size and model robustness.

#### Recommended augmentation pipeline

Apply the following **randomly during training**, **not at inference**:

```python
import albumentations as A

train_transform = A.Compose([
    # Spatial
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15,
                       rotate_limit=45, p=0.7),
    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),

    # Color & lighting
    A.RandomBrightnessContrast(brightness_limit=0.2,
                               contrast_limit=0.2, p=0.6),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                         val_shift_limit=10, p=0.5),
    A.CLAHE(clip_limit=4.0, p=0.3),

    # Mixing strategies
    # Apply CutMix / MixUp at the batch level in the training loop

    # Noise
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.ISONoise(p=0.2),

    # Blur (simulate focus variation)
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=(3, 7)),
    ], p=0.2),

    # Resize & normalise
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

#### MixUp and CutMix

Apply at batch level for regularisation:

```python
# MixUp example (training loop)
lam = np.random.beta(alpha, alpha)
mixed_x = lam * x + (1 - lam) * x[rand_index]
mixed_y = lam * y + (1 - lam) * y[rand_index]
loss = criterion(model(mixed_x), mixed_y)
```

CutMix is particularly effective for dermoscopy as it preserves spatial structure: cut a rectangular region from one image and paste it into another, mixing labels proportionally.

**Expected improvement:** Augmentation alone typically adds +3–5% AUC on skin lesion datasets.

---

### 4.7 Hair Artifact Handling

**Current state:** 3×3 median blur is applied. This partially softens hair strands but does not remove them.

**Why it matters:** Hair occludes lesion features in dermoscopic images and is one of the most-cited sources of classification error. Two approaches exist:

#### Option A — DL-based inpainting (state-of-the-art)

Use a dedicated hair removal network (e.g., DPA-HairNet, published 2024) as a preprocessing step. These models detect hair strands and inpaint the underlying lesion texture.

```
Raw image → HairNet (segmentation + inpainting) → Clean lesion image → Classifier
```

Papers report 2–4% classification accuracy improvement after hair removal on HAM10000.

#### Option B — Hair augmentation (for robustness)

Instead of removing hair, augment training images **with synthetic hair** to make the model invariant to it. The `TriAug` library provides `HairAugmentation`:

```python
from triAug import HairAugmentation
aug = HairAugmentation(hairs=5, width=(1, 2), p=0.5)
```

This approach is preferred when the goal is model robustness rather than input cleaning.

#### Recommendation

Use **Option B (hair augmentation)** during training and **Option A** as an optional preprocessing step during inference if compute budget allows. The combination yields the most robust results.

---

### 4.8 Loss Function Improvements

**Current state:** SVM optimises hinge loss implicitly. No explicit loss function tunable for clinical needs.

#### Focal Loss (primary recommendation)

Focal Loss (Lin et al., 2017) downweights easy negative examples, forcing the model to focus on hard/minority examples:

```
FL(p_t) = −α_t (1 − p_t)^γ log(p_t)
```

- `γ = 2` (standard starting point; tune on validation set)
- `α = 0.75` (upweight the melanoma class)

This directly addresses the class imbalance without discarding data. Published results on HAM10000 show Focal Loss + Swin Transformer achieving 87.71% vs. 82–84% with cross-entropy.

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()
```

#### Label Smoothing

Prevents overconfidence and improves calibration — important for clinical deployment where probabilities matter:

```python
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(imbalance_ratio),
    label_smoothing=0.05,  # soften hard targets by 5%
)
```

---

### 4.9 Multi-Scale Input Resolution

**Current state:** All images resized to 224×224.

**Why it matters:** Dermoscopic features exist at multiple scales: fine texture (pigment network) at high resolution, overall shape asymmetry at low resolution. Standard ResNet50 at 224px may miss fine-grained texture clues.

#### Recommendations

1. **Increase resolution to 384×384 or 512×512.** EfficientNetV2-L and ViT-L/16 are pretrained at higher resolutions (384×384) and benefit from it. Published results show +2–3% AUC when increasing resolution from 224 to 384 on skin lesion tasks.

2. **Multi-scale feature fusion:** Feed the same image at multiple resolutions to the backbone and concatenate features before the classifier head.

3. **Test-Time Augmentation (TTA):** At inference, generate 5–10 augmented versions of each image (flips, rotations), run forward passes, and average predictions. Typically adds +1–2% AUC at zero training cost:

```python
def predict_with_tta(model, image, n_augments=8):
    predictions = []
    for _ in range(n_augments):
        aug_image = random_tta_transform(image)
        with torch.no_grad():
            pred = torch.sigmoid(model(aug_image))
        predictions.append(pred)
    return torch.stack(predictions).mean(0)
```

---

### 4.10 Segmentation-Aware Training

**Current state:** Lesion masks are applied as hard binary masks in preprocessing (background → black). The mask is used as a simple input filter, not as a training signal.

**Why it matters:** The segmentation masks contain precise lesion boundary information that can supervise the feature extractor to focus on the correct region.

#### Recommended approaches

**Option A — Attention regularisation:** Use the segmentation mask to supervise the model's spatial attention maps (Grad-CAM or learned attention weights should correlate with the lesion mask). Add a spatial attention consistency loss:

```
L_total = L_classification + λ · L_attention_consistency
```

**Option B — Multi-task learning:** Train a shared backbone simultaneously on:
1. Binary melanoma classification (primary task)
2. Lesion segmentation (auxiliary task, U-Net decoder head)

The segmentation task forces the backbone to learn spatially precise lesion representations that benefit classification. Joint training on HAM10000 typically improves classification AUC by 1–3%.

**Option C — Mask-guided attention (soft masking):** Instead of zeroing out background pixels (hard masking), use the mask as a soft spatial weight in an attention layer, allowing the model to see context while weighting lesion pixels more heavily.

---

### 4.11 Ensemble Methods

**Current state:** Single model (ResNet50 + SVM).

**Why it matters:** The best-performing published systems on HAM10000 are all ensembles. Combining diverse models reduces variance and improves robustness.

#### Recommended ensemble strategies

**Strategy 1 — Diverse architecture ensemble (best performance):**

Train 3–5 models with different architectures:
- EfficientNetV2-S (fast, accurate)
- EfficientNetV2-L or ConvNeXt-B (large CNN)
- Swin Transformer-B (global attention)

Average their softmax probabilities at inference. Published ensembles on HAM10000 report 94–99% accuracy on 7-class tasks.

**Strategy 2 — Snapshot ensemble (low cost):**

Use cyclical learning rate schedules (cosine annealing with restarts). Save model checkpoints at each LR minimum. Ensemble 5–10 snapshots from a single training run. Adds +1–2% AUC with no extra training.

**Strategy 3 — Stochastic Weight Averaging (SWA):**

Average model weights (not predictions) over the final few training epochs:
```python
from torch.optim.swa_utils import AveragedModel, SWALR
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=5e-5)
# update swa_model.update_parameters(model) each epoch
```

SWA improves generalisation with essentially zero additional cost and typically adds +0.5–2% on skin lesion benchmarks.

**For C++ deployment:** Export the ensemble's averaged predictions via a single ONNX model (weighted sum of logits baked in) or run multiple ONNX networks and average in C++.

---

### 4.12 Self-Supervised Pre-Training

**Current state:** ImageNet-supervised pretrained ResNet50.

**Why it matters:** ImageNet labels introduce bias toward object recognition features (edges, textures matching real-world objects). Self-supervised pretraining on medical/dermoscopic images learns representations closer to the downstream task without requiring labels.

#### Recommended approaches

**DINOv2 (state-of-the-art, drop-in replacement):**

DINOv2 ViT-B/14 pretrained on 142M diverse images produces features competitive with supervised models and has been shown to outperform supervised ResNet50 on HAM10000 with minimal fine-tuning:

```python
import torch
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
# model outputs 768-d features from [CLS] token
# Fine-tune on HAM10000 with small learning rate
```

**Domain-adaptive SSL (best if compute available):**

Continue DINO or MAE pretraining on unlabeled dermoscopic images (use all available ISIC data: ISIC 2018, 2019, 2020 challenges provide ~50K images total). This adapts the visual representation to dermoscopy domain before fine-tuning on labeled HAM10000 data.

Published results (arXiv:2412.00702, 2024): DINO-initialized models show significantly better generalisation across different scanner types and clinical sites compared to ImageNet-supervised baselines.

---

### 4.13 Clinical Metadata Integration (Multimodal)

**Current state:** Image only.

**Why it matters:** HAM10000 provides rich metadata: patient age, sex, lesion anatomical location, and confirmation type (histopathology, follow-up, expert consensus). Clinical guidelines for melanoma explicitly use these variables (e.g., lesions on back/face in older males are higher risk).

Published result (medRxiv 2024): Multimodal model (image + metadata) achieved **94.11% accuracy, AUC 0.9426** on HAM10000 7-class, compared to image-only baseline of ~88–90%.

#### Recommended architecture

```python
class MultimodalClassifier(nn.Module):
    def __init__(self, image_backbone_dim: int, metadata_dim: int):
        super().__init__()
        self.image_encoder = ...  # EfficientNetV2-S or similar
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.fusion = nn.Sequential(
            nn.Linear(image_backbone_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, image, metadata):
        img_feat = self.image_encoder(image)
        meta_feat = self.metadata_encoder(metadata)
        fused = torch.cat([img_feat, meta_feat], dim=1)
        return self.fusion(fused)
```

**Metadata features to encode:**
- Age (continuous, normalised)
- Sex (binary)
- Anatomical location (one-hot: 7 locations)
- Image acquisition type (one-hot: dermoscopy vs. clinical)

**Note:** HAM10000 metadata CSV already loaded in `pipeline.cpp`. The `dx_type` and `localization` columns are available but currently unused.

---

### 4.14 Explainability / Clinical Interpretability

**Current state:** Black-box SVM predictions. No spatial explanations.

**Why it matters:** Clinicians will not trust and cannot use a black-box system. Regulatory approval (FDA, CE marking) for AI-assisted medical devices increasingly requires explainability. All leading HAM10000 systems in 2024 include XAI.

#### Recommended techniques

**Grad-CAM (minimum viable):**

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

cam = GradCAM(model=model, target_layers=[model.backbone.layer4[-1]])
grayscale_cam = cam(input_tensor=img_tensor)
visualization = show_cam_on_image(img_rgb, grayscale_cam[0])
```

Grad-CAM highlights which image regions drove the prediction. For melanoma, these should correspond to the lesion (verifiable against the existing segmentation masks).

**Clinical validation:** Compute the **IoU between the Grad-CAM heatmap and the lesion mask**. High IoU confirms the model is attending to the right region and not spurious background features. This is a free quality check using data already in the pipeline.

**SHAP (for feature-level explanations):** If metadata integration is added (§4.13), SHAP values identify which clinical features (age, location, sex) most influenced a given prediction — directly interpretable by clinicians.

**Attention map visualisation:** Transformer-based models (Swin, ViT, DINOv2) naturally produce spatial attention weights. These can be overlaid on the image without any additional computation.

---

### 4.15 Evaluation Protocol Improvements

**Current state:** Single 80/20 random split, ~80 test samples, binary accuracy/precision/recall/F1. AUC not reported.

**Why it matters:** 80 test samples is insufficient for statistically meaningful clinical benchmarking. Confidence intervals are very wide. Single splits also risk favourable/unfavourable random splits.

#### Recommended evaluation protocol

1. **5-fold stratified cross-validation** on the full dataset. Report mean ± std across folds.

2. **AUC-ROC** (primary metric for imbalanced binary classification). This is the standard metric in ISIC challenge submissions and the clinical literature.

3. **AUC-PR (Precision-Recall AUC):** More informative than ROC-AUC for highly imbalanced tasks. Measures performance at the relevant operating point.

4. **Sensitivity at 95% specificity:** Clinical standard. "At a false positive rate of 5%, what is our true positive rate for melanoma?"

5. **DeLong test** for statistically comparing AUC between baseline and improved models.

6. **Bootstrap confidence intervals** (n=1000 resamples) on all reported metrics.

```python
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import resample

def bootstrap_auc(y_true, y_score, n=1000):
    aucs = [roc_auc_score(*resample(y_true, y_score, stratify=y_true))
            for _ in range(n)]
    return np.mean(aucs), np.percentile(aucs, [2.5, 97.5])
```

---

## 5. Prioritised Roadmap

Ordered by estimated impact-to-effort ratio:

### Tier 1 — High Impact, Low Effort (do these first)

| # | Change | Estimated Gain | Effort |
|---|---|---|---|
| 1 | Remove 200-sample cap; use full dataset | +8–12% F1 | Low |
| 2 | Add basic augmentation (flips, rotations, colour jitter) | +3–5% AUC | Low |
| 3 | Switch to Focal Loss | +2–4% AUC | Low |
| 4 | Report AUC-ROC; use 5-fold CV | Better estimates | Low |

### Tier 2 — High Impact, Moderate Effort

| # | Change | Estimated Gain | Effort |
|---|---|---|---|
| 5 | Replace ResNet50 with EfficientNetV2-S | +3–6% AUC | Medium |
| 6 | Unfreeze final 2 backbone layers + progressive fine-tuning | +4–8% AUC | Medium |
| 7 | Replace SVM with MLP head, train end-to-end | +5–10% AUC | Medium |
| 8 | Add CutMix/MixUp augmentation | +2–3% AUC | Medium |
| 9 | Increase input resolution to 384×384 | +2–3% AUC | Medium |

### Tier 3 — High Impact, Higher Effort

| # | Change | Estimated Gain | Effort |
|---|---|---|---|
| 10 | Add metadata integration (multimodal) | +3–6% AUC | High |
| 11 | Ensemble 3 diverse architectures | +3–5% AUC | High |
| 12 | Multi-task learning with segmentation | +1–3% AUC | High |
| 13 | DINO/MAE pre-training on unlabeled dermoscopy data | +2–5% AUC | Very High |
| 14 | Grad-CAM XAI integration + IoU clinical validation | Interpretability | High |

### Tier 4 — Polish

| # | Change | Benefit |
|---|---|---|
| 15 | Hair augmentation / DPA-HairNet removal | Robustness |
| 16 | Test-Time Augmentation (TTA) | +1–2% AUC, free at test time |
| 17 | Stochastic Weight Averaging (SWA) | +0.5–1% AUC, free |
| 18 | Bootstrap confidence intervals | Statistical rigor |

---

## 6. Expected Performance Gains

Assuming full implementation of Tier 1 + Tier 2 improvements on the complete HAM10000 binary task:

| System | Binary Mel AUC | F1 | Notes |
|---|---|---|---|
| **Current (our system)** | ~0.85 (est.) | 81.6% | 400-sample subset |
| After Tier 1 only | ~0.91–0.93 | ~86–89% | Full dataset + augmentation + focal loss |
| After Tier 1 + 2 | ~0.95–0.97 | ~91–94% | Fine-tuned EfficientNetV2 + MLP head |
| After Tier 1 + 2 + 3 | ~0.97–0.99 | ~94–96% | Ensemble + multimodal + multitask |
| **Published SOTA (binary)** | ~0.97–0.99 | ~93–97% | Top HAM10000 binary systems |

These projections are derived from additive gain estimates in the cited literature. Actual results may vary depending on implementation quality, hyperparameter tuning, and hardware.

---

## 7. References

### Architecture & Training
- Tan, M., Le, Q. (2021). *EfficientNetV2: Smaller Models and Faster Training.* ICML. arXiv:2104.00298
- Liu, Z., et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV. arXiv:2103.14030
- Oquab, M., et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.* arXiv:2304.07193
- He, K., et al. (2022). *Masked Autoencoders Are Scalable Vision Learners.* CVPR. arXiv:2111.06377

### Skin Lesion Classification — HAM10000
- [Deep Ensemble Learning for Multiclass Skin Lesion Classification](https://www.mdpi.com/2306-5354/12/9/934) — MDPI Bioengineering, 2025
- [Accurate Skin Lesion Classification Using Multimodal Learning on HAM10000 and ISIC 2017](https://www.medrxiv.org/content/10.1101/2024.05.30.24308213v5) — medRxiv, 2024
- [Skin Cancer Segmentation and Classification Using Vision Transformer](https://pmc.ncbi.nlm.nih.gov/articles/PMC10858797/) — PMC, 2024
- [Towards Automated Differential Diagnosis Using Deep Learning and Imbalance-Aware Strategies](https://arxiv.org/html/2601.00286) — arXiv, 2025
- [A Deep Learning Approach for Automated Skin Lesion Diagnosis with Explainable AI](https://arxiv.org/html/2601.00964) — arXiv, 2025
- [Enhancing Skin Disease Classification Leveraging Transformer-based Deep Learning and XAI](https://arxiv.org/html/2407.14757v1) — arXiv, 2024

### Class Imbalance & Loss Functions
- Lin, T.Y., et al. (2017). *Focal Loss for Dense Object Detection.* ICCV. arXiv:1708.02002
- [Tackling Class Imbalanced Dermoscopic Image Classification Using Data Augmentation and GAN](https://link.springer.com/article/10.1007/s11042-023-17067-1) — Springer, 2023
- [Algorithmic Fairness in Lesion Classification by Mitigating Class Imbalance](https://dl.acm.org/doi/10.1007/978-3-031-72378-0_35) — MICCAI, 2024

### Data Augmentation
- DeVries, T., Taylor, G.W. (2017). *Improved Regularization of CNNs with CutOut.* arXiv:1708.04552
- Yun, S., et al. (2019). *CutMix: Training Strategy that Makes Use of Sample Mixing.* ICCV. arXiv:1905.04899
- Zhang, H., et al. (2018). *MixUp: Beyond Empirical Risk Minimization.* ICLR. arXiv:1710.09412
- [Advancing Dermoscopy Through Synthetic Hair Dataset and Deep Learning-Based Hair Removal](https://www.researchgate.net/publication/385962077) — ResearchGate, 2024

### Self-Supervised Learning for Dermoscopy
- [Automatized Self-Supervised Learning for Skin Lesion Screening](https://www.nature.com/articles/s41598-024-61681-4) — Nature Scientific Reports, 2024
- [Enhancing Skin Lesion Classification Generalization with Active Domain Adaptation (DINO)](https://arxiv.org/html/2412.00702v2) — arXiv, 2024

### Melanoma Ensemble Methods
- [AI-Driven Enhancement of Skin Cancer Diagnosis: Two-Stage Voting Ensemble](https://pmc.ncbi.nlm.nih.gov/articles/PMC11720667/) — PMC, 2025
- [Hybrid Deep Learning Framework for Melanoma Diagnosis](https://www.mdpi.com/2075-4418/14/19/2242) — MDPI Diagnostics, 2024
- [Improved Performance on Melanoma Classification Using Deep Learning Ensemble Technique](https://journals.sagepub.com/doi/10.1177/1088467X251320265) — SAGE, 2025

### Explainability
- Selvaraju, R.R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks.* ICCV. arXiv:1610.02391
- [Multimodal Deep Learning Ensemble Framework for Skin Cancer Detection](https://www.nature.com/articles/s41598-025-30534-z) — Nature Scientific Reports, 2025

---

*Document prepared: March 2026. Based on literature search covering publications through early 2026.*
