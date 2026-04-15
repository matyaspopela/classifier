#!/usr/bin/env python3

import os
import random
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
METADATA_CSV = "/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv"
IMG_DIRS = [
    "/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_1/",
    "/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_2/",
]
# Set to None if you did not upload the segmentation dataset
MASK_DIR     = "/kaggle/input/ham10000-lesion-segmentations/HAM10000_segmentations_lesion_tschandl/"
OUTPUT_ONNX  = "/kaggle/working/feature_extractor.onnx"
CHECKPOINT   = "/kaggle/working/best_checkpoint.pt"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
IMAGE_SIZE   = 224   # matches the existing C++ pipeline input size
BATCH_SIZE   = 64    # split evenly across both T4s by DataParallel
NUM_EPOCHS   = 30
LR           = 3e-4
WEIGHT_DECAY = 1e-2
VAL_SPLIT    = 0.15
SEED         = 42


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ---------------------------------------------------------------------------
# Image-path cache (avoids repeated filesystem probing per epoch)
# ---------------------------------------------------------------------------
_img_cache: dict = {}


def find_image(image_id: str) -> str:
    if image_id not in _img_cache:
        for d in IMG_DIRS:
            p = Path(d) / f"{image_id}.jpg"
            if p.exists():
                _img_cache[image_id] = str(p)
                break
    return _img_cache.get(image_id)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class HAM10000Dataset(Dataset):
    def __init__(self, records: list, transform=None, use_masks: bool = True):
        self.records   = records
        self.transform = transform
        self.use_masks = use_masks and MASK_DIR is not None and Path(MASK_DIR).exists()
        if not self.use_masks and use_masks:
            print("WARNING: mask directory not found — training without lesion masks.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        image_id, label = self.records[idx]

        img = cv2.imread(find_image(image_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.use_masks:
            mask_path = Path(MASK_DIR) / f"{image_id}_segmentation.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                img = cv2.bitwise_and(img, img, mask=mask)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return img, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
train_tf = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.CLAHE(clip_limit=4.0, p=0.3),
    A.GaussNoise(p=0.3),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=(3, 7)),
    ], p=0.2),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_tf = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MelanomaClassifier(nn.Module):
    """
    EfficientNetV2-S backbone + small MLP head.

    The head is only used during training to drive end-to-end gradient flow
    so the backbone learns dermoscopy-specific features.  At export time only
    the backbone is serialised; the head is discarded.
    """

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=True,
            num_classes=0,      # drop the timm classification head
            global_pool="avg",  # output: (batch, 1280)
        )
        feat_dim = self.backbone.num_features  # 1280
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),   # single logit — used only during training
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass (training): backbone → head → logit."""
        return self.head(self.backbone(x))  # (batch, 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone only (export / inference): returns L2-normalised 1280-d vector."""
        features = self.backbone(x)                    # (batch, 1280)
        return F.normalize(features, p=2, dim=1)       # unit-sphere, matches C++ pipeline


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Binary focal loss — down-weights easy negatives to focus on hard/minority examples."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt  = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


# ---------------------------------------------------------------------------
# Train / validation helpers
# ---------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs).squeeze(1), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs).squeeze(1)
        total_loss += criterion(logits, labels).item() * len(imgs)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / len(loader.dataset), auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Device: {device}  |  GPUs: {n_gpus}")

    # ── Data ────────────────────────────────────────────────────────────────
    df = pd.read_csv(METADATA_CSV)
    df["label"] = (df["dx"] == "mel").astype(int)
    records = list(zip(df["image_id"], df["label"]))

    train_rec, val_rec = train_test_split(
        records, test_size=VAL_SPLIT, stratify=df["label"], random_state=SEED
    )

    n_mel   = sum(l for _, l in train_rec)
    n_other = len(train_rec) - n_mel
    print(f"Train: {len(train_rec)}  ({n_mel} mel / {n_other} other)  |  Val: {len(val_rec)}")

    use_masks = MASK_DIR is not None
    train_loader = DataLoader(
        HAM10000Dataset(train_rec, train_tf, use_masks=use_masks),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        HAM10000Dataset(val_rec, val_tf, use_masks=use_masks),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = MelanomaClassifier().to(device)
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"DataParallel across {n_gpus} GPUs")

    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # ── Training loop ────────────────────────────────────────────────────────
    best_auc   = 0.0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss           = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        tag = ""
        if val_auc > best_auc:
            best_auc  = val_auc
            core      = model.module if hasattr(model, "module") else model
            best_state = {k: v.cpu().clone() for k, v in core.state_dict().items()}
            torch.save(best_state, CHECKPOINT)
            tag = "  ← best"

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS}  "
            f"tr_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_auc={val_auc:.4f}  lr={scheduler.get_last_lr()[0]:.2e}{tag}"
        )

    print(f"\nBest val AUC: {best_auc:.4f}")

    # ── Export ONNX (OpenCV-compatible: legacy exporter, opset 13) ───────────
    # Load best weights into a clean (non-DataParallel) model, then wrap just
    # the backbone in a thin nn.Module so the export graph contains only the
    # feature-extraction path.  The MLP head is intentionally excluded — the
    # C++ pipeline handles classification via its own PCA + RBF-SVM.
    core = MelanomaClassifier()
    core.load_state_dict(best_state)
    core.eval()

    class BackboneOnly(nn.Module):
        """Thin wrapper: backbone → L2-normalised 1280-d feature vector."""
        def __init__(self, classifier: MelanomaClassifier):
            super().__init__()
            self.backbone = classifier.backbone

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.normalize(self.backbone(x), p=2, dim=1)

    backbone_export = BackboneOnly(core)
    backbone_export.eval()

    dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    # Sanity-check output shape before writing the file
    with torch.no_grad():
        test_out = backbone_export(dummy)
    assert test_out.shape == (1, 1280), f"Unexpected output shape: {test_out.shape}"

    print(f"\nExporting ONNX → {OUTPUT_ONNX}")
    torch.onnx.export(
        backbone_export,
        dummy,
        OUTPUT_ONNX,
        export_params       = True,
        opset_version       = 13,   # highest opset fully supported by OpenCV 4.x DNN
        do_constant_folding = True,
        input_names         = ["image"],
        output_names        = ["features"],
        dynamic_axes        = {"image": {0: "batch_size"}, "features": {0: "batch_size"}},
        dynamo              = False, # CRITICAL: legacy exporter required for OpenCV
    )

    # Optional: constant-fold and simplify with onnx-simplifier
    try:
        import onnx
        from onnxsim import simplify as onnx_simplify

        model_proto, ok = onnx_simplify(onnx.load(OUTPUT_ONNX))
        if ok:
            onnx.save(model_proto, OUTPUT_ONNX)
            print("onnx-simplifier: OK")
        else:
            print("onnx-simplifier: check failed, keeping unsimplified model")
    except ImportError:
        print("onnxsim not installed — skipping simplification")

    size_mb = Path(OUTPUT_ONNX).stat().st_size / 1024 ** 2
    print(f"Done → {OUTPUT_ONNX}  ({size_mb:.1f} MB)")
    print(
        "Output node 'features': (batch, 1280) L2-normalised float32.\n"
        "Drop this file into your C++ pipeline in place of the old feature_extractor.onnx.\n"
        "Only change needed in pipeline.cpp: update the FEATURE_DIM comment (2048 → 1280);\n"
        "PCA, StandardScaler, and SVM code are untouched."
    )


if __name__ == "__main__":
    main()
