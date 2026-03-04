"""
Feature Extractor - ResNet50 backbone for melanoma SVM feature extraction.

Architecture notes:
  - All ResNet50 weights are FROZEN; the model is a pure feature extractor.
  - The final FC layer is replaced with nn.Identity() to expose the 2048-d
    global average-pooled feature vector from layer4.
  - Preprocessing (median filter + ImageNet normalisation) is applied in the
    Dataset, keeping this module lightweight and easy to export to ONNX.

SVM guidance (applied after extraction):
  - Kernel  : RBF is typically best for 2048-d dense feature spaces.
  - Tuning  : Grid-search or Bayesian optimisation over C and gamma.
  - Scaling : StandardScaler before fitting the SVM (zero mean, unit variance).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import torchvision.models as models
from pathlib import Path


class FeatureExtractor(nn.Module):
    """ResNet50 feature extractor producing 2048-d vectors for SVM input."""

    FEATURE_DIM = 2048  # output dimensionality

    def __init__(self, freeze_backbone: bool = True):
        super().__init__()

        # Load ImageNet-pretrained ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Drop the classification head — expose the 2048-d pooled representation
        self.resnet.fc = nn.Identity()

        if freeze_backbone:
            self._freeze_backbone()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def _freeze_backbone(self) -> None:
        """Freeze all ResNet50 parameters (no gradient computation needed)."""
        for param in self.resnet.parameters():
            param.requires_grad = False

    def unfreeze_layer(self, layer_name: str) -> None:
        """Selectively unfreeze a named ResNet layer (e.g. 'layer4') for
        optional fine-tuning experiments."""
        layer = getattr(self.resnet, layer_name, None)
        if layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in ResNet50.")
        for param in layer.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Batch of preprocessed, mask-applied images.
               Shape: (B, 3, 224, 224), dtype: float32, ImageNet-normalised.
        Returns:
            features: (B, 2048) — one L2-normalised vector per sample.
        """
        features = self.resnet(x)               # (B, 2048)
        features = F.normalize(features, p=2, dim=1)  # L2-norm → unit sphere
        return features

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def export_onnx(self, output_path: str | Path, image_size: int = 224) -> None:
        """
        Export the feature extractor to ONNX with a dynamic batch dimension.

        The exported graph accepts  (B, 3, image_size, image_size)  float32
        and produces               (B, 2048)                        float32
        which is ready for ONNX Runtime inference in the C++ pipeline.

        Args:
            output_path : Destination .onnx file path.
            image_size  : Spatial resolution the model was trained with (224).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.eval()
        dummy_input = torch.zeros(1, 3, image_size, image_size)

        # Sanity-check: confirm output is 2048-d, not a classification vector
        with torch.no_grad():
            test_out = self(dummy_input)
        assert test_out.shape == (1, self.FEATURE_DIM), (
            f"Expected output shape (1, {self.FEATURE_DIM}), got {test_out.shape}. "
            "Make sure the FC head is replaced with nn.Identity()."
        )

        dynamic_shapes = {"x": {0: torch.export.Dim("batch_size")}}

        torch.onnx.export(
            self,
            dummy_input,
            str(output_path),
            export_params       = True,
            opset_version       = 18,
            do_constant_folding = True,
            input_names         = ["image"],
            output_names        = ["features"],
            dynamic_shapes      = dynamic_shapes,
        )
        print(f"ONNX model exported → {output_path}  "
              f"(output: batch × {self.FEATURE_DIM})")


# ---------------------------------------------------------------------------
# Standalone export — run this file directly to produce the ONNX artefact
# without running the full training pipeline.
#
#   python model.py
#   python model.py --out /some/other/path/model.onnx
#
# No melanoma data or training needed: the ResNet50 backbone ships with
# ImageNet-pretrained weights whose rich feature representations transfer
# directly to dermoscopic images.  The weights are never modified.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export FeatureExtractor to ONNX")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent.parent / "feature_extractor.onnx"),
        help="Destination .onnx file (default: repo root)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Spatial resolution the model expects (default: 224)",
    )
    args = parser.parse_args()

    print("Loading pretrained ResNet50 weights (no training required) …")
    extractor = FeatureExtractor(freeze_backbone=True)
    extractor.export_onnx(args.out, image_size=args.image_size)

