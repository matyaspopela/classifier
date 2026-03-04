#!/usr/bin/env python3
"""
Re-export the ResNet50 feature extractor as an OpenCV-compatible ONNX model.

Key decisions:
  - dynamo=False forces the LEGACY torch.onnx.export codepath.
    (PyTorch ≥ 2.5 defaults to the dynamo exporter which produces graph nodes
    that OpenCV's DNN module cannot parse — Conv without explicit kernel_size.)
  - Opset 13 — fully supported by OpenCV 4.x DNN module.
  - All weights are embedded inside the single .onnx file (no .onnx.data).
  - onnx-simplifier folds redundant ops and constant-propagates shapes.

Usage (run from model_design/):
    python export_opencv.py                          # writes ../feature_extractor.onnx
    python export_opencv.py --out /tmp/model.onnx
"""

import argparse
import sys
from pathlib import Path

import torch

# Ensure model_design/ is importable regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FeatureExtractor


def main():
    parser = argparse.ArgumentParser(description="Export OpenCV-compatible ONNX")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "feature_extractor.onnx"),
        help="Destination .onnx file (default: repo root)",
    )
    args = parser.parse_args()
    output_path = Path(args.out)

    print("Loading pretrained ResNet50 …")
    extractor = FeatureExtractor(freeze_backbone=True)
    extractor.eval()

    dummy = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        test = extractor(dummy)
    assert test.shape == (1, 2048), f"Unexpected output shape: {test.shape}"

    # ---- Legacy ONNX export (OpenCV-compatible) ----------------------------
    #
    # dynamo=False is CRITICAL on PyTorch ≥ 2.5: without it the new
    # TorchDynamo exporter kicks in, which:
    #   1) ignores opset_version and forces opset 18+
    #   2) stores weights externally in .onnx.data
    #   3) produces Conv nodes without explicit kernel_size attributes
    # All three break OpenCV's readNetFromONNX().
    #
    print("Exporting ONNX (legacy exporter, opset 13, dynamo=False) …")
    torch.onnx.export(
        extractor,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["features"],
        dynamic_axes={
            "image":    {0: "batch_size"},
            "features": {0: "batch_size"},
        },
        dynamo=False,           # ← force legacy exporter
    )

    # Verify all weights are embedded (no external .onnx.data)
    data_file = output_path.with_suffix(".onnx.data")
    if data_file.exists():
        # If the legacy exporter still produced external data, inline it
        print("Inlining external weight data …")
        import onnx
        from onnx.external_data_helper import convert_model_to_external_data
        model = onnx.load(str(output_path), load_external_data=True)
        # Remove external data references so everything is in the protobuf
        for tensor in model.graph.initializer:
            tensor.ClearField("data_location")
            for entry in list(tensor.external_data):
                tensor.external_data.remove(entry)
        onnx.save(model, str(output_path))
        data_file.unlink()
        print(f"Removed external data file: {data_file.name}")

    # ---- Simplify (constant folding, shape inference) ----------------------
    try:
        import onnx
        from onnxsim import simplify as onnx_simplify

        print("Running onnx-simplifier …")
        model = onnx.load(str(output_path))
        model_simp, check = onnx_simplify(model)
        if check:
            onnx.save(model_simp, str(output_path))
            print("Simplified model saved.")
        else:
            print("WARNING: onnx-simplifier check failed; keeping unsimplified model.")
    except ImportError:
        print(
            "NOTE: onnx / onnxsim not installed — skipping simplification.\n"
            "      Install with: pip install onnx onnxsim"
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Done → {output_path}  ({size_mb:.1f} MB, all weights embedded)")


if __name__ == "__main__":
    main()
