"""
Exports and quantizes the Whisper Small model for on-device NPU inference on
a Snapdragon target, using Qualcomm AI Hub's `qai_hub_models` toolchain.

This is a thin, convenience wrapper around the documented AI Hub export
command — see:
https://www.qualcomm.com/developer/blog/2025/05/deploy-ai-models-on-snapdragon-x-elite-with-qualcomm-ai-hub

Prerequisites:
  pip install -r requirements-snapdragon.txt
  qai-hub configure --api_token YOUR_API_TOKEN   # free account at aihub.qualcomm.com

Usage:
  python scripts/export_whisper_qnn.py \
      --device "Snapdragon X Elite CRD" \
      --out models/whisper_small_quantized
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model",
        default="whisper_small_quantized",
        help="qai_hub_models model id to export (default: whisper_small_quantized, "
        "already w8a16-quantized and precompiled for Snapdragon X Elite).",
    )
    p.add_argument(
        "--device",
        default="Snapdragon X Elite CRD",
        help='Target device string as recognized by AI Hub, e.g. "Snapdragon X Elite CRD".',
    )
    p.add_argument(
        "--target-runtime",
        default="precompiled_qnn_onnx",
        help="Export runtime target. precompiled_qnn_onnx runs directly on the NPU via QNN.",
    )
    p.add_argument("--out", default="models/whisper_small_quantized", help="Output directory for exported artifacts")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cmd = [
        sys.executable,
        "-m",
        f"qai_hub_models.models.{args.model}.export",
        "--target-runtime",
        args.target_runtime,
        "--device",
        args.device,
        "--output-dir",
        args.out,
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            "\nExport failed. Common causes:\n"
            "  - Not signed in: run `qai-hub configure --api_token YOUR_API_TOKEN`\n"
            "  - Wrong Python architecture on Windows-on-Snapdragon: use 64-bit "
            "(AMDx64) Python, not ARM64\n"
            "  - Model id / device string changed between qai_hub_models releases: "
            "run `python -m qai_hub_models.models --list` to see current model ids\n",
            file=sys.stderr,
        )
        sys.exit(result.returncode)

    print(f"\nDone. Exported artifacts should be under: {args.out}")
    print("Point the app at this directory with:")
    print(f"  python main.py --engine qnn --model-dir {args.out}")


if __name__ == "__main__":
    main()
