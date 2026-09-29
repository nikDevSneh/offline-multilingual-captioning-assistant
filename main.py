"""
Entry point for the Offline Multilingual Captioning & Transcription Assistant.

Examples:
    # Development machine, CPU fallback engine, live mic:
    python main.py --engine cpu_dev

    # Snapdragon HP PC, NPU engine:
    python main.py --engine qnn --model-dir models/whisper_small_quantized

    # Headless demo against a wav file, no mic / no GUI required:
    python main.py --engine cpu_dev --demo-wav audio_samples/sample_hi_en.wav --headless
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.app import CaptioningApp  # noqa: E402
from src.audio_capture import MockAudioSource  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--engine", choices=["cpu_dev", "qnn"], help="Override asr.engine from config")
    p.add_argument("--model-dir", help="Override asr.model_dir (qnn engine only)")
    p.add_argument("--demo-wav", help="Run against a wav file instead of the live microphone")
    p.add_argument("--headless", action="store_true", help="Run without the caption overlay GUI (prints to stdout)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.engine:
        config.setdefault("asr", {})["engine"] = args.engine
    if args.model_dir:
        config.setdefault("asr", {})["model_dir"] = args.model_dir

    audio_source = None
    if args.demo_wav:
        import soundfile as sf

        samples, sr = sf.read(args.demo_wav, dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)  # downmix to mono
        audio_source = MockAudioSource(
            samples=samples,
            sample_rate=sr,
            chunk_seconds=config["audio"]["chunk_seconds"],
            overlap_seconds=config["audio"]["chunk_overlap_seconds"],
        )

    app = CaptioningApp(config, audio_source=audio_source)

    if args.headless or args.demo_wav:
        app.run_headless()
    else:
        app.run()

    paths = app.export_transcript()
    if paths:
        print("Transcript saved to:")
        for p in paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
