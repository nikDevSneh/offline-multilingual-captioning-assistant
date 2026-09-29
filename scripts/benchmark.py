"""
Benchmarks ASR latency on the CPU fallback engine vs. the Snapdragon NPU
(QNN) engine over the same audio sample, and prints a comparison table.

This is the script to run for the "Technical Implementation" evidence in the
Challenge submission — real, measured numbers rather than estimates.

Usage:
  python scripts/benchmark.py --sample audio_samples/sample_hi_en.wav \
      --model-dir models/whisper_small_quantized

If the QNN engine / model isn't available (e.g. running this on a dev
machine rather than the Snapdragon target), the script still reports CPU
numbers and notes that the NPU column was skipped.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf  # noqa: E402

from src.asr_engine import CPUWhisperEngine, QNNWhisperEngine  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", required=True, help="Path to a wav file (mono or stereo) to benchmark against")
    p.add_argument("--model-dir", default="models/whisper_small_quantized", help="Exported QNN model directory")
    p.add_argument("--runs", type=int, default=5, help="Number of repeated runs to average over")
    p.add_argument("--cpu-model-size", default="small", help="faster-whisper model size for the CPU baseline")
    return p.parse_args()


def load_audio(path: str):
    samples, sr = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, sr


def run_engine(engine, samples, sample_rate, runs: int):
    latencies = []
    text = ""
    for _ in range(runs):
        result = engine.transcribe(samples, sample_rate)
        latencies.append(result.latency_seconds)
        text = result.text
    return latencies, text


def main() -> None:
    args = parse_args()
    samples, sr = load_audio(args.sample)
    audio_seconds = len(samples) / sr

    print(f"Audio sample: {args.sample} ({audio_seconds:.1f}s, {sr}Hz)")
    print(f"Averaging over {args.runs} runs\n")

    rows = []

    print("Running CPU baseline (faster-whisper)...")
    cpu_engine = CPUWhisperEngine(model_size=args.cpu_model_size)
    cpu_latencies, cpu_text = run_engine(cpu_engine, samples, sr, args.runs)
    cpu_mean = statistics.mean(cpu_latencies)
    rows.append(("CPU (faster-whisper)", cpu_mean, audio_seconds / cpu_mean, cpu_text))

    print("Running NPU engine (Qualcomm AI Hub / QNN)...")
    try:
        qnn_engine = QNNWhisperEngine(model_dir=args.model_dir)
        qnn_latencies, qnn_text = run_engine(qnn_engine, samples, sr, args.runs)
        qnn_mean = statistics.mean(qnn_latencies)
        rows.append(("NPU (Snapdragon QNN)", qnn_mean, audio_seconds / qnn_mean, qnn_text))
    except Exception as exc:  # noqa: BLE001
        print(f"  Skipped: {exc}\n")

    print(f"\n{'Engine':<24}{'Avg latency (s)':<18}{'Real-time factor':<18}")
    print("-" * 60)
    for name, mean_latency, rtf, _ in rows:
        print(f"{name:<24}{mean_latency:<18.3f}{rtf:<18.2f}")

    if len(rows) == 2:
        speedup = rows[0][1] / rows[1][1]
        print(f"\nNPU speedup over CPU: {speedup:.2f}x")


if __name__ == "__main__":
    main()
