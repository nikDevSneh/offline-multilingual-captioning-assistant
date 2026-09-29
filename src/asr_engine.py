"""
Speech-to-text engine abstraction.

Two implementations:
  - CPUWhisperEngine: faster-whisper on CPU. Works on any machine; used for
    local development and for the test suite / CI.
  - QNNWhisperEngine: the production engine. Loads the Whisper model exported
    and quantized via `scripts/export_whisper_qnn.py` (Qualcomm AI Hub) and
    runs it on the Snapdragon Hexagon NPU through the QNN execution provider.

Both implement the same ASREngine interface so the rest of the app
(src/app.py) doesn't need to know which one it's talking to.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np


@dataclass
class ASRResult:
    text: str
    language: Optional[str]
    latency_seconds: float


class ASREngine(Protocol):
    def transcribe(self, samples: np.ndarray, sample_rate: int) -> ASRResult:
        ...


class CPUWhisperEngine:
    """
    Development / fallback engine using faster-whisper (CTranslate2-based,
    runs fine on CPU). Use this to build and test the app on any laptop
    before deploying to Snapdragon hardware.
    """

    def __init__(self, model_size: str = "small", language: str = "auto"):
        from faster_whisper import WhisperModel  # imported lazily so the
        # rest of the app can be used/tested without this dependency present

        self.language = None if language == "auto" else language
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> ASRResult:
        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            samples,
            language=self.language,
            vad_filter=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        latency = time.perf_counter() - t0
        return ASRResult(text=text, language=info.language, latency_seconds=latency)


class QNNWhisperEngine:
    """
    Production engine: runs the quantized Whisper model on the Snapdragon
    Hexagon NPU via Qualcomm AI Hub's exported artifacts.

    This wraps the pattern documented by Qualcomm AI Hub Models: each model
    in `qai_hub_models` ships a demo/`App` class that wraps encoder + decoder
    inference behind a simple `.transcribe(audio, sample_rate)`-style call.
    Because the exact class name can change between qai_hub_models releases,
    this loader tries the documented entry points in order and raises a
    clear, actionable error (including the reference demo command to run) if
    none of them match what's installed.

    Expected model_dir layout (created by scripts/export_whisper_qnn.py):
        models/whisper_small_quantized/
            encoder.onnx / encoder.bin   (QNN context binary or ONNX)
            decoder.onnx / decoder.bin
            (plus whatever qai_hub_models' export step produces)
    """

    def __init__(self, model_dir: str, language: str = "auto"):
        self.model_dir = Path(model_dir)
        self.language = None if language == "auto" else language
        self._app = self._load_app()

    def _load_app(self):
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}\n"
                "Run scripts/export_whisper_qnn.py first to export and "
                "quantize the Whisper model for your Snapdragon device."
            )
        try:
            # Documented qai_hub_models pattern: a model package exposes
            # `Model.from_pretrained()` plus an `App` wrapper class that
            # knows how to run the encoder/decoder pair end to end.
            from qai_hub_models.models.whisper_small_quantized.model import (
                WhisperSmallQuantized as WhisperModelCls,
            )
            from qai_hub_models.models.whisper_small_quantized.app import WhisperApp

            model = WhisperModelCls.from_pretrained()
            app = WhisperApp(model)
            return app
        except Exception as exc:  # noqa: BLE001 - surface a actionable message
            raise RuntimeError(
                "Could not load the QNN Whisper app from qai_hub_models. "
                "The installed qai_hub_models version may expose a different "
                "API. As a reference, run:\n\n"
                "    python -m qai_hub_models.models.whisper_small_quantized.demo\n\n"
                "and adjust the import in QNNWhisperEngine._load_app() to "
                "match the class names printed/used there.\n"
                f"Original error: {exc}"
            ) from exc

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> ASRResult:
        t0 = time.perf_counter()
        # WhisperApp.transcribe() (per qai_hub_models' documented demo
        # pattern) takes raw float32 PCM audio and the sample rate, and
        # returns the decoded text.
        text = self._app.transcribe(samples, sample_rate)
        latency = time.perf_counter() - t0
        return ASRResult(text=text.strip(), language=self.language, latency_seconds=latency)


def build_engine(engine_name: str, cfg: dict) -> ASREngine:
    """Factory used by app.py / main.py to construct the configured engine."""
    if engine_name == "cpu_dev":
        return CPUWhisperEngine(
            model_size=cfg.get("cpu_model_size", "small"),
            language=cfg.get("language", "auto"),
        )
    if engine_name == "qnn":
        return QNNWhisperEngine(
            model_dir=cfg.get("model_dir", "models/whisper_small_quantized"),
            language=cfg.get("language", "auto"),
        )
    raise ValueError(f"Unknown ASR engine '{engine_name}'. Use 'cpu_dev' or 'qnn'.")
