"""
Wires audio capture -> ASR engine -> (optional translation) -> caption
overlay + transcript writer into a running application.

The audio/ASR pipeline runs on a background thread; the caption overlay's
Tkinter mainloop owns the main thread (Tkinter requires this on most
platforms). `CaptioningApp.run()` starts both and blocks until the overlay
window is closed.
"""
from __future__ import annotations

import threading
from typing import Optional

from .asr_engine import ASREngine, build_engine
from .audio_capture import MockAudioSource, StreamingAudioCapture
from .caption_overlay import CaptionOverlay
from .transcript_writer import TranscriptSession
from .translator import Translator, build_translator


class CaptioningApp:
    def __init__(
        self,
        config: dict,
        audio_source: Optional[object] = None,
        engine: Optional[ASREngine] = None,
        translator: Optional[Translator] = None,
    ):
        """
        `audio_source`: inject a MockAudioSource (or any object exposing the
        same start()/stop()/chunks() interface) for testing/demoing without a
        real microphone. Defaults to a real StreamingAudioCapture built from
        config["audio"].

        `engine` / `translator`: inject test doubles (e.g. a stub ASREngine)
        to smoke-test the pipeline wiring without loading real model weights.
        Defaults to building the configured engine/translator from config.
        """
        self.config = config
        self.asr_cfg = config.get("asr", {})
        self.audio_cfg = config.get("audio", {})
        self.translation_cfg = config.get("translation", {})
        self.overlay_cfg = config.get("overlay", {})
        self.transcript_cfg = config.get("transcript", {})

        self.engine: ASREngine = engine or build_engine(self.asr_cfg.get("engine", "cpu_dev"), self.asr_cfg)
        self.translator: Translator = translator or build_translator(self.translation_cfg)
        self.transcript = TranscriptSession()

        self.audio_source = audio_source or StreamingAudioCapture(
            sample_rate=self.audio_cfg.get("sample_rate", 16000),
            channels=self.audio_cfg.get("channels", 1),
            chunk_seconds=self.audio_cfg.get("chunk_seconds", 4.0),
            overlap_seconds=self.audio_cfg.get("chunk_overlap_seconds", 1.0),
            input_device=self.audio_cfg.get("input_device"),
        )

        self.overlay = CaptionOverlay(
            font_size=self.overlay_cfg.get("font_size", 26),
            max_lines=self.overlay_cfg.get("max_lines", 3),
            position=self.overlay_cfg.get("position", "bottom"),
            width_fraction=self.overlay_cfg.get("width_fraction", 0.8),
            opacity=self.overlay_cfg.get("opacity", 0.85),
            background_color=self.overlay_cfg.get("background_color", "#0B2545"),
            text_color=self.overlay_cfg.get("text_color", "#FFFFFF"),
        )

        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ---- pipeline worker (background thread) ----
    def _pipeline_worker(self) -> None:
        self.audio_source.start()
        try:
            for chunk in self.audio_source.chunks():
                if self._stop_event.is_set():
                    break
                result = self.engine.transcribe(chunk.samples, chunk.sample_rate)
                if not result.text:
                    continue

                text = result.text
                if self.translation_cfg.get("enabled", False):
                    text = self.translator.translate(
                        text,
                        self.translation_cfg.get("source_lang", "hi"),
                        self.translation_cfg.get("target_lang", "en"),
                    )

                self.transcript.add(chunk.start_time, chunk.end_time, text)
                self.overlay.update(text)
        finally:
            self.audio_source.stop()

    # ---- process a single chunk directly (used by tests / MockAudioSource runs) ----
    def process_chunk(self, chunk) -> str:
        result = self.engine.transcribe(chunk.samples, chunk.sample_rate)
        text = result.text
        if text and self.translation_cfg.get("enabled", False):
            text = self.translator.translate(
                text,
                self.translation_cfg.get("source_lang", "hi"),
                self.translation_cfg.get("target_lang", "en"),
            )
        if text:
            self.transcript.add(chunk.start_time, chunk.end_time, text)
        return text

    def run(self) -> None:
        """Blocking: starts the pipeline thread and the overlay's mainloop."""
        self._worker_thread = threading.Thread(target=self._pipeline_worker, daemon=True)
        self._worker_thread.start()
        try:
            self.overlay.run()
        finally:
            self.stop()

    def run_headless(self) -> None:
        """Runs the pipeline without the GUI overlay — useful for a MockAudioSource
        demo/benchmark run, or a server-side/no-display environment."""
        self.audio_source.start()
        try:
            for chunk in self.audio_source.chunks():
                text = self.process_chunk(chunk)
                if text:
                    print(f"[{chunk.start_time:6.1f}s] {text}")
        finally:
            self.audio_source.stop()

    def stop(self) -> None:
        self._stop_event.set()
        self.overlay.close()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)

    def export_transcript(self):
        return self.transcript.export(
            export_dir=self.transcript_cfg.get("export_dir", "transcripts"),
            session_prefix=self.transcript_cfg.get("session_prefix", "session"),
            formats=self.transcript_cfg.get("formats", ["txt", "srt"]),
        )
