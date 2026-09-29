"""
Microphone / system-audio capture with an overlapping sliding-window chunker.

The chunk-window math is factored into a pure function (`compute_chunk_windows`)
so it can be unit-tested without touching an actual audio device — see
tests/test_chunker.py.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Generator, List, Tuple

import numpy as np

try:
    import sounddevice as sd
except OSError:
    # PortAudio's native library may not be present on this machine (e.g. a
    # headless CI box). Audio capture will simply be unavailable; pure-logic
    # helpers below still work.
    sd = None


@dataclass
class AudioChunk:
    samples: np.ndarray      # float32, shape (n_samples,)
    sample_rate: int
    start_time: float        # seconds since the capture session started
    end_time: float


def compute_chunk_windows(
    total_samples: int,
    sample_rate: int,
    chunk_seconds: float,
    overlap_seconds: float,
) -> List[Tuple[int, int]]:
    """
    Pure function: given a total number of samples, return a list of
    (start_sample, end_sample) windows implementing a sliding window with
    the requested overlap. No I/O — easy to unit test.
    """
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be >= 0 and < chunk_seconds")

    chunk_samples = int(round(chunk_seconds * sample_rate))
    step_samples = int(round((chunk_seconds - overlap_seconds) * sample_rate))
    if step_samples <= 0:
        raise ValueError("step_samples resolved to <= 0; increase chunk_seconds or reduce overlap")

    windows: List[Tuple[int, int]] = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        windows.append((start, end))
        if end == total_samples:
            break
        start += step_samples
    return windows


class StreamingAudioCapture:
    """
    Captures audio from the default (or a specified) input device and yields
    overlapping AudioChunk windows in real time via `chunks()`.

    Usage:
        capture = StreamingAudioCapture(sample_rate=16000, chunk_seconds=4.0,
                                         overlap_seconds=1.0)
        capture.start()
        for chunk in capture.chunks():
            ...
        capture.stop()
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_seconds: float = 4.0,
        overlap_seconds: float = 1.0,
        input_device: int | None = None,
    ):
        if sd is None:
            raise RuntimeError(
                "sounddevice / PortAudio is not available on this system. "
                "Install PortAudio (e.g. `apt install portaudio19-dev` on "
                "Linux, bundled on Windows) and reinstall sounddevice."
            )
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.input_device = input_device

        self._chunk_samples = int(round(chunk_seconds * sample_rate))
        self._step_samples = int(round((chunk_seconds - overlap_seconds) * sample_rate))

        self._buffer = np.zeros(0, dtype=np.float32)
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream: "sd.InputStream | None" = None
        self._running = False
        self._elapsed_samples = 0

    def _callback(self, indata, frames, time_info, status):  # noqa: ANN001
        if status:
            # Non-fatal (e.g. input overflow); log and continue.
            print(f"[audio_capture] stream status: {status}")
        mono = indata[:, 0] if indata.ndim > 1 else indata
        self._q.put(mono.copy())

    def start(self) -> None:
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            device=self.input_device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def chunks(self) -> Generator[AudioChunk, None, None]:
        """Blocking generator: yields AudioChunk objects as enough audio accumulates."""
        while self._running:
            try:
                block = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            self._buffer = np.concatenate([self._buffer, block])

            while len(self._buffer) >= self._chunk_samples:
                window = self._buffer[: self._chunk_samples]
                start_t = self._elapsed_samples / self.sample_rate
                end_t = (self._elapsed_samples + self._chunk_samples) / self.sample_rate
                yield AudioChunk(
                    samples=window.copy(),
                    sample_rate=self.sample_rate,
                    start_time=start_t,
                    end_time=end_t,
                )
                self._buffer = self._buffer[self._step_samples :]
                self._elapsed_samples += self._step_samples


class MockAudioSource:
    """
    Test/demo double for StreamingAudioCapture: replays a numpy array (e.g.
    loaded from a wav file) as a sequence of AudioChunks, with no real audio
    device involved. Useful for testing app.py end-to-end without a mic.
    """

    def __init__(
        self,
        samples: np.ndarray,
        sample_rate: int,
        chunk_seconds: float = 4.0,
        overlap_seconds: float = 1.0,
    ):
        self.samples = samples.astype(np.float32)
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds

    def chunks(self) -> Generator[AudioChunk, None, None]:
        windows = compute_chunk_windows(
            total_samples=len(self.samples),
            sample_rate=self.sample_rate,
            chunk_seconds=self.chunk_seconds,
            overlap_seconds=self.overlap_seconds,
        )
        for start, end in windows:
            yield AudioChunk(
                samples=self.samples[start:end],
                sample_rate=self.sample_rate,
                start_time=start / self.sample_rate,
                end_time=end / self.sample_rate,
            )

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass
