"""
Collects timestamped caption segments for a session and exports them as
plain text and/or SubRip (.srt) subtitle files.

Pure logic, no I/O side effects until export_txt/export_srt are called, so
this module is fully unit-testable — see tests/test_transcript_writer.py.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Segment:
    start_time: float  # seconds
    end_time: float    # seconds
    text: str


@dataclass
class TranscriptSession:
    segments: List[Segment] = field(default_factory=list)

    def add(self, start_time: float, end_time: float, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.segments.append(Segment(start_time, end_time, text))

    def to_txt(self) -> str:
        """Plain transcript: one line per segment, with a [mm:ss] timestamp."""
        lines = []
        for seg in self.segments:
            lines.append(f"[{_format_mmss(seg.start_time)}] {seg.text}")
        return "\n".join(lines) + ("\n" if lines else "")

    def to_srt(self) -> str:
        """Standard SubRip subtitle format."""
        blocks = []
        for i, seg in enumerate(self.segments, start=1):
            blocks.append(
                f"{i}\n"
                f"{_format_srt_timestamp(seg.start_time)} --> {_format_srt_timestamp(seg.end_time)}\n"
                f"{seg.text}\n"
            )
        return "\n".join(blocks)

    def export(self, export_dir: str, session_prefix: str, formats: List[str]) -> List[Path]:
        out_dir = Path(export_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = out_dir / f"{session_prefix}_{timestamp}"

        written: List[Path] = []
        if "txt" in formats:
            path = base.with_suffix(".txt")
            path.write_text(self.to_txt(), encoding="utf-8")
            written.append(path)
        if "srt" in formats:
            path = base.with_suffix(".srt")
            path.write_text(self.to_srt(), encoding="utf-8")
            written.append(path)
        return written


def _format_mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _format_srt_timestamp(seconds: float) -> str:
    td = dt.timedelta(seconds=max(0.0, seconds))
    total_ms = int(td.total_seconds() * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
