import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.transcript_writer import TranscriptSession, _format_mmss, _format_srt_timestamp


def test_add_skips_empty_text():
    session = TranscriptSession()
    session.add(0.0, 1.0, "   ")
    assert session.segments == []


def test_txt_export_format():
    session = TranscriptSession()
    session.add(0.0, 4.0, "Namaste, this is a test.")
    session.add(4.0, 8.0, "Aap kaise hain?")
    txt = session.to_txt()
    assert "[00:00] Namaste, this is a test." in txt
    assert "[00:04] Aap kaise hain?" in txt


def test_srt_export_format():
    session = TranscriptSession()
    session.add(0.0, 2.5, "Hello world")
    srt = session.to_srt()
    assert "1\n00:00:00,000 --> 00:00:02,500\nHello world" in srt


def test_format_mmss():
    assert _format_mmss(0) == "00:00"
    assert _format_mmss(65) == "01:05"
    assert _format_mmss(3661) == "61:01"


def test_format_srt_timestamp():
    assert _format_srt_timestamp(0) == "00:00:00,000"
    assert _format_srt_timestamp(3725.123) == "01:02:05,123"


def test_export_writes_files(tmp_path):
    session = TranscriptSession()
    session.add(0.0, 2.0, "Test segment")
    paths = session.export(export_dir=str(tmp_path), session_prefix="unit_test", formats=["txt", "srt"])
    assert len(paths) == 2
    for p in paths:
        assert p.exists()
        assert p.read_text(encoding="utf-8").strip() != ""


def test_export_respects_requested_formats_only(tmp_path):
    session = TranscriptSession()
    session.add(0.0, 2.0, "Test segment")
    paths = session.export(export_dir=str(tmp_path), session_prefix="unit_test", formats=["txt"])
    assert len(paths) == 1
    assert paths[0].suffix == ".txt"
