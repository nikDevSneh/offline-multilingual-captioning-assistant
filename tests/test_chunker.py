import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.audio_capture import compute_chunk_windows


def test_basic_windows_no_overlap():
    # 10 seconds at 16kHz, 2s chunks, no overlap -> 5 windows, no gaps/overlaps
    windows = compute_chunk_windows(total_samples=160_000, sample_rate=16_000, chunk_seconds=2.0, overlap_seconds=0.0)
    assert windows[0] == (0, 32_000)
    assert windows[1] == (32_000, 64_000)
    assert windows[-1][1] == 160_000
    assert len(windows) == 5


def test_overlapping_windows_step_size():
    # 4s chunks with 1s overlap -> step of 3s
    windows = compute_chunk_windows(total_samples=16_000 * 10, sample_rate=16_000, chunk_seconds=4.0, overlap_seconds=1.0)
    starts = [w[0] for w in windows]
    steps = [b - a for a, b in zip(starts, starts[1:])]
    assert all(s == 3 * 16_000 for s in steps)


def test_last_window_covers_remaining_tail_without_overrun():
    windows = compute_chunk_windows(total_samples=100_000, sample_rate=16_000, chunk_seconds=4.0, overlap_seconds=1.0)
    assert windows[-1][1] == 100_000
    assert all(end <= 100_000 for _, end in windows)


def test_short_audio_single_window():
    # audio shorter than one chunk -> a single window covering everything
    windows = compute_chunk_windows(total_samples=8_000, sample_rate=16_000, chunk_seconds=4.0, overlap_seconds=1.0)
    assert windows == [(0, 8_000)]


@pytest.mark.parametrize("overlap", [-1.0, 4.0, 5.0])
def test_invalid_overlap_raises(overlap):
    with pytest.raises(ValueError):
        compute_chunk_windows(total_samples=100_000, sample_rate=16_000, chunk_seconds=4.0, overlap_seconds=overlap)


def test_zero_chunk_seconds_raises():
    with pytest.raises(ValueError):
        compute_chunk_windows(total_samples=100_000, sample_rate=16_000, chunk_seconds=0.0, overlap_seconds=0.0)
