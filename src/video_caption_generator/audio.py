"""Extract audio from a video file using a bundled ffmpeg binary."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import imageio_ffmpeg


SAMPLE_RATE = 16000
WAV_HEADER_BYTES = 44
WAV_BYTES_PER_SECOND = SAMPLE_RATE * 1 * 2
"""Byte rate of the extracted track: 16 kHz, mono, 16-bit PCM = 32000 B/s.
Used to convert between WAV file size and audio duration without ffprobe."""


def get_ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _ffmpeg_banner(video_path: Path) -> str:
    """ffmpeg's stderr probe output for a file (imageio-ffmpeg ships no ffprobe)."""
    return subprocess.run(
        [get_ffmpeg(), "-i", str(video_path)],
        capture_output=True,
        text=True,
    ).stderr


def has_audio_stream(video_path: Path) -> bool:
    """Whether the file has at least one audio stream."""
    return bool(re.search(r"Stream #\d+:\d+.*: Audio:", _ffmpeg_banner(video_path)))


def media_duration_seconds(video_path: Path) -> float:
    """Container duration in seconds, parsed from ffmpeg's banner (0.0 if unknown)."""
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", _ffmpeg_banner(video_path))
    if not m:
        return 0.0
    h, mn, s = m.groups()
    return int(h) * 3600 + int(mn) * 60 + float(s)


def extract_audio(video_path: Path, audio_path: Path) -> Path:
    """Extract a 16 kHz mono WAV track suitable for the Whisper API."""
    ffmpeg = get_ffmpeg()
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i", str(video_path),
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "-vn",
            "-loglevel", "error",
            str(audio_path),
        ],
        check=True,
    )
    return audio_path


def wav_duration_seconds(audio_path: Path) -> float:
    """Duration of a 16 kHz mono 16-bit PCM WAV, derived from its file size.

    Exact for the format :func:`extract_audio` produces, so accumulating these
    durations gives a drift-free time offset for each chunk.
    """
    return max(0.0, (audio_path.stat().st_size - WAV_HEADER_BYTES) / WAV_BYTES_PER_SECOND)


def split_audio(audio_path: Path, out_dir: Path, chunk_seconds: int) -> list[Path]:
    """Split a WAV into consecutive ``chunk_seconds``-long pieces.

    Uses ffmpeg's segment muxer in a single pass; the PCM stream is copied
    (not re-encoded). Returns the chunk paths in playback order.
    """
    ffmpeg = get_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "chunk_%04d.wav")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i", str(audio_path),
            "-f", "segment",
            "-segment_time", str(chunk_seconds),
            "-c", "copy",
            "-reset_timestamps", "1",
            "-loglevel", "error",
            pattern,
        ],
        check=True,
    )
    return sorted(out_dir.glob("chunk_*.wav"))
