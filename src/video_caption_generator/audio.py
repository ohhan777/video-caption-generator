"""Extract audio from a video file using a bundled ffmpeg binary."""
from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg


def get_ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


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
            "-ar", "16000",
            "-vn",
            "-loglevel", "error",
            str(audio_path),
        ],
        check=True,
    )
    return audio_path
