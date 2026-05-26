"""Change a video's playback speed with ffmpeg.

Video timestamps are scaled with ``setpts=PTS/rate``; audio is sped up with
``atempo``, which keeps pitch natural. A single ``atempo`` only accepts factors
in [0.5, 2.0], so larger changes are split into a chain (e.g. 3.0 -> 2.0*1.5).
Re-encodes the output.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from .audio import get_ffmpeg


def _atempo_chain(rate: float) -> str:
    """Build an ``atempo`` filter chain for an arbitrary positive ``rate``.

    Each stage stays within ffmpeg's supported [0.5, 2.0] range.
    """
    factors: list[float] = []
    remaining = rate
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={f:g}" for f in factors)


def change_speed(
    video_path: Path,
    output_path: Path,
    *,
    rate: float,
    crf: int = 20,
    preset: str = "medium",
) -> Path:
    """Re-time ``video_path`` by ``rate`` (e.g. 1.5 = 1.5x faster)."""
    if rate <= 0:
        raise click.BadParameter("--rate must be greater than 0", param_hint="--rate")

    ffmpeg = get_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filter_complex = (
        f"[0:v]setpts=PTS/{rate:g}[v];"
        f"[0:a]{_atempo_chain(rate)}[a]"
    )
    cmd = [
        ffmpeg, "-y",
        "-i", str(video_path),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "[a]",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        "-loglevel", "error",
        "-stats",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def _default_output(video: Path, rate: float) -> Path:
    return video.parent / f"{video.stem}.{rate:g}x.mp4"


@click.command(name="speed")
@click.argument(
    "video", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--rate", "-r",
    type=float,
    required=True,
    help="Playback speed multiplier, e.g. 1.25 or 1.5 (>1 faster, <1 slower).",
)
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output MP4 path. Default: <video-stem>.<rate>x.mp4 next to the input.",
)
@click.option(
    "--crf",
    default=20,
    show_default=True,
    type=int,
    help="x264 CRF (lower = higher quality, larger file).",
)
@click.option(
    "--preset",
    default="medium",
    show_default=True,
    type=click.Choice(
        [
            "ultrafast", "superfast", "veryfast", "faster", "fast",
            "medium", "slow", "slower", "veryslow",
        ]
    ),
    help="x264 encoding preset.",
)
def speed_command(
    video: Path,
    rate: float,
    output: Path | None,
    crf: int,
    preset: str,
) -> None:
    """Change VIDEO's playback speed, producing a new MP4.

    \b
    Examples:
      vcg speed talk.mp4 --rate 1.25
      vcg speed talk.mp4 -r 1.5 -o talk_fast.mp4
    """
    if output is None:
        output = _default_output(video, rate)

    click.echo(f"re-timing {video.name} at {rate:g}x -> {output.name}")
    try:
        change_speed(video, output, rate=rate, crf=crf, preset=preset)
    except subprocess.CalledProcessError as e:
        click.echo(f"ffmpeg failed with exit code {e.returncode}", err=True)
        sys.exit(e.returncode)
    click.echo(f"done: {output}")
