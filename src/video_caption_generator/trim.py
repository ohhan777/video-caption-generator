"""Trim a video to a [start, end] range using ffmpeg.

Re-encodes the video for frame-accurate cuts.

``-ss`` is placed **before** ``-i`` so ffmpeg can fast-seek through the
container index to the nearest preceding keyframe rather than decoding from
the file's beginning. ``-accurate_seek`` (on by default) then decodes the
short span from that keyframe forward to the requested start before
re-encoding begins, preserving frame accuracy. With this layout the decoder's
timestamps are rebased to zero at the seek point, so the end of the range is
expressed as a duration via ``-t`` rather than as an absolute ``-to``.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from .audio import get_ffmpeg


def _parse_timestamp(ts: str) -> float:
    """Parse ``HH:MM:SS[.ms]``, ``MM:SS[.ms]``, or seconds into a float."""
    parts = ts.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])


def trim_video(
    video_path: Path,
    output_path: Path,
    *,
    start: str | None = None,
    end: str | None = None,
    crf: int = 20,
    preset: str = "medium",
) -> Path:
    """Trim ``video_path`` to ``[start, end]`` and write to ``output_path``."""
    ffmpeg = get_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [ffmpeg, "-y"]
    if start is not None:
        cmd += ["-ss", start]
    cmd += ["-i", str(video_path)]
    if end is not None:
        if start is not None:
            duration = _parse_timestamp(end) - _parse_timestamp(start)
            if duration <= 0:
                raise click.BadParameter(
                    "--end must be after --start", param_hint="--end"
                )
            cmd += ["-t", f"{duration:.3f}"]
        else:
            cmd += ["-to", end]
    cmd += [
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


def _default_output(video: Path) -> Path:
    return video.parent / f"{video.stem}.trimmed.mp4"


@click.command(name="trim")
@click.argument(
    "video", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--start",
    default=None,
    help="Start timestamp HH:MM:SS[.ms]. Default: beginning of video.",
)
@click.option(
    "--end",
    default=None,
    help="End timestamp HH:MM:SS[.ms]. Default: end of video.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output MP4 path. Default: <video-stem>.trimmed.mp4 next to the input.",
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
def trim_command(
    video: Path,
    start: str | None,
    end: str | None,
    output: Path | None,
    crf: int,
    preset: str,
) -> None:
    """Trim VIDEO to a [start, end] range, producing a new MP4.

    \b
    Examples:
      vcg trim input.mp4 --start 00:05:01.23 --end 00:05:09.23
      vcg trim input.mp4 --end 00:01:00 -o intro.mp4
      vcg trim input.mp4 --start 00:10:00
    """
    if start is None and end is None:
        click.echo(
            "warning: neither --start nor --end given; re-encoding the full "
            "video.",
            err=True,
        )
    if output is None:
        output = _default_output(video)

    click.echo(
        f"trimming {video.name} -> {output.name}  "
        f"(start={start or 'begin'}, end={end or 'end'})"
    )
    try:
        trim_video(video, output, start=start, end=end, crf=crf, preset=preset)
    except subprocess.CalledProcessError as e:
        click.echo(f"ffmpeg failed with exit code {e.returncode}", err=True)
        sys.exit(e.returncode)
    click.echo(f"done: {output}")
