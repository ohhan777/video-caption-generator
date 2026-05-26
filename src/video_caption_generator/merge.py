"""Merge several videos into one, end to end, using ffmpeg.

Two strategies:

* ``--copy`` uses the concat *demuxer* with stream copy. No re-encoding, so it
  is near-instant, but every input must share the same codec, resolution, and
  frame rate or playback breaks.
* The default uses the concat *filter*: each input is scaled (letterboxed) to
  the first video's resolution, its frame rate and audio format normalized,
  then re-encoded. Slower, but works across mismatched sources.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import click

from .audio import get_ffmpeg, has_audio_stream, media_duration_seconds


def _probe_resolution_fps(video_path: Path) -> tuple[int, int, float]:
    """Read the first video stream's width, height, and fps from ffmpeg output.

    imageio-ffmpeg ships ffmpeg but not ffprobe, so the dimensions are parsed
    from ffmpeg's stderr banner. Falls back to 1920x1080 @ 30 if unparseable.
    """
    result = subprocess.run(
        [get_ffmpeg(), "-i", str(video_path)],
        capture_output=True,
        text=True,
    )
    info = result.stderr
    width, height, fps = 1920, 1080, 30.0
    dims = re.search(r"Video:.*?, (\d{2,5})x(\d{2,5})", info)
    if dims:
        width, height = int(dims.group(1)), int(dims.group(2))
    fps_match = re.search(r"(\d+(?:\.\d+)?)\s+fps", info)
    if fps_match:
        fps = float(fps_match.group(1))
    return width, height, fps


def _merge_copy(inputs: list[Path], output: Path) -> None:
    ffmpeg = get_ffmpeg()
    listfile = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            for p in inputs:
                # concat demuxer wants forward slashes and escaped single quotes
                safe = p.resolve().as_posix().replace("'", "'\\''")
                f.write(f"file '{safe}'\n")
            listfile = f.name
        subprocess.run(
            [
                ffmpeg, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", listfile,
                "-c", "copy",
                "-movflags", "+faststart",
                "-loglevel", "error",
                "-stats",
                str(output),
            ],
            check=True,
        )
    finally:
        if listfile:
            Path(listfile).unlink(missing_ok=True)


def _merge_reencode(
    inputs: list[Path], output: Path, *, crf: int, preset: str
) -> None:
    ffmpeg = get_ffmpeg()
    width, height, fps = _probe_resolution_fps(inputs[0])
    n = len(inputs)
    has_audio = [has_audio_stream(p) for p in inputs]

    cmd: list[str] = [ffmpeg, "-y"]
    for p in inputs:
        cmd += ["-i", str(p)]

    # The concat filter needs every segment to carry an audio stream. For any
    # silent input, feed an anullsrc track sized to that clip's duration; its
    # ffmpeg input index follows the video inputs.
    silence_input: dict[int, int] = {}
    next_input = n
    for i, p in enumerate(inputs):
        if not has_audio[i]:
            cmd += [
                "-f", "lavfi",
                "-t", f"{media_duration_seconds(p):.3f}",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            ]
            silence_input[i] = next_input
            next_input += 1

    steps: list[str] = []
    for i in range(n):
        steps.append(
            f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}[v{i}];"
        )
        audio_src = i if has_audio[i] else silence_input[i]
        steps.append(
            f"[{audio_src}:a]aformat=sample_rates=48000:channel_layouts=stereo[a{i}];"
        )
    pairs = "".join(f"[v{i}][a{i}]" for i in range(n))
    filter_complex = "".join(steps) + f"{pairs}concat=n={n}:v=1:a=1[outv][outa]"

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        "-loglevel", "error",
        "-stats",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def merge_videos(
    inputs: list[Path],
    output: Path,
    *,
    copy: bool = False,
    crf: int = 20,
    preset: str = "medium",
) -> Path:
    """Concatenate ``inputs`` (in order) into ``output``."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        _merge_copy(inputs, output)
    else:
        _merge_reencode(inputs, output, crf=crf, preset=preset)
    return output


@click.command(name="merge")
@click.argument(
    "videos",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("merged.mp4"),
    show_default=True,
    help="Output MP4 path.",
)
@click.option(
    "--copy",
    is_flag=True,
    default=False,
    help="Stream-copy without re-encoding (fast). Requires all inputs to share "
    "codec, resolution, and frame rate.",
)
@click.option(
    "--crf",
    default=20,
    show_default=True,
    type=int,
    help="x264 CRF for re-encode mode (lower = higher quality, larger file).",
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
    help="x264 encoding preset for re-encode mode.",
)
def merge_command(
    videos: tuple[Path, ...],
    output: Path,
    copy: bool,
    crf: int,
    preset: str,
) -> None:
    """Merge VIDEOS into one MP4, end to end, in the order given.

    \b
    Examples:
      vcg merge intro.mp4 main.mp4 outro.mp4 -o full.mp4
      vcg merge part1.mp4 part2.mp4 --copy -o joined.mp4
    """
    inputs = list(videos)
    if len(inputs) < 2:
        raise click.BadParameter("provide at least two videos to merge.")

    mode = "copy (no re-encode)" if copy else "re-encode"
    click.echo(f"merging {len(inputs)} videos -> {output.name}  ({mode})")
    for i, p in enumerate(inputs, 1):
        click.echo(f"  {i}) {p.name}")
    try:
        merge_videos(inputs, output, copy=copy, crf=crf, preset=preset)
    except subprocess.CalledProcessError as e:
        click.echo(f"ffmpeg failed with exit code {e.returncode}", err=True)
        if copy:
            click.echo(
                "the --copy fast path needs identical codecs/resolution; "
                "try again without --copy to re-encode and normalize.",
                err=True,
            )
        sys.exit(e.returncode)
    click.echo(f"done: {output}")
