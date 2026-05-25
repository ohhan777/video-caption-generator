"""Download a video from a URL (YouTube etc.) via yt-dlp.

Probes available formats, presents a (resolution + fps) menu, then downloads
the chosen video stream merged with the best available audio into an MP4
using the bundled ffmpeg from ``imageio-ffmpeg``.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import click
import yt_dlp

from .audio import get_ffmpeg


def _ytdlp_ffmpeg_dir() -> str:
    """yt-dlp searches ``ffmpeg_location`` for a file literally named
    ``ffmpeg(.exe)``, but ``imageio-ffmpeg`` ships a versioned filename
    (e.g. ``ffmpeg-win-x86_64-v7.1.exe``). Materialize a stable hardlink
    (or copy fallback) under the temp dir and return that directory."""
    src = Path(get_ffmpeg())
    target_name = "ffmpeg" + src.suffix
    if src.name == target_name:
        return str(src.parent)
    cache_dir = Path(tempfile.gettempdir()) / "vcg-ffmpeg"
    cache_dir.mkdir(exist_ok=True)
    cached = cache_dir / target_name
    if not cached.exists() or cached.stat().st_size != src.stat().st_size:
        if cached.exists():
            cached.unlink()
        try:
            os.link(src, cached)
        except OSError:
            shutil.copy2(src, cached)
    return str(cache_dir)


_RES_LABELS = {
    4320: "8K",
    2160: "4K",
    1440: "2K",
    1080: "FHD",
    720: "HD",
    480: "SD",
}


def _res_label(height: int) -> str:
    return _RES_LABELS.get(height, f"{height}p")


def _fmt_filesize(fmt: dict) -> str:
    size = fmt.get("filesize") or fmt.get("filesize_approx")
    if not size:
        return "?"
    mb = size / (1024 * 1024)
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.1f} MB"


def _probe_formats(url: str) -> tuple[str, list[dict]]:
    """Return ``(title, choices)`` where ``choices`` is the best video format
    per (height, rounded-fps) combo, sorted from highest to lowest quality."""
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(url, download=False)

    title = info.get("title") or "video"
    formats = info.get("formats") or []

    video_only = [
        f for f in formats
        if f.get("vcodec") and f.get("vcodec") != "none" and f.get("height")
    ]

    best: dict[tuple[int, int], dict] = {}
    for f in video_only:
        h = int(f["height"])
        fps = int(round(f.get("fps") or 0))
        key = (h, fps)
        cur = best.get(key)
        if cur is None or (f.get("tbr") or 0) > (cur.get("tbr") or 0):
            best[key] = f

    choices = sorted(
        best.values(),
        key=lambda f: (int(f["height"]), int(round(f.get("fps") or 0))),
        reverse=True,
    )
    return title, choices


def _resolve_outtmpl(output: Path | None) -> str:
    """Convert the user's ``--output`` hint into a yt-dlp ``outtmpl`` string."""
    if output is None:
        return str(Path.cwd() / "%(title)s.%(ext)s")
    if output.suffix:
        output.parent.mkdir(parents=True, exist_ok=True)
        return str(output.with_suffix("")) + ".%(ext)s"
    output.mkdir(parents=True, exist_ok=True)
    return str(output / "%(title)s.%(ext)s")


def _final_path(info: dict, ydl: yt_dlp.YoutubeDL) -> Path:
    """Locate the merged MP4 path after a download completes."""
    requested = info.get("requested_downloads") or []
    if requested:
        fp = requested[0].get("filepath")
        if fp:
            return Path(fp)
    guess = Path(ydl.prepare_filename(info))
    if guess.suffix.lower() != ".mp4":
        mp4 = guess.with_suffix(".mp4")
        if mp4.exists():
            return mp4
    return guess


def download_video(url: str, output: Path | None = None) -> Path:
    click.echo(f"probing {url}")
    title, choices = _probe_formats(url)
    if not choices:
        raise click.ClickException("No video formats found for this URL.")

    click.echo("")
    click.echo(f"title: {title}")
    click.echo("")
    click.echo("Available resolution + fps:")
    for i, f in enumerate(choices, 1):
        h = int(f["height"])
        fps = int(round(f.get("fps") or 0)) or "?"
        codec = (f.get("vcodec") or "?").split(".")[0]
        click.echo(
            f"  {i}) {_res_label(h)} {h}p @ {fps}fps  "
            f"(codec={codec}, ~{_fmt_filesize(f)})"
        )

    pick = click.prompt(
        "\nSelect", type=click.IntRange(1, len(choices))
    )
    chosen = choices[pick - 1]

    ffmpeg_dir = _ytdlp_ffmpeg_dir()
    ydl_opts = {
        "format": f"{chosen['format_id']}+bestaudio/best",
        "merge_output_format": "mp4",
        "ffmpeg_location": ffmpeg_dir,
        "outtmpl": _resolve_outtmpl(output),
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = _final_path(info, ydl)

    click.echo(f"done: {path}")
    return path


@click.command(name="download")
@click.argument("url")
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path. Either a file (e.g. clip.mp4) or a directory. "
    "Default: current directory, filename derived from the video title.",
)
def download_command(url: str, output: Path | None) -> None:
    """Download a video from URL (YouTube etc.), choosing resolution + fps.

    \b
    Examples:
      vcg download https://youtu.be/xxxxxxx
      vcg download https://youtu.be/xxxxxxx -o downloads/
      vcg download https://youtu.be/xxxxxxx -o myclip.mp4
    """
    try:
        download_video(url, output=output)
    except yt_dlp.utils.DownloadError as e:
        click.echo(f"download failed: {e}", err=True)
        sys.exit(1)
