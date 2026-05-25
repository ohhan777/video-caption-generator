"""Burn subtitles into a video using ffmpeg's ``subtitles`` filter."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .audio import get_ffmpeg


def _ffmpeg_subtitle_path(path: Path) -> str:
    """Format an absolute path for the ffmpeg ``subtitles`` filter on Windows.

    The filter parses its argument with a shell-like syntax, so the drive colon
    (``C:``) must be escaped and backslashes are converted to forward slashes.
    """
    s = str(path.resolve()).replace("\\", "/")
    if len(s) > 1 and s[1] == ":":
        s = s[0] + "\\:" + s[2:]
    return s


def burn_subtitles(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    *,
    font_name: str = "Malgun Gothic",
    font_size: int = 22,
    margin_v: int = 10,
) -> Path:
    """Burn an SRT into the video and re-encode video while copying audio.

    ``margin_v`` is the bottom margin in the libass coordinate system: smaller
    values push the caption closer to the bottom edge.
    """
    ffmpeg = get_ffmpeg()
    sub_arg = _ffmpeg_subtitle_path(srt_path)
    style = (
        f"FontName={font_name},FontSize={font_size},"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,"
        f"BorderStyle=1,Outline=2,Shadow=0,MarginV={margin_v}"
    )
    vf = f"subtitles='{sub_arg}':force_style='{style}'"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i", str(video_path),
            "-vf", vf,
            "-c:a", "copy",
            "-loglevel", "error",
            "-stats",
            str(output_path),
        ],
        check=True,
    )
    return output_path
