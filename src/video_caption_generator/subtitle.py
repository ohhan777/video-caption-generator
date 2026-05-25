"""SRT subtitle file I/O."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .transcribe import Segment


def _format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(
    segments: Iterable[Segment],
    overrides: dict[int, str] | None,
    path: Path,
) -> None:
    """Write segments to an SRT file.

    If ``overrides`` is given, the override text replaces the segment text
    (used to write the Korean translation while keeping original timestamps).
    """
    lines: list[str] = []
    for s in segments:
        text = overrides.get(s.index, s.text) if overrides else s.text
        lines.append(str(s.index))
        lines.append(f"{_format_timestamp(s.start)} --> {_format_timestamp(s.end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


_TIMESTAMP_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def _parse_timestamp(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def read_srt(path: Path) -> list[Segment]:
    """Parse a UTF-8 SRT file into ``Segment`` objects."""
    text = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", text.strip())
    segments: list[Segment] = []
    for block in blocks:
        block_lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(block_lines) < 2:
            continue
        try:
            idx = int(block_lines[0].strip())
        except ValueError:
            continue
        m = _TIMESTAMP_RE.search(block_lines[1])
        if not m:
            continue
        start = _parse_timestamp(*m.group(1, 2, 3, 4))
        end = _parse_timestamp(*m.group(5, 6, 7, 8))
        body = "\n".join(block_lines[2:])
        segments.append(Segment(index=idx, start=start, end=end, text=body))
    return segments
