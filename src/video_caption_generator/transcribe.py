"""Transcribe audio using the OpenAI Whisper API.

Subtitle timing strategy
------------------------
Whisper's *segment*-level timestamps cover long chunks (often 10-20s) that may
bundle several sentences, and tend to start before the actual first word and
end well after the last word of speech. That makes captions appear before the
speaker opens their mouth and linger long after they stop.

Instead we request *word*-level timestamps and split into one SRT entry per
sentence:

- a new entry starts at a sentence-ending punctuation (``. ! ? . ! ?``) or
  whenever the silence between two consecutive words exceeds
  :data:`PAUSE_GAP_SECONDS`
- the entry's ``start`` is the first word's start (caption appears with the
  speech, not before)
- the entry's ``end`` is the last word's end plus a small *linger* that scales
  with sentence length (longer sentences get more reading time, capped by
  :data:`MAX_LINGER_SECONDS`) and never bleeds into the next entry
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from openai import OpenAI

from .audio import WAV_BYTES_PER_SECOND, split_audio, wav_duration_seconds


MAX_UPLOAD_BYTES = 24 * 1024 * 1024
"""Upload size at which we switch to chunked transcription. Whisper's hard cap
is 25 MB; we stay under it with a safety margin."""

PAUSE_GAP_SECONDS = 1.5
"""Silence between consecutive words that counts as a sentence boundary."""

MAX_BLOCK_DURATION = 25.0
"""Soft cap on a caption block's on-screen length. Only triggers a sub-split
when Whisper omits sentence punctuation for a long passage; naturally short
blocks (sentences ending in ``.``) are left untouched. The splitter halves
recursively, so a 60-70s passage typically becomes ~4 pieces of 15-17s each."""

MAX_BLOCK_CHARS = 200
"""Soft cap on a caption block's character count, paired with
:data:`MAX_BLOCK_DURATION`."""

BASE_LINGER_SECONDS = 0.4
PER_CHAR_LINGER_SECONDS = 0.04
MAX_LINGER_SECONDS = 1.5
MIN_DISPLAY_SECONDS = 1.0
NEXT_GAP_BUFFER_SECONDS = 0.05

_SENTENCE_END_CHARS = (".", "!", "?", "。", "！", "？")
_SOFT_BREAK_CHARS = (",", ";", ":", "，", "；", "：")


@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str


def transcribe(
    audio_path: Path,
    language: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[Segment]:
    """Transcribe audio and return one :class:`Segment` per sentence.

    The model is configurable via ``OPENAI_TRANSCRIBE_MODEL``
    (default: ``whisper-1``).

    Files above :data:`MAX_UPLOAD_BYTES` (the Whisper 25 MB cap) are split into
    consecutive chunks; each chunk's word timestamps are shifted by the chunk's
    start offset and the chunks are merged back into one timeline before
    sentence splitting, so the result is indistinguishable from a single pass.
    ``progress`` is an optional callback for status messages.
    """
    client = OpenAI()
    model = os.environ.get("OPENAI_TRANSCRIBE_MODEL", "whisper-1")

    if audio_path.stat().st_size <= MAX_UPLOAD_BYTES:
        result = _call_api(client, model, audio_path, language)
        words = list(getattr(result, "words", None) or [])
        if words:
            return _words_to_sentence_segments(words)
        return _from_segment_level([(result, 0.0)])

    return _transcribe_chunked(client, model, audio_path, language, progress)


def _call_api(client: OpenAI, model: str, path: Path, language: str | None):
    with path.open("rb") as f:
        return client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            language=language,
        )


def _transcribe_chunked(
    client: OpenAI,
    model: str,
    audio_path: Path,
    language: str | None,
    progress: Callable[[str], None] | None,
) -> list[Segment]:
    chunk_seconds = max(1, MAX_UPLOAD_BYTES // WAV_BYTES_PER_SECOND)
    with tempfile.TemporaryDirectory() as tmp:
        chunks = split_audio(audio_path, Path(tmp), chunk_seconds)
        if progress:
            progress(f"audio exceeds 25 MB; split into {len(chunks)} chunks")

        all_words: list = []
        results_with_offset: list[tuple[object, float]] = []
        offset = 0.0
        for i, chunk in enumerate(chunks, 1):
            if progress:
                progress(f"transcribing chunk {i}/{len(chunks)}")
            result = _call_api(client, model, chunk, language)
            words = list(getattr(result, "words", None) or [])
            all_words.extend(_offset_words(words, offset))
            results_with_offset.append((result, offset))
            offset += wav_duration_seconds(chunk)

    if all_words:
        return _words_to_sentence_segments(all_words)
    return _from_segment_level(results_with_offset)


def _offset_words(words: list, offset: float) -> list[dict]:
    """Copy words into plain dicts with their timestamps shifted by ``offset``."""
    return [
        {
            "word": _word_text(w),
            "start": _word_start(w) + offset,
            "end": _word_end(w) + offset,
        }
        for w in words
    ]


def _word_field(w, name: str, default):
    if isinstance(w, dict):
        return w.get(name, default)
    return getattr(w, name, default)


def _word_text(w) -> str:
    return str(_word_field(w, "word", "") or "")


def _word_start(w) -> float:
    return float(_word_field(w, "start", 0.0) or 0.0)


def _word_end(w) -> float:
    return float(_word_field(w, "end", 0.0) or 0.0)


_NO_SPACE_BEFORE = set(".,;:!?)]}»”’\"'…")


def _join_word_texts(words: list) -> str:
    """Join word texts with spaces, but keep punctuation attached.

    The Whisper API word objects usually contain just the bare token
    (``"Hello"``) with no surrounding whitespace, so naive ``"".join`` would
    produce ``"Hellothere"``. Some models return ``" Hello"`` with a leading
    space — those cases are also handled correctly here.
    """
    parts: list[str] = []
    for w in words:
        text = _word_text(w)
        if not text:
            continue
        if text.startswith((" ", "\t", "\n")):
            parts.append(text)
        elif not parts:
            parts.append(text)
        elif text[0] in _NO_SPACE_BEFORE:
            parts.append(text)
        else:
            parts.append(" " + text)
    return "".join(parts).strip()


def _best_split_index(sent_words: list) -> int:
    """Pick the most natural mid-block split index (>=1, < len).

    Priority: a word ending with a soft-break char (comma, etc.) closest to
    the middle, otherwise the largest internal word-to-word pause with a
    small midpoint preference for tiebreaking.
    """
    n = len(sent_words)
    mid = n // 2
    for offset in range(0, n // 2 + 1):
        for i in (mid + offset, mid - offset):
            if 1 <= i < n and _word_text(
                sent_words[i - 1]
            ).rstrip().endswith(_SOFT_BREAK_CHARS):
                return i
    best_i, best_score = max(1, mid), -float("inf")
    for i in range(1, n):
        gap = _word_start(sent_words[i]) - _word_end(sent_words[i - 1])
        balance = 1.0 - abs(i - n / 2) / (n / 2)
        score = gap + 0.01 * balance
        if score > best_score:
            best_score, best_i = score, i
    return best_i


def _split_oversized(sent_words: list) -> list[list]:
    """Recursively split a block while it exceeds the duration/char caps."""
    if len(sent_words) < 2:
        return [sent_words]
    duration = _word_end(sent_words[-1]) - _word_start(sent_words[0])
    text_len = len(_join_word_texts(sent_words))
    if duration <= MAX_BLOCK_DURATION and text_len <= MAX_BLOCK_CHARS:
        return [sent_words]
    i = _best_split_index(sent_words)
    return _split_oversized(sent_words[:i]) + _split_oversized(sent_words[i:])


def _words_to_sentence_segments(words: list) -> list[Segment]:
    sentences: list[list] = []
    current: list = []
    prev_end: float | None = None

    for w in words:
        if not _word_text(w).strip():
            continue
        if current and prev_end is not None:
            if _word_start(w) - prev_end >= PAUSE_GAP_SECONDS:
                sentences.append(current)
                current = []
        current.append(w)
        if _word_text(w).rstrip().endswith(_SENTENCE_END_CHARS):
            sentences.append(current)
            current = []
        prev_end = _word_end(w)
    if current:
        sentences.append(current)

    sentences = [piece for s in sentences for piece in _split_oversized(s)]

    segments: list[Segment] = []
    for i, sent_words in enumerate(sentences):
        text = _join_word_texts(sent_words)
        if not text:
            continue
        start = _word_start(sent_words[0])
        speech_end = _word_end(sent_words[-1])

        linger = min(
            MAX_LINGER_SECONDS,
            BASE_LINGER_SECONDS + PER_CHAR_LINGER_SECONDS * len(text),
        )
        desired_end = max(speech_end + linger, start + MIN_DISPLAY_SECONDS)

        if i + 1 < len(sentences):
            next_start = _word_start(sentences[i + 1][0])
            end = min(desired_end, next_start - NEXT_GAP_BUFFER_SECONDS)
            if end < speech_end:
                end = speech_end
        else:
            end = desired_end

        segments.append(
            Segment(index=len(segments) + 1, start=start, end=end, text=text)
        )
    return segments


def _from_segment_level(results_with_offset: list[tuple[object, float]]) -> list[Segment]:
    """Fallback path when no word-level timestamps are available.

    Takes ``(result, offset)`` pairs so chunked transcriptions stitch onto one
    timeline; a single-pass call passes one pair with a zero offset.
    """
    segments: list[Segment] = []
    for result, offset in results_with_offset:
        for s in getattr(result, "segments", []) or []:
            start = float(getattr(s, "start", 0.0)) + offset
            end = float(getattr(s, "end", start)) + offset
            text = str(getattr(s, "text", "")).strip()
            if not text:
                continue
            segments.append(
                Segment(index=len(segments) + 1, start=start, end=end, text=text)
            )
    return segments
