"""Translate transcript segments to Korean using an OpenAI chat model."""
from __future__ import annotations

import json
import os
import re
from typing import Sequence

from openai import OpenAI

from .transcribe import Segment


MAX_KOREAN_SENTENCES_PER_BLOCK = 2
"""Cap on sentences per Korean SRT entry. When a translation exceeds this,
:func:`cap_korean_sentences` splits the entry's time range proportionally
by character count so a single English segment can map to several shorter
Korean entries (e.g. when Whisper failed to punctuate a long passage)."""


SYSTEM_PROMPT = (
    "You are a professional subtitle translator. Translate each English segment "
    "into natural, fluent Korean.\n\n"
    "Rules:\n"
    "- Preserve segment indices exactly.\n"
    "- Translate each segment with the surrounding context in mind, but keep "
    "  one Korean line per input segment.\n"
    "- Keep meaning and tone faithful; prefer concise spoken Korean for subtitles.\n"
    "- Do not include the English text in your output.\n"
    "- Output strictly as JSON: "
    '{"translations": [{"index": 1, "ko": "..."}, ...]}'
)


def _extract_json(text: str) -> dict:
    """Extract a JSON object from a model response, tolerating ```json fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def translate_segments(
    segments: Sequence[Segment],
    *,
    batch_size: int = 30,
) -> dict[int, str]:
    """Translate segments to Korean. Returns ``{index: korean_text}``.

    The model is configurable via OPENAI_TRANSLATE_MODEL.
    """
    client = OpenAI()
    model = os.environ.get("OPENAI_TRANSLATE_MODEL", "gpt-4o")

    translations: dict[int, str] = {}

    for i in range(0, len(segments), batch_size):
        batch = segments[i : i + batch_size]
        user_payload = {
            "segments": [{"index": s.index, "en": s.text} for s in batch],
        }

        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            temperature=0.2,
        )

        try:
            response = client.chat.completions.create(
                response_format={"type": "json_object"},
                **kwargs,
            )
        except Exception:
            response = client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or ""
        data = _extract_json(content)
        for item in data.get("translations", []):
            try:
                idx = int(item["index"])
                translations[idx] = str(item["ko"]).strip()
            except (KeyError, ValueError, TypeError):
                continue

    return translations


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


def _split_into_sentences(text: str) -> list[str]:
    if not text:
        return []
    return [p.strip() for p in _SENTENCE_SPLIT_RE.split(text.strip()) if p.strip()]


def _group_sentences(sentences: list[str], max_per_group: int) -> list[str]:
    return [
        " ".join(sentences[i : i + max_per_group])
        for i in range(0, len(sentences), max_per_group)
    ]


def cap_korean_sentences(
    segments: Sequence[Segment],
    translations: dict[int, str],
    *,
    max_sentences: int = MAX_KOREAN_SENTENCES_PER_BLOCK,
) -> tuple[list[Segment], dict[int, str]]:
    """Split Korean translations exceeding ``max_sentences`` into sub-blocks.

    Returns a new ``(segments, translations)`` pair with re-numbered indices.
    For a segment whose Korean translation has more than ``max_sentences``
    sentences, the segment's [start, end] is divided proportionally by the
    character count of each Korean sub-group, producing several shorter SRT
    entries. The English ``text`` is kept on the first sub-entry only (it is
    only used by the English SRT, which writes the original segments).
    """
    new_segments: list[Segment] = []
    new_translations: dict[int, str] = {}
    next_index = 1

    for seg in segments:
        ko = translations.get(seg.index, "")
        sentences = _split_into_sentences(ko)
        if len(sentences) <= max_sentences:
            new_segments.append(
                Segment(
                    index=next_index,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                )
            )
            if ko:
                new_translations[next_index] = ko
            next_index += 1
            continue

        groups = _group_sentences(sentences, max_sentences)
        total_chars = sum(len(g) for g in groups) or 1
        duration = seg.end - seg.start
        cursor = seg.start
        for j, group in enumerate(groups):
            if j == len(groups) - 1:
                end = seg.end
            else:
                end = cursor + duration * (len(group) / total_chars)
            new_segments.append(
                Segment(
                    index=next_index,
                    start=cursor,
                    end=end,
                    text=seg.text if j == 0 else "",
                )
            )
            new_translations[next_index] = group
            cursor = end
            next_index += 1

    return new_segments, new_translations
