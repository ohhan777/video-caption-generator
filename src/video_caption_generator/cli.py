"""CLI entry point for the video caption generator (``vcg``)."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import click
from dotenv import load_dotenv

from .audio import extract_audio
from .burn import burn_subtitles
from .download import download_command
from .merge import merge_command
from .subtitle import read_srt, write_srt
from .transcribe import Segment, transcribe
from .translate import cap_korean_sentences, translate_segments
from .trim import trim_command


def _default_paths(video: Path) -> tuple[Path, Path, Path]:
    stem = video.with_suffix("")
    return (
        Path(f"{stem}.en.srt"),
        Path(f"{stem}.ko.srt"),
        Path(f"{stem}.captioned.mp4"),
    )


@click.group()
def cli() -> None:
    """Transcribe a video with Whisper, translate to Korean, then burn the
    subtitles into the video. Workflow: ``generate`` -> edit SRT -> ``burn``."""
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        click.echo("warning: OPENAI_API_KEY is not set", err=True)


@cli.command()
@click.argument(
    "video", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--language",
    default=None,
    help="Source language hint for Whisper, e.g. 'en'.",
)
def generate(video: Path, language: str | None) -> None:
    """Transcribe VIDEO and write <stem>.en.srt and <stem>.ko.srt for review."""
    en_srt, ko_srt, _ = _default_paths(video)

    click.echo(f"[1/3] extracting audio from {video.name}")
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "audio.wav"
        extract_audio(video, audio)

        click.echo("[2/3] transcribing via Whisper API")
        segments = transcribe(
            audio,
            language=language,
            progress=lambda m: click.echo(f"      {m}"),
        )

    click.echo(f"      got {len(segments)} segments")
    write_srt(segments, None, en_srt)
    click.echo(f"      wrote {en_srt.name}")

    if not segments:
        click.echo("no segments returned; aborting before translation", err=True)
        sys.exit(1)

    click.echo("[3/3] translating to Korean")
    translations = translate_segments(segments)
    missing = [s.index for s in segments if s.index not in translations]
    if missing:
        click.echo(
            f"warning: {len(missing)} segment(s) missing a translation "
            f"(first: {missing[:5]}); their English text will be used.",
            err=True,
        )
    ko_segments, ko_translations = cap_korean_sentences(segments, translations)
    write_srt(ko_segments, ko_translations, ko_srt)
    click.echo(f"      wrote {ko_srt.name}")

    click.echo("")
    click.echo("Next steps:")
    click.echo(
        f"  1) Review side-by-side:  vcg review \"{en_srt}\" \"{ko_srt}\""
    )
    click.echo(
        f"  2) Edit the Korean SRT:   {ko_srt}  (any text editor; keep timestamps)"
    )
    click.echo(
        f"  3) Burn into video:       vcg burn \"{video}\" \"{ko_srt}\""
    )


@cli.command()
@click.argument(
    "en_srt", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Korean SRT output path (default: replace '.en.srt' with '.ko.srt').",
)
def translate(en_srt: Path, output: Path | None) -> None:
    """Re-translate an existing English SRT into Korean.

    Useful when you already have an EN.srt from a previous ``generate`` run
    (no need to re-pay for Whisper) or want to try a different model.
    """
    segments = read_srt(en_srt)
    if not segments:
        click.echo(f"no segments parsed from {en_srt}", err=True)
        sys.exit(1)

    if output is None:
        name = en_srt.name
        if name.endswith(".en.srt"):
            output = en_srt.with_name(name[: -len(".en.srt")] + ".ko.srt")
        else:
            output = en_srt.with_suffix(".ko.srt")

    click.echo(f"translating {len(segments)} segments -> Korean")
    translations = translate_segments(segments)
    missing = [s.index for s in segments if s.index not in translations]
    if missing:
        click.echo(
            f"warning: {len(missing)} segment(s) missing a translation "
            f"(first: {missing[:5]}); their English text will be used.",
            err=True,
        )
    ko_segments, ko_translations = cap_korean_sentences(segments, translations)
    write_srt(ko_segments, ko_translations, output)
    click.echo(f"wrote {output}")


@cli.command()
@click.argument(
    "en_srt", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "ko_srt", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
def review(en_srt: Path, ko_srt: Path) -> None:
    """Print a side-by-side review of two SRT files.

    The Korean SRT is re-numbered by sentence splitting, so its indices no
    longer line up with the English SRT. Each Korean entry is therefore paired
    with the English entry it overlaps most in time, and a single English entry
    may show several Korean lines.
    """
    en = read_srt(en_srt)
    ko = read_srt(ko_srt)

    def _best_en(k: Segment) -> Segment | None:
        best, best_overlap = None, 0.0
        for e in en:
            overlap = min(k.end, e.end) - max(k.start, e.start)
            if overlap > best_overlap:
                best, best_overlap = e, overlap
        return best

    ko_by_en: dict[int, list[Segment]] = {e.index: [] for e in en}
    unmatched: list[Segment] = []
    for k in ko:
        match = _best_en(k)
        if match is None:
            unmatched.append(k)
        else:
            ko_by_en[match.index].append(k)

    for e in en:
        click.echo(f"#{e.index}  [{e.start:7.2f} -> {e.end:7.2f}]")
        click.echo(f"  EN: {e.text}")
        matches = ko_by_en[e.index]
        if matches:
            for k in matches:
                click.echo(f"  KO: {k.text}")
        else:
            click.echo("  KO: (missing)")
        click.echo("")

    if unmatched:
        click.echo("Korean entries with no overlapping English entry:")
        for k in unmatched:
            click.echo(f"  [{k.start:7.2f} -> {k.end:7.2f}] {k.text}")


@cli.command()
@click.argument(
    "video", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "srt", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output MP4 path (default: <video>.captioned.mp4).",
)
@click.option(
    "--font-name",
    default="Malgun Gothic",
    help="Font family that supports Korean (default: Malgun Gothic on Windows).",
)
@click.option("--font-size", default=22, type=int, show_default=True)
@click.option(
    "--margin-v",
    default=10,
    type=int,
    show_default=True,
    help="Bottom margin in libass units; lower values push captions closer to "
    "the bottom edge.",
)
@click.confirmation_option(
    prompt="Burn-in is non-reversible. The Korean SRT will be hardcoded into a "
    "new MP4. Continue?"
)
def burn(
    video: Path,
    srt: Path,
    output: Path | None,
    font_name: str,
    font_size: int,
    margin_v: int,
) -> None:
    """Burn SRT into VIDEO, producing a new MP4."""
    if output is None:
        _, _, output = _default_paths(video)
    click.echo(f"burning {srt.name} into {video.name} -> {output.name}")
    burn_subtitles(
        video,
        srt,
        output,
        font_name=font_name,
        font_size=font_size,
        margin_v=margin_v,
    )
    click.echo(f"done: {output}")


cli.add_command(trim_command)
cli.add_command(download_command)
cli.add_command(merge_command)


if __name__ == "__main__":
    cli()
