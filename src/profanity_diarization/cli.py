"""Typer CLI: process, batch, prefetch-models, report."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import typer

from .audio import AudioError
from .config import load_settings
from .pipeline import Pipeline

app = typer.Typer(add_completion=False, help="Profanity detection + speaker diarization.")
MEDIA = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def load_dotenv(path: Path = Path(".env")) -> None:
    """KEY=VALUE lines from ./.env into os.environ (existing variables win). No dependency."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        k, sep, v = line.strip().partition("=")
        if sep and k and not k.startswith("#"):
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def _setup(preset, device, offline, keep, verbose=True):
    load_dotenv()
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")  # fmt: skip
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return load_settings(preset, device=device, offline=offline or None,
                         keep_intermediate=keep or None)  # fmt: skip


def _run(pipe: Pipeline, f: Path, out: Path) -> None:
    r = pipe.process(f, out)
    paths = pipe.write_outputs(r, out, f.stem)
    typer.echo(f"{f.name}: {r.counts} -> {paths[0]}")


@app.command()
def process(
    file: Path,
    preset: str = "paper",
    output_dir: Path = Path("output"),
    device: Optional[str] = typer.Option(None, help="auto|cpu|cuda"),
    offline: bool = False,
    keep_intermediate: bool = False,
):
    """Process one video/audio file."""
    cfg = _setup(preset, device, offline, keep_intermediate)
    try:
        _run(Pipeline(cfg), file, output_dir)
    except (AudioError, RuntimeError, FileNotFoundError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def batch(
    directory: Path,
    preset: str = "paper",
    output_dir: Path = Path("output"),
    device: Optional[str] = None,
    offline: bool = False,
    keep_intermediate: bool = False,
):
    """Process every media file in a directory (models are loaded once)."""
    cfg = _setup(preset, device, offline, keep_intermediate)
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in MEDIA)
    if not files:
        typer.echo(f"error: no media files in {directory}", err=True)
        raise typer.Exit(1)
    pipe, failed = Pipeline(cfg), 0
    for f in files:
        try:
            _run(pipe, f, output_dir)
        except (AudioError, RuntimeError) as e:
            failed += 1
            typer.echo(f"error: {f.name}: {e}", err=True)
    raise typer.Exit(1 if failed else 0)


@app.command("prefetch-models")
def prefetch(preset: str = "paper"):
    """One-time online download of Whisper, pyannote and the WordNet corpus."""
    cfg = _setup(preset, None, False, False)
    import nltk
    import whisper

    from .diarize import PyannoteDiarizer

    try:
        nltk.download("wordnet")
        whisper.load_model(cfg.whisper_model, device="cpu")
        PyannoteDiarizer(cfg.diarization_model, "cpu")._load()
    except RuntimeError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"whisper cache: {os.path.expanduser('~/.cache/whisper')}")
    typer.echo(f"HF cache: {os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))}")


@app.command()
def report(json_file: Path, output_dir: Optional[Path] = None):
    """Re-render CSV/SRT/Markdown from a saved JSON report."""
    from .config import Settings
    from .profanity import build_detector
    from .schemas import Report

    r = Report.model_validate_json(json_file.read_text(encoding="utf-8"))
    cfg = Settings(**r.config)
    det = build_detector(cfg)
    pipe = Pipeline(cfg, transcriber=object(), diarizer=object(), detector=det)
    for p in pipe.write_outputs(r, output_dir or json_file.parent, json_file.stem):
        typer.echo(p)


if __name__ == "__main__":
    app()
