"""Lexicon, normalizer and detector.

Paper rule (final_gpu.py:94-104): lower -> split() -> strip ALL punctuation ->
lemmatize(pos="v") -> exact set membership.
"""

from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Callable, Optional

from .schemas import ProfanityHit, Segment

LEXICON_DIR = Path(__file__).parent / "lexicon"
_PUNCT = str.maketrans("", "", string.punctuation)


def load_lexicon(spec: Optional[str] = None) -> frozenset:
    """``None`` -> bundled lexicon_b; a bare name -> bundled file; else a path."""
    spec = spec or "lexicon_b"
    p = Path(spec)
    if not p.exists():
        p = LEXICON_DIR / f"{spec}.txt"
    if not p.exists():
        raise FileNotFoundError(f"lexicon not found: {spec}")
    lines = (ln.strip().lower() for ln in p.read_text(encoding="utf-8").splitlines())
    return frozenset(ln for ln in lines if ln and not ln.startswith("#"))


def verb_lemmatizer() -> Callable[[str], str]:
    """WordNet verb lemmatizer as in legacy. Needs the wordnet corpus (``prefetch-models``)."""
    from nltk.stem import WordNetLemmatizer

    wn = WordNetLemmatizer()
    return lambda w: wn.lemmatize(w, pos="v")


def normalize(token: str, lemma: Callable[[str], str] = lambda w: w) -> str:
    return lemma(token.lower().translate(_PUNCT))


class Detector:
    def __init__(self, lexicon: frozenset, lemma: Callable[[str], str] = lambda w: w):
        self.lexicon, self.lemma = lexicon, lemma

    def match(self, token: str) -> Optional[str]:
        n = normalize(token, self.lemma)
        return n if n in self.lexicon else None

    def detect(self, segments: list[Segment]) -> list[ProfanityHit]:
        hits = []
        for s in segments:
            for tok in s.text.lower().split():
                term = self.match(tok)
                if term:
                    hits.append(
                        ProfanityHit(term=term, token=tok, speaker=s.speaker, group=s.group,
                                     start=s.start, end=s.end, context=s.text)  # fmt: skip
                    )
        return hits

    def mask(self, text: str) -> str:
        """Mask matching tokens (``f***``) for docs, Markdown and SRT."""

        def sub(m: re.Match) -> str:
            tok = m.group(0)
            return tok[0] + "***" if self.match(tok) else tok

        return re.sub(r"\S+", sub, text)


def counts(hits: list[ProfanityHit], groups: list[str]) -> tuple:
    tot = {g: 0 for g in groups}
    uniq = {g: set() for g in groups}
    for h in hits:
        tot[h.group] = tot.get(h.group, 0) + 1
        uniq.setdefault(h.group, set()).add(h.term)
    return tot, {g: len(v) for g, v in uniq.items()}
