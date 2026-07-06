"""Text handling for ARTUR `std` transcriptions.

Two distinct jobs, two functions:

- `clean_std` prepares *training/reference* text: keeps case and punctuation
  (Nemotron's recipe wants punctuated, properly-cased targets; Whisper copes
  either way), removes transcription markup that is not speech:
    * filler tokens spelled with a leading '#' (`#Eem`, `#eee`, ...)
    * cut-off markers `word()` — the marker is dropped, the partial word kept
      (test-split utterances carrying cut-offs are dropped upstream instead)
- `normalize_eval` maps hypothesis/reference to the space where WER_norm/CER_norm
  are computed: lowercase, punctuation stripped, whitespace collapsed.
  Slovenian diacritics (č š ž) are letters and are NEVER folded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FILLER_RE = re.compile(r"(?<!\S)#[^\s]+")
_CUTOFF_RE = re.compile(r"\(\)")
_WS_RE = re.compile(r"\s+")

# characters treated as punctuation for the normalized metric space
_EVAL_PUNCT_RE = re.compile(r"[.,!?…;:\"'`’‘„“”«»()\[\]{}—–\-­/\\+*=<>|~^%$§#&@]")

# tokens that still look like markup after cleaning — reported, not silently kept
_SUSPECT_RE = re.compile(r"[\[\]{}@*_/\\|<>~^=]")


@dataclass
class CleanResult:
    text: str
    had_filler: bool
    had_cutoff: bool
    suspect_tokens: tuple[str, ...]


def clean_std(text: str) -> CleanResult:
    had_filler = bool(_FILLER_RE.search(text))
    had_cutoff = "()" in text
    t = _FILLER_RE.sub(" ", text)
    t = _CUTOFF_RE.sub("", t)
    t = _WS_RE.sub(" ", t).strip()
    suspects = tuple(tok for tok in t.split() if _SUSPECT_RE.search(tok))
    return CleanResult(text=t, had_filler=had_filler, had_cutoff=had_cutoff, suspect_tokens=suspects)


def normalize_eval(text: str) -> str:
    t = text.replace(" ", " ").lower()
    t = _EVAL_PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def count_digit_tokens(text: str) -> int:
    """ARTUR spells numerals as words; digit tokens in hypotheses inflate WER.
    Counted and reported so the caveat is quantified, not hand-waved."""
    return sum(1 for tok in text.split() if any(ch.isdigit() for ch in tok))
