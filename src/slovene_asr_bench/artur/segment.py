"""Merge TRS sync sub-segments into ASR training utterances.

Whisper's encoder window is 30 s, so utterances are capped at `max_dur`
(default 25 s, margin included); NeMo's lhotse loader wants >= 0.1 s, and very
short clips invite Whisper hallucinations, so anything under `min_dur`
(default 0.5 s) is dropped. Consecutive spans of the same single-speaker turn
are contiguous in time and are merged greedily up to the cap.
"""

from __future__ import annotations

from dataclasses import dataclass

from .textnorm import clean_std
from .trs import Turn


@dataclass
class Utterance:
    utt_id: str
    recording_id: str
    speaker_id: str
    subcorpus: str  # B / J / N / P
    t0: float
    t1: float
    text: str
    has_event: bool
    has_cutoff: bool
    has_filler: bool
    suspect_tokens: tuple[str, ...] = ()

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


def subcorpus_of(recording_id: str) -> str:
    # Artur-J-Gvecg-P580017 -> "J"
    parts = recording_id.split("-")
    if len(parts) < 2 or parts[0] != "Artur":
        raise ValueError(f"unexpected recording id {recording_id!r}")
    return parts[1]


def build_utterances(
    recording_id: str,
    turns: list[Turn],
    min_dur: float = 0.5,
    max_dur: float = 25.0,
) -> tuple[list[Utterance], dict]:
    """Returns (utterances, stats). Drops pause turns, overlap turns, and spans
    that alone exceed max_dur (no word alignment available to split them)."""
    sub = subcorpus_of(recording_id)
    utts: list[Utterance] = []
    stats = {
        "turns_total": len(turns),
        "turns_no_speaker": 0,
        "turns_overlap": 0,
        "spans_too_long": 0,
        "utts_too_short": 0,
    }

    counter = 0
    for turn in turns:
        if turn.speaker is None and not turn.multi:
            stats["turns_no_speaker"] += 1
            continue
        if turn.multi or turn.speaker is None:
            stats["turns_overlap"] += 1
            continue

        cur: list = []  # accumulated (span, CleanResult)

        def flush() -> None:
            nonlocal cur, counter
            if not cur:
                return
            t0, t1 = cur[0][0].t0, cur[-1][0].t1
            if t1 - t0 < min_dur:
                stats["utts_too_short"] += 1
                cur = []
                return
            text = " ".join(c.text for _, c in cur)
            counter += 1
            utts.append(
                Utterance(
                    utt_id=f"{recording_id}-{counter:04d}",
                    recording_id=recording_id,
                    speaker_id=turn.speaker,
                    subcorpus=sub,
                    t0=t0,
                    t1=t1,
                    text=text,
                    has_event=any(s.has_event for s, _ in cur),
                    has_cutoff=any(c.had_cutoff for _, c in cur),
                    has_filler=any(c.had_filler for _, c in cur),
                    suspect_tokens=tuple(t for _, c in cur for t in c.suspect_tokens),
                )
            )
            cur = []

        for span in turn.spans:
            cleaned = clean_std(span.text)
            if not cleaned.text:
                flush()  # empty span breaks contiguity of speech
                continue
            if span.t1 - span.t0 > max_dur:
                flush()
                stats["spans_too_long"] += 1
                continue
            if cur and span.t1 - cur[0][0].t0 > max_dur:
                flush()
            cur.append((span, cleaned))
        flush()

    return utts, stats
