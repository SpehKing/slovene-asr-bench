"""Shared metric computation so all six benchmark rows are comparable.

Every system (Whisper offline, Nemotron streaming at any chunk size) funnels
its (references, hypotheses) through `compute_asr_metrics` and writes the same
JSON schema via `write_result`. WER from framework-internal scripts is
deliberately ignored — normalization differences would make rows incomparable.

Imports jiwer lazily: only GPU containers have it; local tooling (collate,
tests) never triggers the import.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..artur.textnorm import count_digit_tokens, normalize_eval


def compute_asr_metrics(
    refs: list[str],
    hyps: list[str],
    audio_seconds: float | None = None,
    wall_seconds: float | None = None,
) -> dict:
    import jiwer

    if len(refs) != len(hyps):
        raise ValueError(f"refs/hyps length mismatch: {len(refs)} vs {len(hyps)}")

    pairs = [(r, h) for r, h in zip(refs, hyps) if r and r.strip()]
    raw_r = [r for r, _ in pairs]
    raw_h = [h if h and h.strip() else "" for _, h in pairs]

    norm = [(normalize_eval(r), normalize_eval(h)) for r, h in pairs]
    norm = [(r, h) for r, h in norm if r]
    norm_r = [r for r, _ in norm]
    norm_h = [h for _, h in norm]

    out = {
        "wer_raw": round(jiwer.wer(raw_r, raw_h), 4),
        "cer_raw": round(jiwer.cer(raw_r, raw_h), 4),
        "wer_norm": round(jiwer.wer(norm_r, norm_h), 4),
        "cer_norm": round(jiwer.cer(norm_r, norm_h), 4),
        "n_scored": len(pairs),
        "hyp_digit_tokens": sum(count_digit_tokens(h) for h in raw_h),
    }
    if audio_seconds and wall_seconds:
        out["rtfx"] = round(audio_seconds / wall_seconds, 2)
        out["audio_seconds"] = round(audio_seconds, 1)
        out["wall_seconds"] = round(wall_seconds, 1)
    return out


def write_result(
    results_dir: str | Path,
    *,
    system: str,
    model: str,
    checkpoint: str,
    mode: str,
    split: str,
    metrics: dict,
    decode_args: dict,
    n_utts: int,
    audio_hours: float,
) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "system": system,
        "model": model,
        "checkpoint": checkpoint,
        "mode": mode,
        "split": split,
        **metrics,
        "decode_args": decode_args,
        "n_utts": n_utts,
        "audio_hours": round(audio_hours, 3),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = results_dir / f"{system}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[metrics] wrote {path}")
    return path
