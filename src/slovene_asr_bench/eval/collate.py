"""Collate results/*.json into results/results.md and the README table.

Runs locally (stdlib only): `uv run python -m slovene_asr_bench.eval.collate`
README gets the same table injected between RESULTS markers so the two can
never drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def _sort_key(row: dict) -> tuple:
    model_rank = 0 if "whisper" in row["model"].lower() else 1
    ckpt_rank = 0 if row["checkpoint"] == "baseline" else 1
    latency = row.get("decode_args", {}).get("chunk_ms", 0)
    return (model_rank, ckpt_rank, latency)


def build_table(rows: list[dict]) -> str:
    rows = sorted(rows, key=_sort_key)
    header = (
        "| System | Mode | WER (norm) | CER (norm) | WER (raw) | CER (raw) | RTFx |\n"
        "|---|---|---|---|---|---|---|\n"
    )

    def pct(x: float | None) -> str:
        return f"{100 * x:.1f}%" if x is not None else "—"

    lines = []
    for r in rows:
        chunk = r.get("decode_args", {}).get("chunk_ms")
        mode = f"streaming @ {chunk} ms" if chunk else r.get("mode", "offline")
        name = f"{r['model'].split('/')[-1]} ({r['checkpoint']})"
        lines.append(
            f"| {name} | {mode} | **{pct(r.get('wer_norm'))}** | {pct(r.get('cer_norm'))} "
            f"| {pct(r.get('wer_raw'))} | {pct(r.get('cer_raw'))} | {r.get('rtfx', '—')} |"
        )
    return header + "\n".join(lines) + "\n"


def main(repo_root: Path | None = None) -> None:
    root = repo_root or Path(__file__).resolve().parents[3]
    results_dir = root / "results"
    rows = []
    for p in sorted(results_dir.glob("*.json")):
        with open(p) as f:
            rows.append(json.load(f))
    if not rows:
        raise SystemExit(f"no result JSONs found in {results_dir}")

    table = build_table(rows)
    n_utts = rows[0].get("n_utts", "?")
    hours = rows[0].get("audio_hours", "?")
    doc = (
        "# Results\n\n"
        f"Held-out ARTUR test split: {n_utts} utterances, {hours} h, speaker-disjoint.\n\n"
        + table
    )
    (results_dir / "results.md").write_text(doc)
    print(f"wrote {results_dir / 'results.md'}")

    readme = root / "README.md"
    if readme.exists():
        content = readme.read_text()
        if START in content and END in content:
            pre, rest = content.split(START, 1)
            _, post = rest.split(END, 1)
            readme.write_text(pre + START + "\n" + table + END + post)
            print(f"updated table in {readme}")
        else:
            print(f"README markers not found; skipped README injection")


if __name__ == "__main__":
    main()
