# slovene-asr-bench

Fine-tuning and benchmarking **streaming vs. offline ASR on Slovenian** — a
low-resource language — using the open [ARTUR 1.0](http://hdl.handle.net/11356/1776)
corpus, entirely on [Modal](https://modal.com) serverless GPUs.

Two models, both **fully fine-tuned** on identical speaker-disjoint data:

| | `nvidia/nemotron-3.5-asr-streaming-0.6b` | `openai/whisper-large-v3-turbo` |
|---|---|---|
| Architecture | cache-aware FastConformer-RNNT (streaming) | attention encoder-decoder (offline) |
| Params | 0.6 B | 0.8 B |
| Slovenian support out of the box | "adaptation-ready" (tokenizer only) | trained, weak |
| Fine-tune recipe | [NVIDIA's official NeMo recipe](https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr) | HF `Seq2SeqTrainer` |
| Evaluated | **streaming** at 80 ms and 1120 ms chunks | offline batch decode |

## Results

Held-out ARTUR test split (speaker- and session-disjoint, spontaneous speech).
WER/CER computed identically for every row (`src/slovene_asr_bench/eval/metrics.py`);
"norm" = lowercased, punctuation stripped, diacritics preserved.

<!-- RESULTS:START -->
| System | Mode | WER (norm) | CER (norm) | WER (raw) | CER (raw) | RTFx |
|---|---|---|---|---|---|---|
| whisper-large-v3-turbo (baseline) | offline | **15.2%** | 6.5% | 24.0% | 8.4% | 384.35 |
| whisper-large-v3-turbo (finetuned) | offline | **9.2%** | 2.4% | 18.6% | 4.2% | 286.71 |
| nemotron-3.5-asr-streaming-0.6b (baseline) | streaming @ 80 ms | **60.2%** | 26.5% | 67.3% | 29.0% | 33.72 |
| nemotron-3.5-asr-streaming-0.6b (baseline) | streaming @ 1120 ms | **54.4%** | 23.1% | 62.0% | 25.6% | 63.03 |
| nemotron-3.5-asr-streaming-0.6b (finetuned) | streaming @ 80 ms | **18.3%** | 4.9% | 28.7% | 7.0% | 29.82 |
| nemotron-3.5-asr-streaming-0.6b (finetuned) | streaming @ 1120 ms | **15.7%** | 4.2% | 25.9% | 6.2% | 94.38 |
<!-- RESULTS:END -->

What to look at:

- **Baseline → fine-tuned delta.** Slovenian is an *adaptation-ready* language
  for Nemotron: the tokenizer covers it but the acoustic model has seen little
  of it. 10.75 h of ARTUR takes it from unusable (60.2% WER) to 18.3% at an
  80 ms decision window — and to within half a point of *offline* Whisper's
  baseline (15.7% vs 15.2%) at 1120 ms, from a model 25% smaller.
- **The price of latency, quantified.** The same fine-tuned checkpoint loses
  2.6 WER points going from a 1120 ms to an 80 ms chunk (15.7% → 18.3%) — the
  trade-off that matters for voice agents, measured on the same audio with the
  same scoring.
- **Fine-tuning also fixes conventions, not just acoustics.** Whisper's
  digit-vs-spelled-numerals mismatch with ARTUR (52 digit tokens in baseline
  hypotheses → 3 after FT) accounts for part of its 15.2% → 9.2% gain.

## How it works

Three independent Modal apps (NeMo and the HF stack don't share images) over
three shared volumes:

```
                       ┌──────────────────────────────────────────────┐
 CLARIN.SI (ARTUR) ──► │ slasr-data (CPU)                             │
                       │  download → parse TRS → cut 16 kHz utterances│
                       │  → speaker-disjoint split → manifests + HF ds│
                       └───────────────┬──────────────────────────────┘
                              volume: slasr-artur-data
                       ┌───────────────┴──────────────┐
        ┌──────────────▼─────────────┐  ┌─────────────▼───────────────┐
        │ slasr-whisper (L40S)       │  │ slasr-nemotron (H100/L40S)  │
        │ full FT + offline eval     │  │ NeMo full FT + cache-aware  │
        │ (transformers/Seq2Seq)     │  │ streaming eval              │
        └──────────────┬─────────────┘  └─────────────┬───────────────┘
                       └───── volume: slasr-outputs ──┘  → results/*.json → results.md
```

Data pipeline details worth knowing:

- **Speaker identity comes from the TRS XML**, not filenames (multi-speaker
  recordings are named `Gvecg`). Splits are built on a speaker graph
  (recordings = edges), so speaker- and session-disjointness hold by
  construction; the ID lists are committed under `splits/`.
- Transcripts keep **case + punctuation** (Nemotron's recipe requires it);
  filler tokens (`#eee`) are removed, cut-off markers (`word()`) stripped, and
  utterances carrying noise events or cut-offs are excluded from the test set.
- Utterances are Sync-mark merges capped at 25 s (Whisper's 30 s window) and
  floored at 0.5 s; audio is resampled 44.1 kHz → 16 kHz mono PCM.
- ARTUR has **no published canonical split** (the RSDO Conformer's 630 h split
  was never released), so this repo defines and commits its own.

## Reproduce

Prereqs: [uv](https://docs.astral.sh/uv/), a [Modal](https://modal.com) account
(`uv run modal setup`), a HF token.

```bash
uv sync
uv run modal secret create huggingface-secret HF_TOKEN="$(cat ~/.cache/huggingface/token)"

./scripts/run_all.sh          # smoke first, then the full tiny run
```

Or stage by stage — see `scripts/run_all.sh`. Every stage is idempotent and
checkpointed on the volume; `configs/tiny.yaml` (~10 h train) scales to 100 h+
by listing more ARTUR tars and raising `data.hours`.

Measured cost on Modal for the runs behind the table above: **≈ $25** total —
data prep ~$1 (CPU), Whisper full FT ~$0.50 (15 min on an L40S!), Nemotron
full FT ~$12 (3 h on an H100; the RNNT joint OOMs an L40S on long batches),
all six evals ~$4, plus ~$3 of smoke tests and dead ends. Volume storage
(~30 GiB) is inside Modal's free tier.

## Limitations

- **10 h is a tiny fine-tune.** The point is the methodology and the deltas,
  not SOTA: the RSDO Conformer reaches ~5.6 % WER with 630 h. The pipeline
  scales to that data volume by config.
- ARTUR spells numerals as words; hypotheses containing digits are penalized.
  Digit tokens per system are counted and reported in `results/*.json`.
- RTFx is batch-decode throughput (Whisper) / eval-subprocess wall time
  (Nemotron, includes model load) — comparable within a column, not a
  single-stream latency claim.
- Slovenian evaluation normalization is minimal by design (lowercase, strip
  punctuation, preserve č š ž); no inverse text normalization.

## Licenses & attribution

Code MIT. Data: **ARTUR 1.0** (CC BY-SA 4.0, CLARIN.SI —
[audio](http://hdl.handle.net/11356/1776),
[transcriptions](http://hdl.handle.net/11356/1772)) is downloaded at run time
and never redistributed here. Models: Nemotron 3.5 ASR (OpenMDW-1.1), Whisper
(Apache-2.0), NeMo (Apache-2.0). See [NOTICE.md](NOTICE.md).

## Sources

- Nemotron 3.5 ASR: [model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) ·
  [fine-tuning blog](https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr) ·
  [companion notebook](https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb)
- Whisper large-v3-turbo: [model card](https://huggingface.co/openai/whisper-large-v3-turbo)
- ARTUR corpus paper: [Verdonik et al., LREV 2024](https://link.springer.com/article/10.1007/s10579-024-09792-2)
- Reference Slovene ASR: [RSDO Conformer](https://www.clarin.si/repository/xmlui/handle/11356/1737) ·
  [SloBench ASR leaderboard](https://slobench.cjvt.si/leaderboard/view/10)
- Modal patterns: [Whisper fine-tune example](https://modal.com/docs/examples/fine_tune_asr)
