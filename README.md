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
_Pending — run the pipeline below to populate._
<!-- RESULTS:END -->

What to look at:

- **Baseline → fine-tuned delta.** Slovenian is an *adaptation-ready* language
  for Nemotron: the tokenizer covers it but the acoustic model has seen little
  of it. ~10 h of ARTUR takes it from unusable to competitive.
- **The price of latency.** The same fine-tuned Nemotron checkpoint evaluated
  at an 80 ms decision window vs. 1120 ms quantifies the streaming
  accuracy/latency trade-off — the number that matters for voice agents.
- **Streaming RNNT vs. offline encoder-decoder** under identical training data
  and identical scoring.

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
        │ slasr-whisper (L40S)       │  │ slasr-nemotron (L40S)       │
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

Approximate cost on Modal (tiny run): **≈ $16–23** total — data prep ~$0.7 on
CPU, each fine-tune $4–10 on an L40S, all six evals ~$5. Volume storage
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
