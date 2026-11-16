# slovene-asr-bench

Fine-tuning and evaluation of streaming and offline speech recognition for
Slovenian, on the open [ARTUR 1.0](http://hdl.handle.net/11356/1776) corpus.
Runs end to end on [Modal](https://modal.com). Total compute cost: about $25.

## Setup

| Split |   Audio | Contents                                                     |
| ----- | ------: | ------------------------------------------------------------ |
| Train | 10.75 h | ARTUR public speech, speaker- and session-disjoint from test |
| Test  |  1.80 h | 379 utterances, 14k words, 4 recordings                      |

| System                                   | Type                          | Params | Training                                                                                      |
| ---------------------------------------- | ----------------------------- | -----: | --------------------------------------------------------------------------------------------- |
| `nvidia/nemotron-3.5-asr-streaming-0.6b` | streaming, FastConformer-RNNT |  0.6 B | full fine-tune, [NeMo recipe](https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr) |
| `openai/whisper-large-v3-turbo`          | offline, encoder-decoder      |  0.8 B | full fine-tune, HF `Seq2SeqTrainer`                                                           |
| Soniox `stt-async-v6` / `stt-rt-v6`      | commercial API                |      – | none, `language_hints=["sl"]` only                                                            |

All systems use the same scorer. _Norm_ means lowercased, punctuation removed,
č š ž kept.

## Results

WER (norm)

| System                 | Mode               | Out of the box | Fine-tuned |
| ---------------------- | ------------------ | -------------: | ---------: |
| Soniox stt-async-v6    | offline            |       **4.7%** |          – |
| Soniox stt-rt-v6       | streaming          |       **4.9%** |          – |
| Whisper large-v3-turbo | offline            |          15.2% |   **9.2%** |
| Nemotron 3.5 ASR       | streaming, 1120 ms |          54.4% |  **15.7%** |
| Nemotron 3.5 ASR       | streaming, 80 ms   |          60.2% |  **18.3%** |

- Fine-tuning on 10.75 h cuts Nemotron's WER by more than two thirds and
  Whisper's by 40%.
- Chunk size matters: the same Nemotron checkpoint loses 2.6 points going
  from 1120 ms to 80 ms chunks.
- At 1120 ms, fine-tuned Nemotron (15.7%) is within half a point of offline
  Whisper before fine-tuning (15.2%).
- Soniox, with no fine-tuning, has about half the WER of fine-tuned Whisper.

### All metrics

<!-- RESULTS:START -->

| System                                      | Mode                      | WER (norm) | CER (norm) | WER (raw) | CER (raw) | RTFx   |
| ------------------------------------------- | ------------------------- | ---------- | ---------- | --------- | --------- | ------ |
| whisper-large-v3-turbo (baseline)           | offline                   | **15.2%**  | 6.5%       | 24.0%     | 8.4%      | 384.35 |
| whisper-large-v3-turbo (finetuned)          | offline                   | **9.2%**   | 2.4%       | 18.6%     | 4.2%      | 286.71 |
| nemotron-3.5-asr-streaming-0.6b (baseline)  | streaming @ 80 ms         | **60.2%**  | 26.5%      | 67.3%     | 29.0%     | 33.72  |
| nemotron-3.5-asr-streaming-0.6b (baseline)  | streaming @ 1120 ms       | **54.4%**  | 23.1%      | 62.0%     | 25.6%     | 63.03  |
| nemotron-3.5-asr-streaming-0.6b (finetuned) | streaming @ 80 ms         | **18.3%**  | 4.9%       | 28.7%     | 7.0%      | 29.82  |
| nemotron-3.5-asr-streaming-0.6b (finetuned) | streaming @ 1120 ms       | **15.7%**  | 4.2%       | 25.9%     | 6.2%      | 94.38  |
| stt-async-v6 (commercial API)               | offline (async API)       | **4.7%**   | 2.6%       | 17.1%     | 4.9%      | —      |
| stt-rt-v6 (commercial API)                  | streaming (real-time API) | **4.9%**   | 2.6%       | 17.9%     | 5.0%      | —      |

<!-- RESULTS:END -->

RTFx is throughput: batch decode for Whisper, and eval wall time including
model load for Nemotron. Compare values within the column only; it is not
a single-stream latency figure.

## Soniox vs. fine-tuned open models

### Error rate by word type

|                                          | Nemotron FT 80 ms | Nemotron FT 1120 ms | Whisper FT | Soniox rt | Soniox async |
| ---------------------------------------- | ----------------: | ------------------: | ---------: | --------: | -----------: |
| WER                                      |              18.3 |                15.7 |        9.2 |       4.9 |          4.7 |
| WER, formatting excluded¹                |              18.3 |                15.7 |        9.2 |       3.0 |          3.0 |
| Names (n = 359)                          |              36.5 |                34.8 |       20.9 |      10.0 |         10.6 |
| Words not in the training set (n = 2123) |              37.5 |                35.2 |       20.3 |       5.8 |          6.1 |
| Function words, ≤ 2 chars (n = 4317)     |              12.8 |                10.6 |        4.8 |       2.2 |          2.0 |

All values in %. ¹ Digits and symbols scored against spelled-out references,
and inserted hesitations.

### Per recording (WER %)

| Recording | Nemotron FT 80 ms | Nemotron FT 1120 ms | Whisper FT | Soniox rt | Soniox async |
| --------- | ----------------: | ------------------: | ---------: | --------: | -----------: |
| P500014   |              18.6 |                15.7 |        9.5 |       4.9 |          4.5 |
| P500015   |              20.0 |                17.4 |       10.5 |       4.6 |          4.7 |
| P500016   |              19.2 |                17.1 |        8.6 |       3.8 |          4.4 |
| P500029   |              14.8 |                12.7 |        7.1 |       5.9 |          5.4 |

### Per utterance, Soniox async vs. fine-tuned models

Formatting excluded, 379 utterances.

|                     | Soniox better | Tie | Open model better |
| ------------------- | ------------: | --: | ----------------: |
| Whisper FT          |           266 |  90 |                23 |
| Nemotron FT 1120 ms |           338 |  35 |                 6 |

### Where Soniox loses points

**1. Names.** Without context, Soniox gets 10.6% of names wrong:
`Onkonetu → Onetu`, `primarij Cevc → Pri Mari Cevc`,
`Rotdajčeva → rod dajčeva`, `Draškovič → Drašković`.

**2. Standardization.** Soniox writes standard Slovenian where the speaker
used colloquial forms: `bojo → bodo`, `neki → nek`, `kakor → kot`,
`doktor → dr.` The cost is about 0.2 points of WER.

## Pipeline

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

- Speaker IDs come from the TRS XML. Splits are built on a speaker graph, so
  they are speaker- and session-disjoint by construction. ID lists are in
  `splits/`.
- Transcripts keep case and punctuation. Fillers (`#eee`) and cut-off markers
  are removed. Test utterances with noise events or cut-offs are excluded.
- Utterances are 0.5–25 s, resampled to 16 kHz mono.
- ARTUR has no published canonical split, so this repo defines its own.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/), a Modal account
(`uv run modal setup`) and a Hugging Face token.

```bash
uv sync
uv run modal secret create huggingface-secret HF_TOKEN="$(cat ~/.cache/huggingface/token)"
./scripts/run_all.sh
```

Soniox runs: `scripts/soniox_eval.py`, `scripts/soniox_rt_eval.py`.
Error analysis: `scripts/error_analysis.py`.

Every stage is idempotent and checkpointed on a Modal volume.
`configs/tiny.yaml` scales to 100 h+ by adding ARTUR tars and raising
`data.hours`.

| Stage                  | Hardware     |   Cost |
| ---------------------- | ------------ | -----: |
| Data prep              | CPU          |    ~$1 |
| Whisper full FT        | L40S, 15 min | ~$0.50 |
| Nemotron full FT       | H100, 3 h    |   ~$12 |
| Six open-model evals   | –            |    ~$4 |
| Smoke tests, dead ends | –            |    ~$3 |

## Limitations

- 10.75 h is a small fine-tune. The RSDO Conformer reaches about 5.6% WER
  with 630 h.
- The test set is 4 recordings from one setting: public STA online panels on
  health and patient associations. It has no conversational or read speech.
- References are ARTUR's standardized transcripts and were not checked by
  ear.
- Each API system was run once, on 2026-11-15.
- Names are detected heuristically (mid-sentence capitals), so the category
  includes some places and organizations.

## License

Code: MIT. Data: ARTUR 1.0, CC BY-SA 4.0, CLARIN.SI
([audio](http://hdl.handle.net/11356/1776),
[transcriptions](http://hdl.handle.net/11356/1772)). It is downloaded at run
time and not redistributed. Models: Nemotron 3.5 ASR (OpenMDW-1.1), Whisper
(Apache-2.0), NeMo (Apache-2.0). See [NOTICE.md](NOTICE.md).

## Sources

- Nemotron 3.5 ASR: [model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) ·
  [fine-tuning blog](https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr) ·
  [notebook](https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb)
- Whisper large-v3-turbo: [model card](https://huggingface.co/openai/whisper-large-v3-turbo)
- ARTUR: [Verdonik et al., LREV 2024](https://link.springer.com/article/10.1007/s10579-024-09792-2)
- Slovene ASR reference: [RSDO Conformer](https://www.clarin.si/repository/xmlui/handle/11356/1737) ·
  [SloBench leaderboard](https://slobench.cjvt.si/leaderboard/view/10)
- Modal: [Whisper fine-tune example](https://modal.com/docs/examples/fine_tune_asr)
