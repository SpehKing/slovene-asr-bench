# Results

Held-out ARTUR test split: 379 utterances, 1.803 h, speaker-disjoint.

| System | Mode | WER (norm) | CER (norm) | WER (raw) | CER (raw) | RTFx |
|---|---|---|---|---|---|---|
| whisper-large-v3-turbo (baseline) | offline | **15.2%** | 6.5% | 24.0% | 8.4% | 384.35 |
| whisper-large-v3-turbo (finetuned) | offline | **9.2%** | 2.4% | 18.6% | 4.2% | 286.71 |
| nemotron-3.5-asr-streaming-0.6b (baseline) | streaming @ 80 ms | **60.2%** | 26.5% | 67.3% | 29.0% | 33.72 |
| nemotron-3.5-asr-streaming-0.6b (baseline) | streaming @ 1120 ms | **54.4%** | 23.1% | 62.0% | 25.6% | 63.03 |
| nemotron-3.5-asr-streaming-0.6b (finetuned) | streaming @ 80 ms | **18.3%** | 4.9% | 28.7% | 7.0% | 29.82 |
| nemotron-3.5-asr-streaming-0.6b (finetuned) | streaming @ 1120 ms | **15.7%** | 4.2% | 25.9% | 6.2% | 94.38 |
