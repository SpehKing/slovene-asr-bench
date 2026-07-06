# Third-party data and models

The **code** in this repository is MIT-licensed (see LICENSE). It builds on the
following third-party resources, which keep their own licenses:

## ARTUR 1.0 speech database (training/eval data)

- Verdonik, Darinka; et al., 2023: *ASR database ARTUR 1.0 (audio)* and
  *(transcriptions)*, Slovenian language resource repository CLARIN.SI.
  - Audio: http://hdl.handle.net/11356/1776
  - Transcriptions: http://hdl.handle.net/11356/1772
- License: **CC BY-SA 4.0**. This repository does **not** redistribute ARTUR
  audio or transcription text — only utterance ID lists (`splits/`) and
  aggregate metrics (`results/`). The pipeline downloads ARTUR directly from
  CLARIN.SI at run time.

## NVIDIA Nemotron 3.5 ASR Streaming 0.6B

- https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
- License: **OpenMDW-1.1** (https://openmdw.ai/license/1-1/). Fine-tuned
  checkpoints derived from this model are governed by that license and are not
  committed to this repository.

## OpenAI Whisper large-v3-turbo

- https://huggingface.co/openai/whisper-large-v3-turbo
- License: **Apache-2.0**.

## NVIDIA NeMo

- https://github.com/NVIDIA/NeMo — **Apache-2.0**. Used inside the Modal
  training image, pinned to a fixed commit.
