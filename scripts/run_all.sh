#!/usr/bin/env bash
# Full reproduction sequence. Smoke first (<$1.5), then the tiny benchmark run.
set -euo pipefail
cd "$(dirname "$0")/.."

CFG_SMOKE=configs/smoke.yaml
CFG=configs/tiny.yaml

echo "== unit tests =="
uv run pytest -q

echo "== smoke: data -> whisper -> nemotron =="
uv run modal run -m slovene_asr_bench.apps.data_app --config $CFG_SMOKE
uv run modal run -m slovene_asr_bench.apps.whisper_app::train --config $CFG_SMOKE
uv run modal run -m slovene_asr_bench.apps.nemotron_app::train --config $CFG_SMOKE

echo "== tiny: data =="
uv run modal run -m slovene_asr_bench.apps.data_app --config $CFG

echo "== tiny: train both models =="
uv run modal run -m slovene_asr_bench.apps.whisper_app::train --config $CFG
uv run modal run -m slovene_asr_bench.apps.nemotron_app::train --config $CFG

echo "== evals (6 benchmark rows) =="
uv run modal run -m slovene_asr_bench.apps.whisper_app::evaluate --config $CFG --checkpoint baseline
uv run modal run -m slovene_asr_bench.apps.whisper_app::evaluate --config $CFG --checkpoint finetuned
uv run modal run -m slovene_asr_bench.apps.nemotron_app::evaluate --config $CFG --checkpoint baseline --contexts "56,0;56,13"
uv run modal run -m slovene_asr_bench.apps.nemotron_app::evaluate --config $CFG --checkpoint finetuned --contexts "56,0;56,13"

echo "== collate =="
uv run python -m slovene_asr_bench.eval.collate
