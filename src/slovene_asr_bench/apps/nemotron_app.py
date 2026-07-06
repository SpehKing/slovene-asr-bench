"""Modal app "slasr-nemotron": full fine-tune + cache-aware streaming eval of
nvidia/nemotron-3.5-asr-streaming-0.6b via NVIDIA's official NeMo recipe
(HF blog "How to Fine-Tune Nemotron 3.5 ASR", companion notebook in
nvidia-riva/tutorials). NeMo is pinned to a fixed main-branch commit.

Train:  uv run modal run -m slovene_asr_bench.apps.nemotron_app::train --config configs/tiny.yaml
Eval:   uv run modal run -m slovene_asr_bench.apps.nemotron_app::evaluate \
            --checkpoint baseline --contexts "56,0;56,13"
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import modal

# NVIDIA/NeMo main, resolved 2026-07-06
NEMO_SHA = "981fd4e0d2cb2eeff6106baaa9c6918a013e9f7b"

FINETUNE_SCRIPT = "/opt/NeMo/examples/asr/speech_to_text_finetune.py"
FINETUNE_CONFIG_PATH = "/opt/NeMo/examples/asr/conf/fastconformer/cache_aware_streaming"
FINETUNE_CONFIG_NAME = "fastconformer_transducer_bpe_streaming_prompt.yaml"
STREAMING_EVAL_SCRIPT = (
    "/opt/NeMo/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py"
)

EXP_DIR = "/outputs/nemotron-sl"
EXP_NAME = "nemotron_sl_ft"

app = modal.App("slasr-nemotron")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libsndfile1", "sox", "build-essential")
    .uv_pip_install("Cython", "packaging", "huggingface_hub")
    .uv_pip_install(f"nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@{NEMO_SHA}")
    .run_commands(
        "git init /opt/NeMo"
        " && cd /opt/NeMo"
        " && git remote add origin https://github.com/NVIDIA/NeMo"
        f" && git fetch --depth 1 origin {NEMO_SHA}"
        " && git checkout FETCH_HEAD"
        " && rm -rf /opt/NeMo/.git"
    )
    .env({"HF_HOME": "/cache", "HYDRA_FULL_ERROR": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("slovene_asr_bench")
)

data_vol = modal.Volume.from_name("slasr-artur-data", create_if_missing=True)
out_vol = modal.Volume.from_name("slasr-outputs", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

VOLUMES = {"/data": data_vol, "/outputs": out_vol, "/cache": hf_cache_vol}
SECRETS = [modal.Secret.from_name("huggingface-secret")]


def _base_ckpt(cfg: dict) -> str:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=cfg["nemotron"]["model_id"], filename=cfg["nemotron"]["nemo_filename"]
    )
    hf_cache_vol.commit()
    return path


def _run(cmd: list[str]) -> tuple[int, list[str]]:
    """Run a command, stream output to Modal logs, keep a tail for diagnostics."""
    print("[cmd]", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line)
        tail.append(line)
        if len(tail) > 200:
            tail.pop(0)
    return proc.wait(), tail


def _find_ft_ckpt() -> str:
    candidates = sorted(
        Path(EXP_DIR).rglob("*.nemo"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not candidates:
        raise FileNotFoundError(f"no .nemo checkpoint under {EXP_DIR} — did training finish?")
    return str(candidates[0])


@app.function(
    image=image, gpu="L40S", timeout=8 * 3600, volumes=VOLUMES, secrets=SECRETS, cpu=8, memory=49152
)
def train_remote(cfg: dict) -> dict:
    ncfg = cfg["nemotron"]
    base = _base_ckpt(cfg)

    cmd = [
        "python",
        FINETUNE_SCRIPT,
        f"--config-path={FINETUNE_CONFIG_PATH}",
        f"--config-name={FINETUNE_CONFIG_NAME}",
        f"+init_from_nemo_model={base}",
        "++model.train_ds.manifest_filepath=/data/manifests/train.jsonl",
        "++model.validation_ds.manifest_filepath=/data/manifests/dev.jsonl",
        "++model.optim.sched.d_model=1024",
        "++trainer.devices=1",
        "++trainer.precision=bf16",
        f"++trainer.max_epochs={ncfg['max_epochs']}",
        f"++model.train_ds.batch_duration={ncfg['batch_duration']}",
        "++model.optim.name=adamw",
        f"++model.optim.lr={ncfg['lr']}",
        f"++model.optim.weight_decay={ncfg['weight_decay']}",
        f"++model.optim.sched.warmup_steps={ncfg['warmup_steps']}",
        f"++exp_manager.exp_dir={EXP_DIR}",
        f"++exp_manager.name={EXP_NAME}",
        "++exp_manager.use_datetime_version=False",
        "++exp_manager.create_wandb_logger=False",
        "++trainer.enable_progress_bar=False",
        "++trainer.log_every_n_steps=10",
    ]
    if ncfg.get("limit_train_batches", -1) > 0:
        cmd.append(f"++trainer.limit_train_batches={ncfg['limit_train_batches']}")
        cmd.append("++trainer.limit_val_batches=4")

    t0 = time.time()
    code, tail = _run(cmd)
    if code != 0:
        raise RuntimeError(f"finetune failed (exit {code}); last lines:\n" + "\n".join(tail[-40:]))
    ckpt = _find_ft_ckpt()
    out_vol.commit()
    summary = {"ckpt": ckpt, "train_seconds": round(time.time() - t0), "config": ncfg}
    print(json.dumps(summary, indent=2))
    return summary


def _chunk_ms(att_context: tuple[int, int]) -> int:
    # cache-aware FastConformer: chunk = (right_context + 1) * 80 ms subsampled frames
    return (att_context[1] + 1) * 80


@app.function(
    image=image, gpu="L40S", timeout=3 * 3600, volumes=VOLUMES, secrets=SECRETS, cpu=8, memory=32768
)
def eval_remote(cfg: dict, checkpoint: str, contexts: str) -> list[dict]:
    from slovene_asr_bench.eval.metrics import compute_asr_metrics

    ncfg = cfg["nemotron"]
    model_path = _base_ckpt(cfg) if checkpoint == "baseline" else _find_ft_ckpt()

    manifest = Path("/data/manifests/test.jsonl")
    records = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    refs_by_path = {r["audio_filepath"]: r["text"] for r in records}
    audio_seconds = sum(r["duration"] for r in records)

    payloads = []
    for ctx_str in contexts.split(";"):
        left, right = (int(x) for x in ctx_str.split(","))
        chunk = _chunk_ms((left, right))
        out_path = f"/outputs/preds/nemotron-{checkpoint}-stream{chunk}ms.json"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        target_lang = ncfg.get("target_lang", "sl-SI")
        used_lang = target_lang
        t0 = time.time()
        code, tail = _run(_eval_cmd(model_path, str(manifest), target_lang, left, right, out_path))
        if code != 0 and any("lang" in l.lower() for l in tail[-40:]):
            print(f"[eval] target_lang={target_lang} rejected, retrying with auto")
            used_lang = "auto"
            t0 = time.time()
            code, tail = _run(_eval_cmd(model_path, str(manifest), "auto", left, right, out_path))
        if code != 0:
            raise RuntimeError(
                f"streaming eval failed (exit {code}); last lines:\n" + "\n".join(tail[-40:])
            )
        wall = time.time() - t0

        refs, hyps = _collect_pairs(out_path, records, refs_by_path)
        metrics = compute_asr_metrics(refs, hyps, audio_seconds=audio_seconds, wall_seconds=wall)
        payload = {
            "system": f"nemotron-{checkpoint}-stream{chunk}ms",
            "model": ncfg["model_id"],
            "checkpoint": checkpoint,
            "mode": "streaming",
            "split": "test",
            **metrics,
            "decode_args": {
                "att_context_size": [left, right],
                "chunk_ms": chunk,
                "target_lang": used_lang,
                "batch_size": 32,
                "decoder_type": "rnnt",
                "rtfx_note": "wall time includes one-off model load in the eval subprocess",
            },
            "n_utts": len(refs),
            "audio_hours": round(audio_seconds / 3600, 3),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        out_dir = Path("/outputs/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{payload['system']}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )
        print(json.dumps({k: v for k, v in payload.items() if k != "decode_args"}, indent=2))
        payloads.append(payload)

    out_vol.commit()
    return payloads


def _eval_cmd(
    model_path: str, manifest: str, target_lang: str, left: int, right: int, out_path: str
) -> list[str]:
    return [
        "python",
        STREAMING_EVAL_SCRIPT,
        f"model_path={model_path}",
        f"dataset_manifest={manifest}",
        f"target_lang={target_lang}",
        f"att_context_size=[{left},{right}]",
        "decoder_type=rnnt",
        "pad_and_drop_preencoded=true",
        "batch_size=32",
        "cuda=0",
        "strip_lang_tags=true",
        f"output_path={out_path}",
    ]


def _collect_pairs(out_path: str, records: list[dict], refs_by_path: dict) -> tuple[list, list]:
    """Streaming-infer output is a manifest with pred_text; align refs by
    audio_filepath when present, by order otherwise."""
    out_file = Path(out_path)
    if out_file.is_dir():  # some NeMo scripts write a directory of manifests
        files = sorted(out_file.glob("*.json")) + sorted(out_file.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"no prediction manifest inside {out_path}")
        out_file = files[0]
    preds = [json.loads(l) for l in out_file.read_text().splitlines() if l.strip()]
    refs, hyps = [], []
    for i, p in enumerate(preds):
        hyp = p.get("pred_text", p.get("text", ""))
        path = p.get("audio_filepath")
        if path and path in refs_by_path:
            ref = refs_by_path[path]
        elif i < len(records):
            ref = records[i]["text"]
        else:
            continue
        refs.append(ref)
        hyps.append(hyp)
    if not refs:
        raise RuntimeError(f"could not align any predictions from {out_path}")
    return refs, hyps


@app.local_entrypoint()
def train(config: str = "configs/tiny.yaml"):
    from slovene_asr_bench.config import load_config

    summary = train_remote.remote(load_config(config))
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def evaluate(
    config: str = "configs/tiny.yaml",
    checkpoint: str = "baseline",
    contexts: str = "56,0;56,13",
):
    from slovene_asr_bench.config import load_config

    if checkpoint not in ("baseline", "finetuned"):
        raise SystemExit("--checkpoint must be baseline or finetuned")
    payloads = eval_remote.remote(load_config(config), checkpoint, contexts)
    Path("results").mkdir(exist_ok=True)
    for payload in payloads:
        out = Path("results") / f"{payload['system']}.json"
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"[evaluate] wrote {out}")
