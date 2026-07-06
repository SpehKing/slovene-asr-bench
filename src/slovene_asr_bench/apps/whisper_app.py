"""Modal app "slasr-whisper": full fine-tune + offline eval of Whisper large-v3-turbo.

Train:  uv run modal run -m slovene_asr_bench.apps.whisper_app::train --config configs/tiny.yaml
Eval:   uv run modal run -m slovene_asr_bench.apps.whisper_app::evaluate --checkpoint baseline
        uv run modal run -m slovene_asr_bench.apps.whisper_app::evaluate --checkpoint finetuned
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

app = modal.App("slasr-whisper")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1")
    .uv_pip_install(
        "torch==2.7.1",
        "transformers==4.53.2",
        "datasets==3.6.0",
        "accelerate==1.7.0",
        "jiwer==3.1.0",
        "soundfile==0.13.1",
        "librosa==0.11.0",
        "numba==0.61.2",
        "numpy==2.2.4",
    )
    .env({"HF_HOME": "/cache"})
    .add_local_python_source("slovene_asr_bench")
)

data_vol = modal.Volume.from_name("slasr-artur-data", create_if_missing=True)
out_vol = modal.Volume.from_name("slasr-outputs", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

VOLUMES = {"/data": data_vol, "/outputs": out_vol, "/cache": hf_cache_vol}
SECRETS = [modal.Secret.from_name("huggingface-secret")]

FT_DIR = "/outputs/whisper-fullft-sl"


def _load_processor_model(model_path: str, for_training: bool):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(
        model_path, language="slovenian", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=None if for_training else torch.bfloat16
    )
    # turbo ships forced_decoder_ids that fight explicit language/task settings
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = "slovenian"
    model.generation_config.task = "transcribe"
    return processor, model


def _prepare(ds, processor):
    def fn(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    # in-process map: num_proc forks after torch is imported and deadlocks
    return ds.map(fn, remove_columns=ds.column_names, num_proc=None)


class _Collator:
    def __init__(self, processor, decoder_start_token_id: int):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features):
        import torch

        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


@app.function(
    image=image, gpu="L40S", timeout=4 * 3600, volumes=VOLUMES, secrets=SECRETS, cpu=8, memory=32768
)
def train_remote(cfg: dict) -> dict:
    import jiwer
    from datasets import load_from_disk
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    from slovene_asr_bench.artur.textnorm import normalize_eval

    wcfg = cfg["whisper"]
    processor, model = _load_processor_model(wcfg["model_id"], for_training=True)
    model.config.use_cache = False

    train_ds = _prepare(load_from_disk("/data/hf/train"), processor)
    dev_ds = _prepare(load_from_disk("/data/hf/dev"), processor)
    print(f"[train] {len(train_ds)} train / {len(dev_ds)} dev utterances")

    def compute_metrics(pred):
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        hyps = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pairs = [(normalize_eval(r), normalize_eval(h)) for r, h in zip(refs, hyps)]
        pairs = [(r, h) for r, h in pairs if r]
        return {"wer": jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])}

    smoke = wcfg.get("max_steps", -1) > 0
    args = Seq2SeqTrainingArguments(
        output_dir="/tmp/whisper-train",
        per_device_train_batch_size=wcfg["batch_size"],
        per_device_eval_batch_size=wcfg["eval_batch_size"],
        gradient_accumulation_steps=wcfg["grad_accum"],
        learning_rate=wcfg["learning_rate"],
        warmup_steps=wcfg["warmup_steps"],
        num_train_epochs=wcfg["epochs"],
        max_steps=wcfg["max_steps"] if smoke else -1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps" if smoke else "epoch",
        eval_steps=10 if smoke else None,
        save_strategy="steps" if smoke else "epoch",
        save_steps=10 if smoke else None,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        report_to=[],
        dataloader_num_workers=4,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=_Collator(processor, model.config.decoder_start_token_id),
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    baseline_dev = trainer.evaluate()
    print(f"[train] dev before FT: {baseline_dev}")
    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0
    final_dev = trainer.evaluate()
    print(f"[train] dev after FT: {final_dev}")

    model.config.use_cache = True
    trainer.save_model(FT_DIR)
    processor.save_pretrained(FT_DIR)
    summary = {
        "dev_wer_norm_before": round(baseline_dev.get("eval_wer", -1), 4),
        "dev_wer_norm_after": round(final_dev.get("eval_wer", -1), 4),
        "train_seconds": round(train_secs),
        "config": wcfg,
    }
    from slovene_asr_bench.config import to_plain_json

    summary = to_plain_json(summary)
    Path(FT_DIR, "train_summary.json").write_text(json.dumps(summary, indent=2))
    out_vol.commit()
    return summary


@app.function(
    image=image, gpu="L40S", timeout=2 * 3600, volumes=VOLUMES, secrets=SECRETS, cpu=8, memory=32768
)
def eval_remote(cfg: dict, checkpoint: str) -> dict:
    import torch
    from datasets import load_from_disk

    from slovene_asr_bench.eval.metrics import compute_asr_metrics

    wcfg = cfg["whisper"]
    model_path = wcfg["model_id"] if checkpoint == "baseline" else FT_DIR
    processor, model = _load_processor_model(model_path, for_training=False)
    model = model.to("cuda").eval()

    test = load_from_disk("/data/hf/test")
    print(f"[eval] {len(test)} test utterances from /data/hf/test ({checkpoint})")
    bs = wcfg["eval_batch_size"]

    refs: list[str] = []
    hyps: list[str] = []
    audio_seconds = 0.0
    wall = 0.0
    for start in range(0, len(test), bs):
        rows = test[start : start + bs]
        feats = processor.feature_extractor(
            [a["array"] for a in rows["audio"]], sampling_rate=16000, return_tensors="pt"
        ).input_features.to("cuda", dtype=torch.bfloat16)
        t0 = time.time()
        with torch.no_grad():
            ids = model.generate(
                input_features=feats,
                language="slovenian",
                task="transcribe",
                num_beams=1,
                do_sample=False,
                max_new_tokens=225,
            )
        wall += time.time() - t0
        hyps.extend(processor.tokenizer.batch_decode(ids, skip_special_tokens=True))
        refs.extend(rows["text"])
        audio_seconds += sum(rows["duration"])

    from slovene_asr_bench.config import to_plain_json

    metrics = compute_asr_metrics(refs, hyps, audio_seconds=audio_seconds, wall_seconds=wall)
    payload = {
        "system": f"whisper-{checkpoint}-offline",
        "model": wcfg["model_id"],
        "checkpoint": checkpoint,
        "mode": "offline",
        "split": "test",
        **metrics,
        "decode_args": {"language": "sl", "num_beams": 1, "batch_size": bs, "dtype": "bf16"},
        "n_utts": len(refs),
        "audio_hours": round(audio_seconds / 3600, 3),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    payload = to_plain_json(payload)
    out_dir = Path("/outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{payload['system']}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    out_vol.commit()
    print(json.dumps({k: v for k, v in payload.items() if k != "decode_args"}, indent=2))
    return payload


@app.local_entrypoint()
def train(config: str = "configs/tiny.yaml"):
    from slovene_asr_bench.config import load_config

    summary = train_remote.remote(load_config(config))
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def evaluate(config: str = "configs/tiny.yaml", checkpoint: str = "baseline"):
    from slovene_asr_bench.config import load_config

    if checkpoint not in ("baseline", "finetuned"):
        raise SystemExit("--checkpoint must be baseline or finetuned")
    payload = eval_remote.remote(load_config(config), checkpoint)
    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"{payload['system']}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[evaluate] wrote {out}")
