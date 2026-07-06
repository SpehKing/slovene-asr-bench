"""Modal app "slasr-data": ARTUR download -> parse -> cut -> split -> export.

CPU-only. Produces, on the `slasr-artur-data` volume (mounted at /data):

    /data/raw/trs/Artur1.0/...        extracted TRS tree
    /data/raw/wav/<recording_id>.wav  44.1 kHz source wavs (only transcribed ones)
    /data/wav16k/<utt_id>.wav         16 kHz mono utterance cuts
    /data/utterances.parquet          canonical utterance table
    /data/splits/{train,dev,test}.txt
    /data/manifests/{train,dev,test}.jsonl   NeMo format (lang/target_lang=sl-SI)
    /data/hf/{train,dev,test}/        HF datasets (Audio paths into /data/wav16k)
    /data/report.json

Run:  uv run modal run -m slovene_asr_bench.apps.data_app --config configs/tiny.yaml
"""

from __future__ import annotations

import json
import os
import tarfile
import time
from pathlib import Path

import modal

NEMO_LANG = "sl-SI"

app = modal.App("slasr-data")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "pandas==2.2.3",
        "pyarrow==19.0.1",
        "soundfile==0.13.1",
        "scipy==1.15.2",
        "numpy==2.2.4",
        "datasets==3.6.0",
        "tqdm==4.67.1",
    )
    .add_local_python_source("slovene_asr_bench")
)

data_vol = modal.Volume.from_name("slasr-artur-data", create_if_missing=True)

DATA = Path("/data")
MARKERS = DATA / "markers"


def _marker(name: str) -> Path:
    MARKERS.mkdir(parents=True, exist_ok=True)
    return MARKERS / f"{name}.done"


def _trs_index() -> dict[str, Path]:
    """recording_id -> path of its -std.trs file."""
    index: dict[str, Path] = {}
    for p in (DATA / "raw/trs").rglob("*-std.trs"):
        index[p.name[: -len("-std.trs")]] = p
    return index


@app.function(image=image, volumes={"/data": data_vol}, timeout=4 * 3600, cpu=4, memory=8192)
def download_and_extract(cfg: dict) -> dict:
    from slovene_asr_bench.artur import urls

    report: dict = {"tars": {}}

    tmp = Path("/tmp/artur")  # container-local scratch: tars never touch the volume

    # 1) transcriptions (49 MB) — everything else keys off these
    if not _marker("trs").exists():
        url = urls.resolve_url(urls.TRS_ARCHIVE)
        print(f"[download] {urls.TRS_ARCHIVE} <- {url}")
        archive = urls.download_file(url, tmp / urls.TRS_ARCHIVE)
        dest = DATA / "raw/trs"
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest, filter="data")
        archive.unlink()
        _marker("trs").touch()
        data_vol.commit()

    transcribed = set(_trs_index().keys())
    report["transcribed_recordings_total"] = len(transcribed)
    print(f"[trs] {len(transcribed)} transcribed recordings in archive")

    # 2) audio tars — stream, keep only wavs that have a transcription
    wav_dir = DATA / "raw/wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    for tar_name in cfg["data"]["tars"]:
        if _marker(tar_name).exists():
            report["tars"][tar_name] = "already extracted"
            continue
        url = urls.resolve_url(tar_name)
        print(f"[download] {tar_name} <- {url}")
        kept = skipped = 0
        t0 = time.time()
        tar_path = urls.download_file(url, tmp / tar_name)
        with tarfile.open(tar_path, "r") as tf:
            for member in tf:
                base = os.path.basename(member.name)
                if not base.endswith("-avd.wav"):
                    continue
                rec_id = base[: -len("-avd.wav")]
                if rec_id not in transcribed:
                    skipped += 1
                    continue
                src = tf.extractfile(member)
                if src is None:
                    continue
                out = wav_dir / f"{rec_id}.wav"
                with open(out, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
                kept += 1
        tar_path.unlink()
        report["tars"][tar_name] = {"kept": kept, "skipped_untranscribed": skipped, "secs": round(time.time() - t0)}
        print(f"[extract] {tar_name}: kept {kept} wavs, skipped {skipped} untranscribed ({report['tars'][tar_name]['secs']}s)")
        _marker(tar_name).touch()
        data_vol.commit()

    report["wavs_on_disk"] = sum(1 for _ in wav_dir.glob("*.wav"))
    return report


def _cut_recording(args: tuple) -> int:
    """Resample one source recording to 16 kHz once, then slice its utterances."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    src_path, utts = args  # utts: list of (utt_id, t0, t1)
    out_dir = DATA / "wav16k"
    todo = [(u, t0, t1) for u, t0, t1 in utts if not (out_dir / f"{u}.wav").exists()]
    if not todo:
        return 0
    audio, sr = sf.read(src_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr == 44100:
        audio = resample_poly(audio, 160, 441).astype(np.float32)
    elif sr != 16000:
        from math import gcd

        g = gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
    for utt_id, t0, t1 in todo:
        seg = audio[int(t0 * 16000) : int(t1 * 16000)]
        sf.write(out_dir / f"{utt_id}.wav", seg, 16000, subtype="PCM_16")
    return len(todo)


@app.function(image=image, volumes={"/data": data_vol}, timeout=4 * 3600, cpu=8, memory=16384)
def build_dataset(cfg: dict) -> dict:
    from collections import Counter
    from multiprocessing import Pool

    import pandas as pd

    from slovene_asr_bench.artur.segment import build_utterances
    from slovene_asr_bench.artur.split import make_splits
    from slovene_asr_bench.artur.trs import parse_trs
    from slovene_asr_bench.config import required_pool_hours

    dcfg = cfg["data"]
    trs_index = _trs_index()
    wavs = {p.stem: p for p in (DATA / "raw/wav").glob("*.wav")}
    usable = sorted(set(trs_index) & set(wavs))
    print(f"[build] {len(usable)} recordings have both audio and transcription")

    # ---- parse TRS -> utterances
    all_utts = []
    agg = Counter()
    suspects = Counter()
    for rec_id in usable:
        doc = parse_trs(trs_index[rec_id].read_bytes())
        utts, stats = build_utterances(
            rec_id, doc.turns, min_dur=dcfg["min_utt_sec"], max_dur=dcfg["max_utt_sec"]
        )
        agg.update(stats)
        for u in utts:
            suspects.update(u.suspect_tokens)
        all_utts.extend(utts)

    pool_hours = sum(u.duration for u in all_utts) / 3600
    print(f"[build] {len(all_utts)} utterances, {pool_hours:.2f} h usable (need {required_pool_hours(cfg):.2f} h)")
    if pool_hours < required_pool_hours(cfg):
        return {"status": "insufficient", "pool_hours": round(pool_hours, 2), "needed": required_pool_hours(cfg)}

    # ---- cut audio (parallel per recording)
    (DATA / "wav16k").mkdir(exist_ok=True)
    by_rec: dict[str, list] = {}
    for u in all_utts:
        by_rec.setdefault(u.recording_id, []).append((u.utt_id, u.t0, u.t1))
    jobs = [(str(wavs[rec]), utts) for rec, utts in sorted(by_rec.items())]
    t0 = time.time()
    with Pool(processes=8) as pool:
        cut_counts = pool.map(_cut_recording, jobs)
    print(f"[cut] wrote {sum(cut_counts)} utterance wavs in {time.time() - t0:.0f}s")

    # ---- canonical table
    df = pd.DataFrame(
        {
            "utt_id": [u.utt_id for u in all_utts],
            "recording_id": [u.recording_id for u in all_utts],
            "speaker_id": [u.speaker_id for u in all_utts],
            "subcorpus": [u.subcorpus for u in all_utts],
            "t0": [u.t0 for u in all_utts],
            "t1": [u.t1 for u in all_utts],
            "duration": [u.duration for u in all_utts],
            "text": [u.text for u in all_utts],
            "clean": [not (u.has_event or u.has_cutoff) for u in all_utts],
        }
    )
    df.to_parquet(DATA / "utterances.parquet", index=False)

    # ---- speaker-disjoint splits
    splits, split_stats = make_splits(all_utts, hours=dcfg["hours"], seed=cfg["seed"])
    (DATA / "splits").mkdir(exist_ok=True)
    for name, ids in splits.items():
        (DATA / "splits" / f"{name}.txt").write_text("\n".join(ids) + "\n")

    # test purity: drop event/cutoff-carrying utterances from test only
    test_ids = set(splits["test"])
    clean_ids = set(df[df["clean"]]["utt_id"])
    effective = {
        "train": splits["train"],
        "dev": splits["dev"],
        "test": sorted(test_ids & clean_ids),
    }
    split_stats["test_utts_dropped_for_purity"] = len(test_ids) - len(effective["test"])
    split_stats["test_hours_effective"] = round(
        df[df["utt_id"].isin(effective["test"])]["duration"].sum() / 3600, 3
    )

    # ---- exports
    (DATA / "manifests").mkdir(exist_ok=True)
    by_id = df.set_index("utt_id")
    for name, ids in effective.items():
        with open(DATA / "manifests" / f"{name}.jsonl", "w") as f:
            for uid in ids:
                row = by_id.loc[uid]
                f.write(
                    json.dumps(
                        {
                            "audio_filepath": f"/data/wav16k/{uid}.wav",
                            "duration": round(float(row["duration"]), 3),
                            "text": row["text"],
                            "lang": NEMO_LANG,
                            "target_lang": NEMO_LANG,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    from datasets import Audio, Dataset

    for name, ids in effective.items():
        sub = df[df["utt_id"].isin(set(ids))][["utt_id", "duration", "text"]].copy()
        sub["audio"] = sub["utt_id"].map(lambda u: f"/data/wav16k/{u}.wav")
        ds = Dataset.from_pandas(sub.reset_index(drop=True))
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        ds.save_to_disk(str(DATA / "hf" / name))

    report = {
        "status": "ok",
        "recordings_used": len(usable),
        "utterances": len(all_utts),
        "pool_hours": round(pool_hours, 2),
        "segmentation": dict(agg),
        "suspect_tokens_top": suspects.most_common(25),
        "splits": split_stats,
        "splits_ids": splits,
    }
    (DATA / "report.json").write_text(
        json.dumps({k: v for k, v in report.items() if k != "splits_ids"}, indent=2, ensure_ascii=False)
    )
    data_vol.commit()
    return report


@app.local_entrypoint()
def main(config: str = "configs/tiny.yaml"):
    from slovene_asr_bench.config import load_config

    cfg = load_config(config)
    tars = list(cfg["data"]["tars"])
    extend = list(cfg["data"].get("extend_tars", []))

    while True:
        run_cfg = {**cfg, "data": {**cfg["data"], "tars": tars}}
        dl = download_and_extract.remote(run_cfg)
        print(json.dumps(dl, indent=2))
        report = build_dataset.remote(run_cfg)
        if report.get("status") == "insufficient" and extend:
            nxt = extend.pop(0)
            print(f"[main] pool too small ({report['pool_hours']} h) -> adding {nxt}")
            tars.append(nxt)
            continue
        break

    if report.get("status") != "ok":
        raise SystemExit(f"data pipeline failed: {report}")

    # persist split ID lists + report into the repo (committed for reproducibility)
    splits_dir = Path("splits")
    splits_dir.mkdir(exist_ok=True)
    for name, ids in report.pop("splits_ids").items():
        (splits_dir / f"{name}.txt").write_text("\n".join(ids) + "\n")
    Path("results").mkdir(exist_ok=True)
    Path("results/data_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["splits"], indent=2))
    print("[main] wrote splits/*.txt and results/data_report.json")
