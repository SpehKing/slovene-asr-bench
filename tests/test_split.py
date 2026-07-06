from dataclasses import dataclass

from slovene_asr_bench.artur.split import build_components, make_splits


@dataclass
class FakeUtt:
    utt_id: str
    recording_id: str
    speaker_id: str
    subcorpus: str
    duration: float


def _pool():
    """Synthetic pool: speakers A+B share recording R1 (one component);
    ~30 single-speaker components across two subcorpora."""
    utts = []
    for i in range(6):  # A and B linked via shared recording R1
        utts.append(FakeUtt(f"R1-{i:04d}", "R1", "A" if i % 2 else "B", "J", 60.0))
    utts.append(FakeUtt("R2-0001", "R2", "A", "J", 120.0))  # A also alone in R2
    for s in range(30):
        spk = f"S{s:02d}"
        for j in range(4):
            utts.append(FakeUtt(f"P{s:02d}-{j:04d}", f"P{s:02d}", spk, "N" if s % 2 else "J", 90.0))
    return utts


def test_linked_speakers_form_one_component():
    comps = build_components(_pool())
    linked = [c for c in comps if "A" in c.speakers]
    assert len(linked) == 1
    assert linked[0].speakers == {"A", "B"}
    assert linked[0].recordings == {"R1", "R2"}


def test_splits_are_speaker_disjoint_and_meet_targets():
    utts = _pool()  # total = 6*60 + 120 + 30*4*90 = 3.25 h
    splits, stats = make_splits(utts, hours={"train": 1.5, "dev": 0.25, "test": 0.5}, seed=17)
    spk = {
        name: {u.speaker_id for u in utts if u.utt_id in set(ids)}
        for name, ids in splits.items()
    }
    assert not spk["train"] & spk["dev"]
    assert not spk["train"] & spk["test"]
    assert not spk["dev"] & spk["test"]
    assert stats["hours"]["test"] >= 0.4
    assert stats["hours"]["train"] >= 1.2
    # component atomicity: A and B always land together
    for name, ids in splits.items():
        present = {"A", "B"} & spk[name]
        assert present in (set(), {"A", "B"})


def test_test_split_covers_all_subcorpora():
    splits, stats = make_splits(_pool(), hours={"train": 1.5, "dev": 0.25, "test": 0.5}, seed=17)
    assert set(stats["test_subcorpora"]) == {"J", "N"}


def test_deterministic_given_seed():
    a, _ = make_splits(_pool(), hours={"train": 1.5, "dev": 0.25, "test": 0.5}, seed=17)
    b, _ = make_splits(_pool(), hours={"train": 1.5, "dev": 0.25, "test": 0.5}, seed=17)
    assert a == b
    c, _ = make_splits(_pool(), hours={"train": 1.5, "dev": 0.25, "test": 0.5}, seed=99)
    assert isinstance(c, dict)  # different seed still valid (may or may not differ)
