"""Speaker-disjoint train/dev/test splits.

ARTUR publishes no canonical split (the RSDO Conformer used one but never
released the lists), so we build our own with a structural guarantee:

- nodes = speakers, edges = recordings (a multi-speaker recording connects all
  its speakers); connected components are the atomic assignment units.
- a component goes to exactly one split -> speaker- AND session-disjointness
  hold by construction, not by luck.

Packing: components ordered largest-first with seeded tie-breaking, greedily
assigned to test, then dev, then train, each up to `1.2 * target` hours.
Train is truncated at ~ its target; the remainder stays unassigned (that's the
scale-up reserve). If test misses a configured subcorpus, the smallest
remaining component covering it is swapped in.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Component:
    key: str
    speakers: set[str] = field(default_factory=set)
    recordings: set[str] = field(default_factory=set)
    utt_ids: list[str] = field(default_factory=list)
    hours: float = 0.0
    subcorpora: set[str] = field(default_factory=set)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_components(utts: list) -> list[Component]:
    """`utts` are segment.Utterance (or anything with utt_id, recording_id,
    speaker_id, subcorpus, duration)."""
    uf = _UnionFind()
    by_recording: dict[str, set[str]] = {}
    for u in utts:
        by_recording.setdefault(u.recording_id, set()).add(u.speaker_id)
    for speakers in by_recording.values():
        speakers = sorted(speakers)
        for other in speakers[1:]:
            uf.union(speakers[0], other)

    comps: dict[str, Component] = {}
    for u in utts:
        root = uf.find(u.speaker_id)
        comp = comps.setdefault(root, Component(key=root))
        comp.speakers.add(u.speaker_id)
        comp.recordings.add(u.recording_id)
        comp.utt_ids.append(u.utt_id)
        comp.hours += u.duration / 3600.0
        comp.subcorpora.add(u.subcorpus)
    for comp in comps.values():
        comp.utt_ids.sort()
    return sorted(comps.values(), key=lambda c: c.key)


def make_splits(
    utts: list,
    hours: dict[str, float],
    seed: int = 17,
) -> tuple[dict[str, list[str]], dict]:
    comps = build_components(utts)
    rng = random.Random(seed)
    order = sorted(comps, key=lambda c: (-c.hours, rng.random()))

    targets = {"test": hours["test"], "dev": hours["dev"], "train": hours["train"]}
    assigned: dict[str, list[Component]] = {"test": [], "dev": [], "train": []}
    got: dict[str, float] = {"test": 0.0, "dev": 0.0, "train": 0.0}
    remaining: list[Component] = []

    for comp in order:
        placed = False
        for split in ("test", "dev", "train"):
            if got[split] < targets[split] and got[split] + comp.hours <= targets[split] * 1.2:
                assigned[split].append(comp)
                got[split] += comp.hours
                placed = True
                break
        if not placed:
            remaining.append(comp)

    # test must cover every subcorpus present in the pool (when >1 configured)
    pool_subs = {s for c in comps for s in c.subcorpora}
    if len(pool_subs) > 1:
        test_subs = {s for c in assigned["test"] for s in c.subcorpora}
        for missing in sorted(pool_subs - test_subs):
            candidates = sorted(
                (c for c in remaining + assigned["train"] if missing in c.subcorpora),
                key=lambda c: c.hours,
            )
            if candidates:
                comp = candidates[0]
                if comp in assigned["train"]:
                    assigned["train"].remove(comp)
                    got["train"] -= comp.hours
                else:
                    remaining.remove(comp)
                assigned["test"].append(comp)
                got["test"] += comp.hours

    splits = {
        name: sorted(uid for c in comps_ for uid in c.utt_ids)
        for name, comps_ in assigned.items()
    }

    # invariants: disjoint speakers across splits
    spk = {name: {s for c in comps_ for s in c.speakers} for name, comps_ in assigned.items()}
    for a in spk:
        for b in spk:
            if a < b and spk[a] & spk[b]:
                raise AssertionError(f"speaker leakage between {a} and {b}: {spk[a] & spk[b]}")

    stats = {
        "hours": {k: round(v, 3) for k, v in got.items()},
        "targets": targets,
        "n_utts": {k: len(v) for k, v in splits.items()},
        "n_speakers": {k: len(v) for k, v in spk.items()},
        "n_components": {k: len(v) for k, v in assigned.items()},
        "unassigned_hours": round(sum(c.hours for c in remaining), 3),
        "test_subcorpora": sorted({s for c in assigned["test"] for s in c.subcorpora}),
    }
    return splits, stats
