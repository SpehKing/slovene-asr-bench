"""Parser for ARTUR's Transcriber (.trs) XML files.

TRS structure (trans-14.dtd):

    <Trans audio_filename="Artur-N-G6033-P600101-avd" ...>
      <Speakers><Speaker id="spk1" name="Artur-N-G6033" .../></Speakers>
      <Episode><Section ...>
        <Turn startTime="0.0" endTime="12.5" speaker="spk1">
          <Sync time="0.0"/> text between syncs
          <Sync time="4.2"/> more text <Event desc="smeh" type="noise"/> tail text
        </Turn>
        <Turn startTime="12.5" endTime="14.0"/>              <!-- pause: no speaker -->
        <Turn ... speaker="spk1 spk2">...</Turn>             <!-- overlap: multiple -->
      </Section></Episode>
    </Trans>

Speaker identity must be resolved through <Speaker name=...> (canonical IDs like
"Artur-N-G6033"); multi-speaker recordings use "Gvecg" in the *filename*, so the
filename is never a reliable speaker key.

Text lives in mixed content: `turn.text` before the first child, then in the
`.tail` of every child element. Stdlib ElementTree only.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


@dataclass
class Span:
    """Sub-segment between two <Sync> marks (or turn boundaries)."""

    t0: float
    t1: float
    text: str
    has_event: bool = False


@dataclass
class Turn:
    speaker: str | None  # resolved canonical name; None if turn had no speaker
    multi: bool  # True if the speaker attribute listed several speakers
    t0: float
    t1: float
    spans: list[Span] = field(default_factory=list)


@dataclass
class TrsDoc:
    audio_filename: str | None
    speakers: dict[str, str]  # spk id -> canonical name
    turns: list[Turn]


def parse_trs(xml_bytes: bytes | str) -> TrsDoc:
    if isinstance(xml_bytes, str):
        xml_bytes = xml_bytes.encode("utf-8")
    root = ET.fromstring(xml_bytes)
    if root.tag != "Trans":
        raise ValueError(f"not a TRS document (root <{root.tag}>)")

    speakers: dict[str, str] = {}
    for spk in root.iter("Speaker"):
        sid = spk.get("id")
        if sid:
            speakers[sid] = spk.get("name") or sid

    turns: list[Turn] = []
    for turn_el in root.iter("Turn"):
        t0 = float(turn_el.get("startTime", "nan"))
        t1 = float(turn_el.get("endTime", "nan"))
        spk_attr = turn_el.get("speaker")
        if spk_attr:
            ids = spk_attr.split()
            multi = len(ids) > 1
            speaker = speakers.get(ids[0], ids[0]) if not multi else None
        else:
            speaker, multi = None, False

        turn = Turn(speaker=speaker, multi=multi, t0=t0, t1=t1)

        cur_t = t0
        parts: list[str] = []
        has_event = False

        def flush(end_t: float) -> None:
            nonlocal parts, has_event, cur_t
            text = " ".join(p.strip() for p in parts if p and p.strip())
            if text:
                turn.spans.append(Span(cur_t, end_t, text, has_event))
            parts, has_event = [], False

        if turn_el.text:
            parts.append(turn_el.text)
        for child in turn_el:
            if child.tag == "Sync":
                sync_t = float(child.get("time", cur_t))
                flush(sync_t)
                cur_t = sync_t
            elif child.tag == "Who":
                # overlap sub-attribution inside a multi-speaker turn
                turn.multi = True
            else:
                # Event / Comment / Background / Vocal ... — mark the span
                has_event = True
            if child.tail:
                parts.append(child.tail)
        flush(t1)

        turns.append(turn)

    return TrsDoc(
        audio_filename=root.get("audio_filename"),
        speakers=speakers,
        turns=turns,
    )
