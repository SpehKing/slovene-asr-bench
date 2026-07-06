"""TRS parser tests against fixtures shaped like the real ARTUR files
(structure sampled from the actual corpus: Speakers block, Sync marks,
Event/Comment elements, pause turns, and multi-speaker overlap turns)."""

from slovene_asr_bench.artur.trs import parse_trs

FIXTURE_N = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Trans SYSTEM "trans-14.dtd">
<Trans scribe="anon" audio_filename="Artur-N-G6033-P600101-avd" version="2" version_date="220419">
<Speakers>
<Speaker id="spk1" name="Artur-N-G6033" check="no" dialect="native" accent="" scope="local"/>
<Speaker id="spk2" name="Artur-N-G6034" check="no" dialect="native" accent="" scope="local"/>
</Speakers>
<Episode>
<Section type="report" startTime="0" endTime="30.0">
<Turn startTime="0.0" endTime="12.5" speaker="spk1">
<Sync time="0.0"/> Dober dan, #eee danes je lep dan.
<Sync time="4.2"/> To je dru()
<Sync time="7.9"/><Event desc="smeh" type="noise" extent="instantaneous"/> se nadaljuje tukaj.
</Turn>
<Turn startTime="12.5" endTime="14.0">
<Sync time="12.5"/>
</Turn>
<Turn startTime="14.0" endTime="20.0" speaker="spk1 spk2">
<Sync time="14.0"/> prekrivajoči govor
</Turn>
<Turn startTime="20.0" endTime="26.0" speaker="spk2">
<Sync time="20.0"/> Čisto <Comment desc="nejasno"/> šumenje in žvižg.
</Turn>
</Section>
</Episode>
</Trans>
"""


def test_speakers_resolved_from_xml():
    doc = parse_trs(FIXTURE_N)
    assert doc.audio_filename == "Artur-N-G6033-P600101-avd"
    assert doc.speakers == {"spk1": "Artur-N-G6033", "spk2": "Artur-N-G6034"}


def test_turn_classification():
    doc = parse_trs(FIXTURE_N)
    assert len(doc.turns) == 4
    t_speech, t_pause, t_overlap, t_spk2 = doc.turns
    assert t_speech.speaker == "Artur-N-G6033" and not t_speech.multi
    assert t_pause.speaker is None and not t_pause.multi
    assert t_overlap.multi and t_overlap.speaker is None
    assert t_spk2.speaker == "Artur-N-G6034"


def test_sync_spans_and_times():
    doc = parse_trs(FIXTURE_N)
    spans = doc.turns[0].spans
    assert [(s.t0, s.t1) for s in spans] == [(0.0, 4.2), (4.2, 7.9), (7.9, 12.5)]
    assert spans[0].text == "Dober dan, #eee danes je lep dan."
    assert spans[1].text == "To je dru()"
    assert not spans[0].has_event and not spans[1].has_event


def test_event_and_comment_flag_span():
    doc = parse_trs(FIXTURE_N)
    assert doc.turns[0].spans[2].has_event  # <Event desc="smeh">
    assert doc.turns[0].spans[2].text == "se nadaljuje tukaj."
    spk2_span = doc.turns[3].spans[0]
    assert spk2_span.has_event  # <Comment>
    # mixed content around the Comment is preserved, diacritics intact
    assert spk2_span.text == "Čisto šumenje in žvižg."
