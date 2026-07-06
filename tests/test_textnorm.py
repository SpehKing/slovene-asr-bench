from slovene_asr_bench.artur.textnorm import (
    clean_std,
    count_digit_tokens,
    normalize_eval,
)


def test_clean_std_removes_fillers_keeps_case_punct():
    r = clean_std("Dober dan, #eee danes je #Eem lep dan.")
    assert r.text == "Dober dan, danes je lep dan."
    assert r.had_filler and not r.had_cutoff


def test_clean_std_strips_cutoff_marker_keeps_partial_word():
    r = clean_std("To je dru() nekaj")
    assert r.text == "To je dru nekaj"
    assert r.had_cutoff


def test_clean_std_reports_suspect_tokens():
    r = clean_std("normalno [gap] besedilo")
    assert r.suspect_tokens == ("[gap]",)


def test_normalize_eval_lowercase_punct_diacritics():
    assert normalize_eval("Dober, dan! ČŠŽ — test.") == "dober dan čšž test"


def test_normalize_eval_idempotent():
    once = normalize_eval("Živjo, „svet“ … 42-krat!")
    assert normalize_eval(once) == once


def test_digit_tokens():
    assert count_digit_tokens("leta 2023 je bilo 42 dni") == 2
    assert count_digit_tokens("dvainštirideset dni") == 0
