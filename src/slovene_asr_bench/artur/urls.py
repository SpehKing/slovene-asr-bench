"""ARTUR 1.0 download locations on CLARIN.SI.

The repository serves bitstreams at
`.../xmlui/bitstream/handle/11356/<item>/<file>?sequence=<n>&isAllowed=y`
where `sequence` differs per file, so we scrape the landing page for the real
href and fall back to URLs that were verified by HEAD request on 2026-07-06.
"""

from __future__ import annotations

import html as _html
import re
import urllib.request

AUDIO_LANDING = "https://www.clarin.si/repository/xmlui/handle/11356/1776"
TRS_LANDING = "https://www.clarin.si/repository/xmlui/handle/11356/1772"

TRS_ARCHIVE = "Artur_1.0_TRS.tgz"

# Verified 2026-07-06 (HTTP 200, Content-Disposition: attachment).
FALLBACK_URLS = {
    "Artur_1.0_TRS.tgz": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1772/Artur_1.0_TRS.tgz?sequence=1&isAllowed=y",
    "Artur-B-Audio_00.tar": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1776/Artur-B-Audio_00.tar?sequence=43&isAllowed=y",
    "Artur-J-Audio_00.tar": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1776/Artur-J-Audio_00.tar?sequence=62&isAllowed=y",
    "Artur-N-Audio_00.tar": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1776/Artur-N-Audio_00.tar?sequence=69&isAllowed=y",
    "Artur-P-Audio_00.tar": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1776/Artur-P-Audio_00.tar?sequence=73&isAllowed=y",
}

_UA = {"User-Agent": "slovene-asr-bench/0.1 (research; contact: repo issues)"}


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def find_href(landing_html: str, filename: str) -> str | None:
    """Find the bitstream href for `filename` in a CLARIN landing page."""
    pat = re.compile(r'href="([^"]*/' + re.escape(filename) + r'(?:\?[^"]*)?)"')
    m = pat.search(landing_html)
    if not m:
        return None
    href = _html.unescape(m.group(1))
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return "https://www.clarin.si" + href
    return href


def resolve_url(filename: str) -> str:
    """Best-effort URL for an ARTUR file: scrape landing page, else fallback."""
    landing = TRS_LANDING if filename.endswith(".tgz") else AUDIO_LANDING
    try:
        href = find_href(_fetch(landing), filename)
        if href:
            return href
    except Exception as exc:  # network hiccup -> fallback
        print(f"[urls] landing-page scrape failed for {filename}: {exc}")
    if filename in FALLBACK_URLS:
        print(f"[urls] using verified fallback URL for {filename}")
        return FALLBACK_URLS[filename]
    raise RuntimeError(
        f"could not resolve download URL for {filename}: scrape failed and no fallback known"
    )


def open_stream(url: str):
    """Open a streaming file-like object for a download URL."""
    req = urllib.request.Request(url, headers=_UA)
    return urllib.request.urlopen(req, timeout=120)
