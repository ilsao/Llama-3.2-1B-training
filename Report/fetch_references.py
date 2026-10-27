#!/usr/bin/env python3
"""Fetch BibTeX from arXiv and independently check DataCite DOI metadata.

No citation is composed from model memory. The downloaded entries and checks
are kept with the report. Existing template references are backed up first.
"""
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import requests

HERE = Path(__file__).resolve().parent
PAPERS = {
    "adamw": "1711.05101", "muon": "2502.16982", "wsd": "2410.05192",
    "lrhorizon": "2409.19913", "gns": "1812.06162", "distillation": "1503.02531",
    "dclm": "2406.11794", "soap": "2409.11321", "cce": "2411.09009",
    "adamuon": "2507.11005", "dion3": "2608.11612",
}


def fetch(item):
    key, arxiv = item
    bib_url = "https://arxiv.org/bibtex/" + arxiv
    doi_url = "https://api.datacite.org/dois/10.48550/arXiv." + arxiv
    bib_r = requests.get(bib_url, timeout=35)
    bib_r.raise_for_status()
    bib = bib_r.text.strip()
    assert bib.startswith("@"), "arXiv did not return BibTeX"
    doi_r = requests.get(doi_url, timeout=35)
    doi_r.raise_for_status()
    attrs = doi_r.json()["data"]["attributes"]
    title = attrs["titles"][0]["title"]
    title_match = re.search(r"title\s*=\s*\{([^\n]+)", bib, re.I)
    assert title_match, "Missing title in retrieved BibTeX"
    clean = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    assert clean(title) in clean(title_match[1]) or clean(title_match[1].rstrip("}, ")) in clean(title), (title, title_match[1])
    bib = re.sub(r"^(@\w+\{)[^,]+,", lambda m: m[1] + key + ",", bib, count=1)
    return key, bib, {"arxiv": arxiv, "doi": attrs["doi"], "title": title,
                      "bibtex_url": bib_url, "verification_url": doi_url,
                      "bibtex_sha256": hashlib.sha256(bib.encode()).hexdigest(),
                      "verified_utc": dt.datetime.now(dt.timezone.utc).isoformat()}


def main():
    entries, verification = [], {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for key, bib, receipt in pool.map(fetch, PAPERS.items()):
            entries.append("% Retrieved from " + receipt["bibtex_url"] + "\n" + bib)
            verification[key] = receipt
            print(key, receipt["title"], flush=True)
    original = HERE / "references.bib"
    backup = HERE / "template-references.bib"
    if original.exists() and not backup.exists():
        backup.write_bytes(original.read_bytes())
    original.write_text("% Machine-fetched entries; see data/citation_verification.json.\n\n" + "\n\n".join(entries) + "\n")
    (HERE / "data/citation_verification.json").write_text(json.dumps(verification, indent=2) + "\n")


if __name__ == "__main__":
    main()
