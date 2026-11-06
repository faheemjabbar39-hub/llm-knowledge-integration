"""Stage 1b: download the latest 10-K filing for each asset manager listed in the config.

SEC requires a User-Agent with your name and email. Set `domain.sec_user_agent`
in configs/experiment.yaml first. Files go to data/raw/sec/<TICKER>_<date>.html

ESMA documents: download the PDFs you want by hand from esma.europa.eu and put
them in data/raw/esma/ (e.g. the MiFID II Q&A, UCITS Q&A, AIFMD Q&A).

Run: python scripts/02_download_sec_filings.py
"""
from __future__ import annotations

import time

import requests

from kis.common import DATA, load_config


def main():
    cfg = load_config()["domain"]
    ua = cfg["sec_user_agent"]
    if "example.com" in ua:
        raise SystemExit("Edit domain.sec_user_agent in configs/experiment.yaml (your name + email) first.")
    s = requests.Session()
    s.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
    out_dir = DATA / "raw" / "sec"
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers = s.get("https://www.sec.gov/files/company_tickers.json", timeout=30).json()
    cik_by_ticker = {v["ticker"].upper(): int(v["cik_str"]) for v in tickers.values()}

    for t in cfg["sec_tickers"]:
        cik = cik_by_ticker.get(t.upper())
        if cik is None:
            print(f"[skip] {t}: ticker not found")
            continue
        sub = s.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", timeout=30).json()
        recent = sub["filings"]["recent"]
        got = 0
        for form, acc, doc, date in zip(recent["form"], recent["accessionNumber"],
                                        recent["primaryDocument"], recent["filingDate"]):
            if form != cfg["sec_form"]:
                continue
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
            path = out_dir / f"{t}_{date}.html"
            if not path.exists():
                r = s.get(url, timeout=60)
                r.raise_for_status()
                path.write_bytes(r.content)
            print(f"[ok] {t} {date} -> {path.name}")
            got += 1
            time.sleep(0.2)  # stay well under SEC's 10 requests/second limit
            if got >= cfg["filings_per_ticker"]:
                break


if __name__ == "__main__":
    main()
