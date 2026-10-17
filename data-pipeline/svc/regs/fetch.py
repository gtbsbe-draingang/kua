"""Download the raw HTML of each act in sources.yaml into DATA_DIR/raw/.

Run once while online. Everything after this step works offline, so the raw files (or the
finished index) can be copied into a network without internet access.
"""
import hashlib
import json
import time
from pathlib import Path

import requests
import yaml

from . import config


def load_sources(path: Path = config.SOURCES_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["documents"]


def raw_path(code: str) -> Path:
    return config.RAW_DIR / f"{code}.html"


def fetch_all(force: bool = False) -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for doc in load_sources():
        out = raw_path(doc["code"])
        if out.exists() and not force:
            print(f"skip  {doc['code']} (already downloaded; --force to refresh)")
            continue
        resp = requests.get(doc["fetch_url"], timeout=config.FETCH_TIMEOUT,
                            headers={"User-Agent": config.USER_AGENT})
        resp.raise_for_status()
        resp.encoding = resp.encoding or "utf-8"
        html = resp.text
        if "<article" not in html:
            raise RuntimeError(f"{doc['code']}: no <article> in {doc['fetch_url']}; the page layout changed")
        out.write_text(html, encoding="utf-8")
        meta = {"code": doc["code"], "fetch_url": doc["fetch_url"],
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(), "bytes": len(html)}
        out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"saved {doc['code']} {len(html) // 1024} KB")
