"""Download settled Kalshi tennis match markets + 1-minute bid/ask candles.

For each event (match) we store both market metadata and the 1-min candles of
ONE side (the other side is its complement). Window: the 12h before the
market closed (covers pre-match + the whole match for nearly all matches).

Output: data/kalshi/<SERIES>.jsonl, one line per event. Resumable.
Usage: python3 kalshi_collect.py KXATPMATCH KXWTAMATCH ...
"""
import json
import os
import sys
import time
from datetime import datetime

import urllib.request
import urllib.error
import urllib.parse

BASE = "https://api.elections.kalshi.com/trade-api/v2"
WINDOW_H = 12
MIN_GAP = 0.06  # seconds between requests (~16 req/s)
_last = [0.0]


def get(path, params=None):
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    for attempt in range(8):
        wait = MIN_GAP - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                time.sleep(2 ** attempt)
                continue
            if e.code == 404:
                return None
            raise
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed: {url}")


def list_markets(series):
    out = []
    for path in ["/markets", "/historical/markets"]:
        cursor = None
        while True:
            p = {"series_ticker": series, "limit": 1000}
            if path == "/markets":
                p["status"] = "settled"
            if cursor:
                p["cursor"] = cursor
            d = get(path, p) or {}
            ms = d.get("markets", [])
            for m in ms:
                m["_hist"] = path.startswith("/historical")
            out += ms
            cursor = d.get("cursor")
            if not cursor or not ms:
                break
    return out


def ts(s):
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def norm_candle(c):
    def f(block, key):
        b = c.get(block) or {}
        v = b.get(key + "_dollars", b.get(key))
        return float(v) if v not in (None, "") else None
    vol = c.get("volume_fp", c.get("volume"))
    return [c["end_period_ts"], f("yes_bid", "close"), f("yes_ask", "close"),
            f("price", "close"), float(vol) if vol not in (None, "") else 0.0]


def main(series_list):
    os.makedirs("data/kalshi", exist_ok=True)
    for series in series_list:
        fn = f"data/kalshi/{series}.jsonl"
        done = set()
        if os.path.exists(fn):
            with open(fn) as f:
                done = {json.loads(l)["event"] for l in f}
        ms = list_markets(series)
        events = {}
        for m in ms:
            events.setdefault(m["event_ticker"], []).append(m)
        print(f"{series}: {len(ms)} markets, {len(events)} events, "
              f"{len(done)} already done", flush=True)
        with open(fn, "a") as out:
            for i, (ev, mk) in enumerate(sorted(events.items())):
                if ev in done or len(mk) != 2:
                    continue
                if not all(m.get("result") in ("yes", "no") for m in mk):
                    continue  # voided / fair-price settlements
                mk.sort(key=lambda m: m["ticker"])
                m0 = mk[0]
                end = ts(m0["close_time"])
                start = max(ts(m0["open_time"]), end - WINDOW_H * 3600)
                path = (f"/historical/markets/{m0['ticker']}/candlesticks"
                        if m0["_hist"] else
                        f"/series/{series}/markets/{m0['ticker']}/candlesticks")
                d = get(path, {"start_ts": start, "end_ts": end,
                               "period_interval": 1}) or {}
                candles = [norm_candle(c) for c in d.get("candlesticks", [])]
                rec = {
                    "event": ev, "series": series,
                    "tickers": [m["ticker"] for m in mk],
                    "names": [m.get("yes_sub_title") for m in mk],
                    "results": [m["result"] for m in mk],
                    "scheduled": m0.get("occurrence_datetime")
                    or m0.get("expected_expiration_time"),
                    "open_time": m0["open_time"], "close_time": m0["close_time"],
                    "volume": [float(m.get("volume_fp", m.get("volume")) or 0)
                               for m in mk],
                    "rules": m0.get("rules_primary", "")[:300],
                    "candles": candles,  # [ts, bid, ask, last, volume] side 0
                }
                out.write(json.dumps(rec) + "\n")
                if i % 200 == 0:
                    out.flush()
                    print(f"  {series} {i}/{len(events)}", flush=True)
        print(f"{series}: finished", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
