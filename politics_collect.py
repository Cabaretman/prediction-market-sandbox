"""Politics/Elections on Kalshi.
step 'list'   : catalogue every settled market in all Politics + Elections
                series -> data/politics/markets.jsonl
step 'candles': daily candles over each market's life for markets with
                volume >= MIN_VOL -> data/politics/candles.jsonl
Usage: python3 politics_collect.py list|candles
"""
import json
import os
import sys

from kalshi_collect import get, ts, norm_candle

OUT = "data/politics"
MIN_VOL = 5000
CUTOFF = ts("2026-09-26T00:00:00Z")


def series_list():
    out = []
    for cat in ["Politics", "Elections"]:
        d = get("/series", {"category": cat}) or {}
        for s in d.get("series", []):
            out.append((s["ticker"], cat, s.get("tags") or [], s.get("title")))
    return out


def list_step():
    os.makedirs(OUT, exist_ok=True)
    fn = f"{OUT}/markets.jsonl"
    done = set()
    if os.path.exists(fn):
        done = {json.loads(l)["series"] for l in open(fn)}
    ss = series_list()
    print(f"{len(ss)} series", flush=True)
    with open(fn, "a") as out:
        for i, (st, cat, tags, stitle) in enumerate(ss):
            if st in done:
                continue
            for path in ["/markets", "/historical/markets"]:
                cursor = None
                while True:
                    p = {"series_ticker": st, "limit": 1000}
                    if path == "/markets":
                        p["status"] = "settled"
                    if cursor:
                        p["cursor"] = cursor
                    d = get(path, p) or {}
                    for m in d.get("markets", []):
                        if m.get("result") not in ("yes", "no"):
                            continue
                        out.write(json.dumps(dict(
                            series=st, cat=cat, tags=tags, series_title=stitle,
                            ticker=m["ticker"], event=m["event_ticker"],
                            title=m.get("title"), yes_sub=m.get("yes_sub_title"),
                            open=m.get("open_time"), close=m.get("close_time"),
                            result=m["result"], hist=path.startswith("/historical"),
                            volume=float(m.get("volume_fp") or m.get("volume") or 0))) + "\n")
                    cursor = d.get("cursor")
                    if not cursor or not d.get("markets"):
                        break
            if i % 200 == 0:
                out.flush()
                print(f"  series {i}/{len(ss)}", flush=True)
    print("list done", flush=True)


def candles_step():
    ms = [json.loads(l) for l in open(f"{OUT}/markets.jsonl")]
    ms = [m for m in ms if m["volume"] >= MIN_VOL and m["open"] and m["close"]]
    fn = f"{OUT}/candles.jsonl"
    done = {json.loads(l)["ticker"] for l in open(fn)} if os.path.exists(fn) else set()
    print(f"{len(ms)} markets with volume >= {MIN_VOL}", flush=True)
    with open(fn, "a") as out:
        for i, m in enumerate(ms):
            if m["ticker"] in done:
                continue
            path = (f"/historical/markets/{m['ticker']}/candlesticks" if m["hist"]
                    else f"/series/{m['series']}/markets/{m['ticker']}/candlesticks")
            d = get(path, {"start_ts": ts(m["open"]), "end_ts": ts(m["close"]),
                           "period_interval": 1440}) or {}
            out.write(json.dumps({"ticker": m["ticker"],
                                  "candles": [norm_candle(c) for c in d.get("candlesticks", [])]}) + "\n")
            if i % 500 == 0:
                out.flush()
                print(f"  candles {i}/{len(ms)}", flush=True)
    print("candles done", flush=True)


if __name__ == "__main__":
    {"list": list_step, "candles": candles_step}[sys.argv[1]]()
