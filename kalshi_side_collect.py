"""Download settled Kalshi tennis SIDE markets (exact score, set winner) with
1-min candles for EVERY market in each event.
Output: data/kalshi/side_<SERIES>.jsonl, one line per event. Resumable.
"""
import json, os, sys
from kalshi_collect import get, ts, norm_candle

WINDOW_H = 12
CUTOFF = ts("2026-09-26T00:00:00Z")

def list_all(series):
    out = []
    for path in ["/markets", "/historical/markets"]:
        cursor = None
        while True:
            p = {"series_ticker": series, "limit": 1000}
            if path == "/markets": p["status"] = "settled"
            if cursor: p["cursor"] = cursor
            d = get(path, p) or {}
            for m in d.get("markets", []):
                m["_hist"] = path.startswith("/historical"); out.append(m)
            cursor = d.get("cursor")
            if not cursor or not d.get("markets"): break
    return out

def main(series, only_set1=False):
    fn = f"data/kalshi/side_{series}.jsonl"
    done = set()
    if os.path.exists(fn):
        done = {json.loads(l)["event"] for l in open(fn)}
    ev = {}
    for m in list_all(series):
        if m.get("result") not in ("yes", "no"): continue
        if only_set1 and not m["event_ticker"].endswith("-1"): continue
        ev.setdefault(m["event_ticker"], []).append(m)
    print(f"{series}: {len(ev)} events", flush=True)
    with open(fn, "a") as out:
        for i, (e, mk) in enumerate(sorted(ev.items())):
            if e in done: continue
            recs = []
            for m in sorted(mk, key=lambda m: m["ticker"]):
                end = ts(m["close_time"]); start = max(ts(m["open_time"]), end - WINDOW_H * 3600)
                path = (f"/historical/markets/{m['ticker']}/candlesticks" if m["_hist"]
                        else f"/series/{series}/markets/{m['ticker']}/candlesticks")
                d = get(path, {"start_ts": start, "end_ts": end, "period_interval": 1}) or {}
                recs.append({"ticker": m["ticker"], "title": m.get("yes_sub_title") or m.get("title"),
                             "result": m["result"], "close_time": m["close_time"],
                             "volume": float(m.get("volume_fp") or m.get("volume") or 0),
                             "candles": [norm_candle(c) for c in d.get("candlesticks", [])]})
            out.write(json.dumps({"event": e, "series": series, "markets": recs}) + "\n")
            if i % 100 == 0:
                out.flush(); print(f"  {series} {i}/{len(ev)}", flush=True)
    print(f"{series}: finished", flush=True)

if __name__ == "__main__":
    s = sys.argv[1]
    main(s, only_set1=len(sys.argv) > 2 and sys.argv[2] == "set1")
