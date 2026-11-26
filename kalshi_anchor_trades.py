"""For every set-1 market in trades_side_<SERIES>.jsonl, download the trade
tape of the corresponding MATCH-WINNER market over the same window.
Output: data/kalshi/anchor_trades_<SERIES>.jsonl
Usage: python3 kalshi_anchor_trades.py KXATPSETWINNER
"""
import json
import os
import sys

from kalshi_collect import get, ts

CUTOFF = ts("2026-09-26T00:00:00Z")
MATCH_SERIES = {"KXATPSETWINNER": "KXATPMATCH", "KXWTASETWINNER": "KXWTAMATCH"}


def main(series):
    mser = MATCH_SERIES[series]
    match = {}
    for line in open(f"data/kalshi/{mser}.jsonl"):
        r = json.loads(line)
        match[r["event"].split("-", 1)[1]] = r
    side = {}
    for line in open(f"data/kalshi/side_{series}.jsonl"):
        r = json.loads(line)
        side[r["event"]] = r
    fn = f"data/kalshi/anchor_trades_{series}.jsonl"
    done = {json.loads(l)["event"] for l in open(fn)} if os.path.exists(fn) else set()
    todo = [json.loads(l)["event"] for l in open(f"data/kalshi/trades_side_{series}.jsonl")]
    n = 0
    with open(fn, "a") as out:
        for ev in todo:
            if ev in done:
                continue
            code = ev.split("-", 1)[1].rsplit("-", 1)[0]
            m = match.get(code)
            s = side.get(ev)
            if m is None or s is None or not s["markets"][0]["candles"]:
                continue
            t0 = s["markets"][0]["candles"][0][0] - 60
            t1 = ts(s["markets"][0]["close_time"])
            tk = m["tickers"][0]
            path = "/historical/trades" if ts(m["close_time"]) < CUTOFF else "/markets/trades"
            trades, cur = [], None
            while True:
                p = {"ticker": tk, "min_ts": t0, "max_ts": t1, "limit": 1000}
                if cur:
                    p["cursor"] = cur
                d = get(path, p) or {}
                for t in d.get("trades", []):
                    trades.append([t["created_time"],
                                   float(t.get("yes_price_dollars") or t.get("yes_price")),
                                   float(t.get("count_fp") or t.get("count")),
                                   t["taker_side"]])
                cur = d.get("cursor")
                if not cur or not d.get("trades"):
                    break
            out.write(json.dumps({"event": ev, "match_event": m["event"],
                                  "match_ticker": tk, "match_name0": m["names"][0],
                                  "trades": trades}) + "\n")
            out.flush()
            n += 1
            if n % 50 == 0:
                print(f"{series} anchors {n}", flush=True)
    print(f"{series} anchors done ({n} new)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
