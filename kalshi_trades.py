"""Download in-play trades (with taker side) for a random sample of Kalshi
tennis matches. Used to measure adverse selection for market making.

Output: data/kalshi/trades_<SERIES>.jsonl
Usage: python3 kalshi_trades.py KXATPMATCH 300
"""
import json
import os
import random
import sys

import kalshi_analyze as ka
from kalshi_collect import get, ts

CUTOFF = ts("2026-09-26T00:00:00Z")  # older trades live under /historical


def main(series, n):
    recs = {}
    with open(f"data/kalshi/{series}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            recs[r["event"]] = r
    ms = {m["event"]: m for m in ka.load(f"data/kalshi/{series}.jsonl")}
    keys = sorted(ms)
    random.Random(1).shuffle(keys)
    fn = f"data/kalshi/trades_{series}.jsonl"
    done = set()
    if os.path.exists(fn):
        with open(fn) as f:
            done = {json.loads(l)["event"] for l in f}
    saved = len(done)
    with open(fn, "a") as out:
        for ev in keys:
            if saved >= n:
                break
            if ev in done:
                continue
            m, r = ms[ev], recs[ev]
            s = ka.match_start(m)
            if s is None:
                continue
            t_start = m["t0"] + 60 * (s - 10)  # a bit before detected start
            t_end = ts(r["close_time"])
            tk = r["tickers"][0]
            path = "/historical/trades" if t_end < CUTOFF else "/markets/trades"
            trades, cursor = [], None
            while True:
                p = {"ticker": tk, "min_ts": t_start, "max_ts": t_end, "limit": 1000}
                if cursor:
                    p["cursor"] = cursor
                d = get(path, p) or {}
                for t in d.get("trades", []):
                    trades.append([
                        t["created_time"],
                        float(t.get("yes_price_dollars") or t.get("yes_price")),
                        float(t.get("count_fp") or t.get("count")),
                        t["taker_side"]])
                cursor = d.get("cursor")
                if not cursor or not d.get("trades"):
                    break
            out.write(json.dumps({"event": ev, "ticker": tk, "y": m["y"],
                                  "trades": trades}) + "\n")
            out.flush()
            saved += 1
            print(f"{saved}/{n} {ev} {len(trades)} trades", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
