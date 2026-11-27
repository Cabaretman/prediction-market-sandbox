"""Trade tape (with taker side) for a random sample of settled Kalshi politics
markets (volume >= 5000). Used to measure who wins at settlement: takers vs
makers, by price, market age, side, market type.

Output: data/politics/trades.jsonl  {ticker, trades: [[ts, yes_price, count, taker_side], ...]}
Usage: python3 politics_trades.py N
"""
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

from kalshi_collect import get, ts

CUTOFF = ts("2026-09-26T00:00:00Z")
MAX_PAGES = 15


def fetch(m):
    t_end = ts(m["close"])
    path = "/historical/trades" if t_end < CUTOFF else "/markets/trades"
    out, cursor = [], None
    for _ in range(MAX_PAGES):
        p = {"ticker": m["ticker"], "limit": 1000}
        if cursor:
            p["cursor"] = cursor
        d = get(path, p) or {}
        for t in d.get("trades", []):
            out.append([round(ts(t["created_time"])),
                        float(t.get("yes_price_dollars") or t.get("yes_price") or 0),
                        float(t.get("count_fp") or t.get("count") or 0),
                        t.get("taker_side")])
        cursor = d.get("cursor")
        if not cursor or not d.get("trades"):
            break
    return {"ticker": m["ticker"], "trades": out}


def main(n):
    ms = [json.loads(l) for l in open("data/politics/markets.jsonl")]
    seen, uniq = set(), []
    for m in ms:
        if m["ticker"] not in seen and m["volume"] >= 5000 and m["open"] and m["close"]:
            seen.add(m["ticker"])
            uniq.append(m)
    random.Random(3).shuffle(uniq)
    fn = "data/politics/trades.jsonl"
    done = {json.loads(l)["ticker"] for l in open(fn)} if os.path.exists(fn) else set()
    todo = [m for m in uniq[:n] if m["ticker"] not in done]
    print(f"{len(todo)} to fetch", flush=True)
    with open(fn, "a") as out, ThreadPoolExecutor(6) as ex:
        for i, r in enumerate(ex.map(fetch, todo)):
            out.write(json.dumps(r) + "\n")
            if i % 100 == 0:
                out.flush()
                print(f"  {i}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
