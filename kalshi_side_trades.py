"""Trades for a random sample of SET-1 winner events (side market), window =
the side market's candle window. Output data/kalshi/trades_side_<SERIES>.jsonl"""
import json, os, random, sys
from kalshi_collect import get, ts
CUTOFF = ts("2026-09-26T00:00:00Z")

def main(series, n):
    recs = [json.loads(l) for l in open(f"data/kalshi/side_{series}.jsonl")]
    random.Random(2).shuffle(recs)
    fn = f"data/kalshi/trades_side_{series}.jsonl"
    done = {json.loads(l)["event"] for l in open(fn)} if os.path.exists(fn) else set()
    saved = len(done)
    with open(fn, "a") as out:
        for r in recs:
            if saved >= n: break
            if r["event"] in done or len(r["markets"]) != 2: continue
            mk = r["markets"][0]
            if not mk["candles"]: continue
            t0, t1 = mk["candles"][0][0] - 60, ts(mk["close_time"])
            path = "/historical/trades" if t1 < CUTOFF else "/markets/trades"
            trades, cur = [], None
            while True:
                p = {"ticker": mk["ticker"], "min_ts": t0, "max_ts": t1, "limit": 1000}
                if cur: p["cursor"] = cur
                d = get(path, p) or {}
                for t in d.get("trades", []):
                    trades.append([t["created_time"], float(t.get("yes_price_dollars") or t.get("yes_price")),
                                   float(t.get("count_fp") or t.get("count")), t["taker_side"]])
                cur = d.get("cursor")
                if not cur or not d.get("trades"): break
            out.write(json.dumps({"event": r["event"], "ticker": mk["ticker"],
                                  "y": 1.0 if mk["result"] == "yes" else 0.0, "trades": trades}) + "\n")
            out.flush(); saved += 1
            if saved % 25 == 0: print(f"{saved}/{n}", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
