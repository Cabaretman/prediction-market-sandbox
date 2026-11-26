"""How much traded at favourite prices 95-97c in the hour before start?"""
import json, glob
import numpy as np, pandas as pd
import kalshi_analyze as ka
from kalshi_collect import get, ts
CUTOFF = ts("2026-09-26T00:00:00Z")
e = pd.read_csv("data/fav9597_entries.csv")
want = set(e.ev)
meta = {}
for fn in glob.glob("data/kalshi/KX*MATCH.jsonl"):
    for l in open(fn):
        r = json.loads(l)
        if r["event"] in want: meta[r["event"]] = r
ms = [m for m in ka.load("data/kalshi/KX*MATCH.jsonl") if m["event"] in want]
rows = []
for m in ms:
    s = ka.match_start(m); r = meta[m["event"]]
    t1 = m["t0"] + 60 * s; t0 = t1 - 3600
    path = "/historical/trades" if ts(r["close_time"]) < CUTOFF else "/markets/trades"
    tr, cur = [], None
    while True:
        p = {"ticker": r["tickers"][0], "min_ts": t0, "max_ts": t1, "limit": 1000}
        if cur: p["cursor"] = cur
        d = get(path, p) or {}
        tr += d.get("trades", []); cur = d.get("cursor")
        if not cur or not d.get("trades"): break
    fav_c = buy_fav = dollars = 0.0
    for t in tr:
        yp = float(t.get("yes_price_dollars") or t.get("yes_price")); c = float(t.get("count_fp") or t.get("count"))
        fp = yp if yp > 0.5 else 1 - yp
        if 0.95 <= fp <= 0.97:
            fav_c += c; dollars += c * fp
            fav_is_yes = yp > 0.5
            if (t["taker_side"] == "yes") == fav_is_yes: buy_fav += c
    rows.append(dict(ev=m["event"], series=m["series"], contracts=fav_c, taker_buys_fav=buy_fav, dollars=dollars))
d = pd.DataFrame(rows)
d.to_csv("data/fav9597_liquidity.csv", index=False)
print(f"{len(d)} matches; contracts traded at fav price 95-97c in the last hour pre-start:")
print(d.contracts.describe(percentiles=[.1,.25,.5,.75,.9]).round(0).to_string())
print("\n'taker bought the favourite' contracts (= ask-side liquidity actually consumed):")
print(d.taker_buys_fav.describe(percentiles=[.1,.25,.5,.75,.9]).round(0).to_string())
print("\nmedian by tier (taker-bought-fav contracts):")
print(d.groupby("series").taker_buys_fav.median().round(0).to_string())
