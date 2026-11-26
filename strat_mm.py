"""Strategy #2: market making — how much does the passive side earn?

For every in-play trade we know the price and which side was the TAKER, so
the MAKER took the other side. Maker P&L per contract, marked at:
  - the mid h minutes later (1, 5, 15), and
  - final settlement,
minus the maker fee 0.0175*p*(1-p). If this is positive across all fills,
a market maker who gets average fills makes money; the 1-min mark vs
settlement tells us how much is eaten by adverse selection.
Contract-weighted averages; errors bootstrapped by match.
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd

import kalshi_analyze as ka

MAKER = 0.0175


def iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def main():
    ms = {m["event"]: m for m in ka.load("data/kalshi/KXATPMATCH.jsonl")}
    rows = []
    for line in open("data/kalshi/trades_KXATPMATCH.jsonl"):
        r = json.loads(line)
        m = ms.get(r["event"])
        if m is None or not r["trades"]:
            continue
        mid = (m["bid"] + m["ask"]) / 2
        n = len(mid)
        for created, p, c, taker in r["trades"]:
            t = iso(created)
            i = int((t - m["t0"]) // 60)  # minute index containing the trade
            if i < 1 or i >= n:
                continue
            m0 = mid[i - 1]  # mid at end of previous minute (pre-trade)
            if not (0.05 < m0 < 0.95):
                continue
            sgn = 1 if taker == "no" else -1  # +1: maker bought YES
            row = dict(ev=r["event"], c=c, p=p, m0=m0, sgn=sgn,
                       half_spread=sgn * (m0 - p),
                       settle=sgn * (r["y"] - p) - MAKER * p * (1 - p))
            for h in (1, 5, 15):
                j = min(i + h, n - 1)
                row[f"h{h}"] = sgn * (mid[j] - p) - MAKER * p * (1 - p)
            rows.append(row)
    d = pd.DataFrame(rows)
    print(f"{d.ev.nunique()} matches, {len(d)} trades, "
          f"{d.c.sum():,.0f} contracts\n")

    def wfmt(label, col, g):
        # contract-weighted mean, bootstrapped by match
        gs = g.assign(w=g[col] * g.c).groupby("ev")[["w", "c"]].sum()
        w, c = gs.w.to_numpy(), gs.c.to_numpy()
        est = w.sum() / c.sum()
        rng = np.random.default_rng(0)
        bs = [(lambda i: w[i].sum() / c[i].sum())(rng.integers(0, len(w), len(w)))
              for _ in range(500)]
        se = np.std(bs)
        print(f"  {label:42s} {100*est:+6.2f}c ± {196*se:4.2f}c  t={est/se:+5.2f}")

    def block(g, title):
        print(f"--- {title}: {len(g)} trades, {g.c.sum():,.0f} contracts")
        wfmt("spread captured vs prior mid (no fee)", "half_spread", g)
        for h in (1, 5, 15):
            wfmt(f"maker P&L marked at mid +{h}m (after fee)", f"h{h}", g)
        wfmt("maker P&L held to settlement (after fee)", "settle", g)

    block(d, "ALL in-play trades")
    d["size"] = pd.cut(d.c, [0, 10, 100, 1000, 1e12],
                       labels=["<10", "10-100", "100-1000", ">1000"])
    for s, g in d.groupby("size", observed=True):
        block(g, f"taker order size {s} contracts")
    d["pb"] = pd.cut(d.m0, [0, .2, .4, .6, .8, 1])
    for s, g in d.groupby("pb", observed=True):
        block(g, f"mid in {s}")


if __name__ == "__main__":
    main()
