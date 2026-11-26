"""Kalshi-only market making: WHEN is it profitable to be the maker?

For every in-play trade the maker took the opposite side of the taker.
Maker P&L (after 0.0175*p(1-p) maker fee) marked at the mid 1 and 5 minutes
later, and at settlement. We split by market state observable BEFORE the
trade (previous minutes only):
  calm5   : |mid move| over the previous 5 minutes
  spread  : quoted spread in the previous minute
  level   : price level (mid)
  flow5   : contracts traded in the previous 5 minutes
Then: choose "quote here" cells on a TRAIN half of matches, test on the other
half (split by match, so no leakage).
Usage: python3 strat_mm_state.py KXATPMATCH [KXWTAMATCH ...]
"""
import json
import sys
import zlib
from datetime import datetime

import numpy as np
import pandas as pd

import kalshi_analyze as ka

MAKER = 0.0175


def rows_for(series):
    ms = {m["event"]: m for m in ka.load(f"data/kalshi/{series}.jsonl")}
    out = []
    for line in open(f"data/kalshi/trades_{series}.jsonl"):
        r = json.loads(line)
        m = ms.get(r["event"])
        if m is None:
            continue
        bid, ask, vol = m["bid"], m["ask"], m["vol"]
        mid = (bid + ask) / 2
        n = len(mid)
        for created, p, c, taker in r["trades"]:
            t = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
            i = int((t - m["t0"]) // 60)
            if i < 7 or i >= n:
                continue
            m0 = mid[i - 1]
            if not (0.05 < m0 < 0.95):
                continue
            sgn = 1 if taker == "no" else -1  # +1: maker bought YES
            fee = MAKER * p * (1 - p)
            out.append(dict(
                series=series, ev=r["event"], c=c,
                calm5=abs(mid[i - 1] - mid[i - 6]),
                spread=ask[i - 1] - bid[i - 1], level=m0,
                flow5=vol[i - 6:i - 1].sum(),
                h1=sgn * (mid[min(i + 1, n - 1)] - p) - fee,
                h5=sgn * (mid[min(i + 5, n - 1)] - p) - fee,
                settle=sgn * (r["y"] - p) - fee))
    return out


def wmean(g, col):
    return (g[col] * g.c).sum() / g.c.sum()


def wfmt(label, g, col):
    gs = g.assign(w=g[col] * g.c).groupby("ev")[["w", "c"]].sum()
    w, c = gs.w.to_numpy(), gs.c.to_numpy()
    est = w.sum() / c.sum()
    rng = np.random.default_rng(0)
    bs = [w[i].sum() / c[i].sum() for i in
          (rng.integers(0, len(w), len(w)) for _ in range(400))]
    se = np.std(bs)
    print(f"  {label:44s} {100*est:+6.2f}c ± {196*se:4.2f}c  t={est/se if se else 0:+5.2f}"
          f"   ({g.c.sum()/1e6:6.1f}M contracts)")


def main(series_list):
    d = pd.DataFrame([r for s in series_list for r in rows_for(s)])
    print(f"{d.ev.nunique()} matches, {len(d):,} trades\n")
    d["calm_b"] = pd.cut(d.calm5, [-1, 0.005, 0.02, 0.05, 1],
                         labels=["flat", "<2c", "2-5c", ">5c"])
    d["spr_b"] = pd.cut(d.spread, [-1, 0.0101, 0.0201, 1], labels=["1c", "2c", "3c+"])
    d["lvl_b"] = pd.cut(d.level, [0, .2, .4, .6, .8, 1])
    d["flow_b"] = pd.qcut(d.flow5.rank(method="first"), 3, labels=["quiet", "normal", "busy"])
    for col, name in [("calm_b", "recent 5-min move"), ("spr_b", "spread"),
                      ("lvl_b", "price level"), ("flow_b", "recent flow")]:
        print(f"== by {name} ==")
        for v, g in d.groupby(col, observed=True):
            wfmt(f"{v}: maker +1m", g, "h1")
            wfmt(f"{v}: maker +5m", g, "h5")
        print()

    # train/test split by match
    d["half"] = d.ev.map(lambda e: zlib.crc32(e.encode()) % 2)
    keys = ["calm_b", "spr_b", "flow_b"]
    tr, te = d[d.half == 0], d[d.half == 1]
    cells = tr.groupby(keys, observed=True).apply(
        lambda g: pd.Series({"h5": wmean(g, "h5"), "c": g.c.sum()}), include_groups=False)
    good = cells[(cells.h5 > 0.002) & (cells.c > 2e5)].index
    print(f"== Rule chosen on TRAIN half: quote only in {len(good)} of {len(cells)} "
          f"(move x spread x flow) cells ==")
    for g_ in good:
        print("    ", g_)
    sel = te.set_index(keys).index.isin(good)
    print("TEST half results:")
    wfmt("all trades (always quoting)", te, "h5")
    wfmt("rule: quote only in chosen cells, +1m", te[sel], "h1")
    wfmt("rule: quote only in chosen cells, +5m", te[sel], "h5")
    wfmt("rule: chosen cells, to settlement", te[sel], "settle")
    wfmt("outside chosen cells, +5m", te[~sel], "h5")


if __name__ == "__main__":
    main(sys.argv[1:] or ["KXATPMATCH"])
