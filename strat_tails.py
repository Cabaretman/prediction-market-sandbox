"""Kalshi-only: favourite/longshot bias at the extremes, where fees are tiny.

Kalshi taker fee = 0.07*p*(1-p): ~0.3c at 5c/95c vs 1.75c at 50c.
In-play, sample each match every 10 minutes after the detected start; at
each sample where one side is a big favourite, test:
  BUY FAV  : buy the favourite at its ask (+fee), hold to settlement
  BUY DOG  : buy the longshot at its ask (+fee), hold to settlement
Also pre-match (last quote before start). Split by time (first/second half)
to check stability. One sample per match per bucket to limit clustering.
"""
import numpy as np
import pandas as pd

import kalshi_analyze as ka

TAKER = 0.07
BUCKETS = [0.80, 0.85, 0.90, 0.94, 0.97, 0.99]


def main():
    ms = ka.load("data/kalshi/KX*MATCH.jsonl")
    rows = []
    for m in ms:
        s = ka.match_start(m)
        if s is None:
            continue
        bid, ask, y = m["bid"], m["ask"], m["y"]
        n = len(bid)
        idxs = [("pre", s - 1)] + [("inplay", i) for i in range(s + 10, n - 3, 10)]
        for phase, i in idxs:
            if i < 0 or np.isnan(bid[i]) or ask[i] - bid[i] > 0.04:
                continue
            mid = (bid[i] + ask[i]) / 2
            if mid >= 0.5:   # side 0 (YES) is favourite
                fav_ask, dog_ask, fav_y = ask[i], 1 - bid[i], y
            else:
                fav_ask, dog_ask, fav_y = 1 - bid[i], ask[i], 1 - y
            fmid = max(mid, 1 - mid)
            rows.append(dict(ev=m["event"], series=m["series"], t=m["t0"] + 60 * i,
                             phase=phase, fmid=fmid,
                             fav=fav_y - fav_ask - TAKER * fav_ask * (1 - fav_ask),
                             dog=(1 - fav_y) - dog_ask - TAKER * dog_ask * (1 - dog_ask),
                             gap=fav_y - fmid))
    d = pd.DataFrame(rows)
    d["b"] = pd.cut(d.fmid, [0.5] + BUCKETS + [1.0])
    med = d.t.median()
    for phase in ["pre", "inplay"]:
        print(f"===== {phase} =====")
        for b, g in d[d.phase == phase].groupby("b", observed=True):
            g = g.groupby("ev").head(1)  # one sample per match per bucket
            print(f" favourite priced {b}: n={len(g)}  fav win {g.gap.mean()*100+g.fmid.mean()*100:.1f}% "
                  f"vs price {g.fmid.mean()*100:.1f}c")
            ka.fmt("   BUY FAVOURITE at ask, hold", g.fav, g.ev)
            ka.fmt("   BUY LONGSHOT at ask, hold", g.dog, g.ev)
            ka.fmt("   fav, first half of period", g[g.t < med].fav, g[g.t < med].ev)
            ka.fmt("   fav, second half of period", g[g.t >= med].fav, g[g.t >= med].ev)
        print()


if __name__ == "__main__":
    main()
