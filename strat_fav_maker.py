"""Kalshi-only: rest BIDS on the heavy favourite (maker), capturing the
longshot bias without paying spread or taker fee.

Uses the trade tape: every trade where the MAKER ended up BUYING the side
priced >= LO (i.e. a taker sold the favourite / bought the longshot into a
resting favourite bid). Maker P&L after 0.0175*p(1-p) fee, at +1m, +5m and
settlement. Also the reverse (maker SELLING the favourite) for contrast.
"""
import json, sys
from datetime import datetime
import numpy as np, pandas as pd
import kalshi_analyze as ka
from strat_mm_state import wfmt

MAKER = 0.0175

def rows(series):
    ms = {m["event"]: m for m in ka.load(f"data/kalshi/{series}.jsonl")}
    out = []
    for line in open(f"data/kalshi/trades_{series}.jsonl"):
        r = json.loads(line); m = ms.get(r["event"])
        if m is None: continue
        mid = (m["bid"] + m["ask"]) / 2; n = len(mid)
        for created, p, c, taker in r["trades"]:
            t = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
            i = int((t - m["t0"]) // 60)
            if i < 2 or i >= n: continue
            maker_yes = taker == "no"           # maker bought YES at p
            # express everything from the maker's held side
            px = p if maker_yes else 1 - p        # price maker paid for its side
            yy = r["y"] if maker_yes else 1 - r["y"]
            mk = lambda j: (mid[min(j, n - 1)] if maker_yes else 1 - mid[min(j, n - 1)])
            fee = MAKER * px * (1 - px)
            out.append(dict(series=series, ev=r["event"], c=c, px=px,
                            h1=mk(i + 1) - px - fee, h5=mk(i + 5) - px - fee,
                            settle=yy - px - fee))
    return out

def main(series_list):
    d = pd.DataFrame([r for s in series_list for r in rows(s)])
    print(f"{d.ev.nunique()} matches, {len(d):,} maker fills\n")
    for lo, hi in [(0.80, 0.90), (0.90, 0.95), (0.95, 0.99), (0.80, 0.99)]:
        g = d[(d.px >= lo) & (d.px < hi)]
        print(f"-- maker BOUGHT the side priced {lo:.2f}-{hi:.2f} (favourite bid filled)")
        wfmt("+1m", g, "h1"); wfmt("+5m", g, "h5"); wfmt("settlement", g, "settle")
        g2 = d[(d.px > 1 - hi) & (d.px <= 1 - lo)]
        print(f"-- maker BOUGHT the side priced {1-hi:.2f}-{1-lo:.2f} (longshot bid filled)")
        wfmt("+1m", g2, "h1"); wfmt("+5m", g2, "h5"); wfmt("settlement", g2, "settle")
        print()

if __name__ == "__main__":
    main(sys.argv[1:] or ["KXATPMATCH"])
