"""Strategy #1: follow in-play shocks, entering as TAKER vs as MAKER.

After the mid moves >= D within W minutes, we want to own the side the move
went towards ("follow" side).
  taker: buy at the follow-side ask one minute later, pay 0.07*p(1-p).
  maker: one minute later, post a bid at the follow-side best bid. Filled if
         the follow-side ask drops to our price within FILL_MIN minutes
         (i.e. the market traded down through us -> adverse selection is
         baked in). Pay maker fee 0.0175*p(1-p). Unfilled = no trade.
Hold to settlement. Results segmented, plus a time-split check.
"""
import sys

import numpy as np
import pandas as pd

import kalshi_analyze as ka

TAKER, MAKER = 0.07, 0.0175
FILL_MIN = 3


def shocks(ms, D, W):
    rows = []
    for m in ms:
        s = m.get("start")
        if s is None:
            continue
        bid, ask, y = m["bid"], m["ask"], m["y"]
        mid = (bid + ask) / 2
        n = len(mid)
        j = s + W
        while j < n - FILL_MIN - 2:
            d = mid[j] - mid[j - W]
            if abs(d) >= D and 0.15 <= mid[j] <= 0.85:
                k = j + 1
                if ask[k] - bid[k] <= ka.MAX_SPREAD:
                    if d > 0:   # follow = YES
                        fb, fa, fy = bid[k], ask[k], y
                        fut_ask = ask[k + 1:k + 1 + FILL_MIN]
                    else:       # follow = NO; NO bid = 1-yes ask, NO ask = 1-yes bid
                        fb, fa, fy = 1 - ask[k], 1 - bid[k], 1 - y
                        fut_ask = 1 - bid[k + 1:k + 1 + FILL_MIN]
                    filled = bool(np.nanmin(fut_ask) <= fb + 1e-9)
                    rows.append(dict(
                        ev=m["event"], series=m["series"], t=m["t0"] + 60 * k,
                        size=abs(d), price=(fb + fa) / 2,
                        phase=(k - s) / max(n - s, 1), totvol=m["totvol"],
                        cumvol=float(m["vol"][:k + 1].sum()), mins_in=k - s,
                        gap=fy - (fb + fa) / 2,
                        taker=fy - fa - TAKER * fa * (1 - fa),
                        filled=filled,
                        maker=(fy - fb - MAKER * fb * (1 - fb)) if filled else np.nan))
                j += 30
            else:
                j += 1
    return pd.DataFrame(rows)


def report(r, label):
    print(f"--- {label}: {len(r)} shocks, {r.ev.nunique()} matches, "
          f"maker fill rate {r.filled.mean():.0%}")
    ka.fmt("follow-side outcome minus mid (no costs)", r.gap, r.ev)
    ka.fmt("TAKER follow, hold to settlement", r.taker, r.ev)
    f = r[r.filled]
    ka.fmt("MAKER follow (filled only), hold", f.maker, f.ev)


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "data/kalshi/*.jsonl"
    ms = ka.load(pat)
    for m in ms:
        m["start"] = ka.match_start(m)
    print(f"{len(ms)} matches\n")
    for D, W in [(0.10, 5), (0.20, 10)]:
        r = shocks(ms, D, W)
        print(f"===== D={D} W={W} =====")
        report(r, "ALL")
        for ser, g in r.groupby("series"):
            report(g, ser)
        r["vol_t"] = pd.qcut(r.totvol, 3, labels=["low vol", "mid vol", "high vol"])
        for v, g in r.groupby("vol_t", observed=True):
            report(g, f"volume tercile: {v}")
        r["ph"] = pd.cut(r.phase, [0, .33, .66, 1.01], labels=["early", "middle", "late"])
        for v, g in r.groupby("ph", observed=True):
            report(g, f"match phase: {v}")
        r["pb"] = pd.cut(r.price, [0, .35, .65, 1], labels=["follow side <35c", "35-65c", ">65c"])
        for v, g in r.groupby("pb", observed=True):
            report(g, f"price: {v}")
        med = r.t.median()
        report(r[r.t < med], "first half of period (in-sample)")
        report(r[r.t >= med], "second half of period (out-of-sample)")
        print()


if __name__ == "__main__":
    main()
