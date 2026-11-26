"""Strategy 5: pre-match Polymarket vs Kalshi.

Before the match prices barely move, so a ~1-minute lag doesn't matter.
If one venue is systematically "soft", bet on it when the two disagree,
hold to settlement.
At T-60/-30/-10 min before the detected start:
  K = Kalshi mid, P = last Polymarket print (must be <= 20 min old).
Bets
  POLY side:   buy on Polymarket the side Polymarket prices LOWER than Kalshi
               (entry = P + POLY_HALF, + Poly taker fee)
  KALSHI side: buy on Kalshi the side Kalshi prices LOWER than Polymarket
               (entry = Kalshi ask, + Kalshi taker fee)
One bet per match (first snapshot that qualifies).
"""
import numpy as np
import pandas as pd

import kalshi_analyze as ka
import strat_xvenue as xv

K_TAKER, P_TAKER, POLY_HALF = 0.07, 0.05, 0.01


def main():
    # rebuild pairs but keep Polymarket print ages
    ms = xv.pairs()
    rows = []
    for m in ms:
        s = m["start"]
        if s is None or s < 70:
            continue
        P = m["P"]  # already NaN where print older than 3 min
        for lead in (60, 30, 10):
            i = s - lead
            k_bid, k_ask = m["bid"][i], m["ask"][i]
            # look back up to 20 minutes for a Polymarket print
            win = P[max(0, i - 20):i + 1]
            pv = win[~np.isnan(win)]
            if len(pv) == 0 or k_ask - k_bid > 0.04:
                continue
            rows.append(dict(ev=m["event"], series=m["series"], lead=lead, y=m["y"],
                             K=(k_bid + k_ask) / 2, kb=k_bid, ka=k_ask, P=pv[-1]))
    d = pd.DataFrame(rows)
    print(f"{d.ev.nunique()} matches with pre-match quotes on both venues\n")
    for lead in (60, 30, 10):
        g = d[d.lead == lead]
        print(f"T-{lead}m: n={len(g)}  log-loss Kalshi {xv_ll(g.K, g.y):.4f}  "
              f"Poly {xv_ll(g.P, g.y):.4f}  mean |gap| {100*(g.P-g.K).abs().mean():.2f}c")
    print()
    d = d.sort_values(["ev", "lead"], ascending=[True, False])
    for thr in [0.02, 0.03, 0.05, 0.08]:
        poly, kal = [], []
        for ev, g in d.groupby("ev"):
            q = g[(g.P - g.K).abs() >= thr]
            if not len(q):
                continue
            r = q.iloc[0]
            # Poly side: Poly is lower on YES -> buy YES on Poly
            if r.P < r.K:
                ent, yy = r.P + POLY_HALF, r.y
            else:
                ent, yy = 1 - r.P + POLY_HALF, 1 - r.y
            poly.append((ev, r.series, yy - ent - P_TAKER * ent * (1 - ent)))
            # Kalshi side: Kalshi is lower on YES -> buy YES on Kalshi
            if r.K < r.P:
                ent, yy = r.ka, r.y
            else:
                ent, yy = 1 - r.kb, 1 - r.y
            kal.append((ev, r.series, yy - ent - K_TAKER * ent * (1 - ent)))
        p = pd.DataFrame(poly, columns=["ev", "series", "pnl"])
        k = pd.DataFrame(kal, columns=["ev", "series", "pnl"])
        print(f"|gap| >= {100*thr:.0f}c:")
        ka.fmt("  bet POLY side cheaper than Kalshi", p.pnl, p.ev)
        ka.fmt("  bet KALSHI side cheaper than Poly", k.pnl, k.ev)
        for ser, g in p.groupby("series"):
            ka.fmt(f"     POLY [{ser}]", g.pnl, g.ev)


def xv_ll(p, y):
    p = np.clip(p, 0.01, 0.99)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


if __name__ == "__main__":
    main()
