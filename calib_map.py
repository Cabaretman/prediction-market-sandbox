"""Full Kalshi tennis mispricing map: every price range, pre-match and in-play.

For each match:
  PRE    : snapshot at T-10 min before detected start (spread <= 4c)
  INPLAY : snapshots every 15 min after start (spread <= 4c), at most one
           per match per price bin (to limit clustering)
Both sides of each snapshot are rows (buying side A = selling side B).
Per price bin: win rate vs price, P&L of buying that side at the ASK
(+0.07*p(1-p) fee), held to settlement; early/late half; tier groups.
Bins are 'discovered' on the EARLY half; the LATE half is the check.
"""
import sys

import numpy as np
import pandas as pd

import kalshi_analyze as ka

TK = 0.07
BINS = [0.0, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
        0.93, 0.95, 0.97, 1.0]
TIER = {"KXATPMATCH": "main", "KXWTAMATCH": "main",
        "KXATPCHALLENGERMATCH": "challenger", "KXWTACHALLENGERMATCH": "challenger",
        "KXITFMATCH": "itf", "KXITFWMATCH": "itf"}


def snapshots():
    rows = []
    for m in ka.load("data/kalshi/KX*MATCH.jsonl"):
        s = ka.match_start(m)
        if s is None:
            continue
        bid, ask, y = m["bid"], m["ask"], m["y"]
        n = len(bid)
        pts = [("pre", s - 10)] + [("inplay", i) for i in range(s + 15, n - 3, 15)]
        for phase, i in pts:
            if i < 0 or np.isnan(bid[i]) or np.isnan(ask[i]) or ask[i] - bid[i] > 0.04:
                continue
            mid = (bid[i] + ask[i]) / 2
            for side in (0, 1):
                p = mid if side == 0 else 1 - mid
                a = ask[i] if side == 0 else 1 - bid[i]
                yy = y if side == 0 else 1 - y
                rows.append((m["event"], TIER.get(m["series"], "other"), m["t0"], phase,
                             p, a, yy, yy - a - TK * a * (1 - a)))
    d = pd.DataFrame(rows, columns=["ev", "tier", "t", "phase", "p", "ask", "y", "pnl"])
    d["bin"] = pd.cut(d.p, BINS)
    # one in-play sample per match per bin
    d = pd.concat([d[d.phase == "pre"],
                   d[d.phase == "inplay"].groupby(["ev", "bin"], observed=True).head(1)])
    return d


def tstat(x):
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0


def main():
    d = snapshots()
    med = d.t.median()
    d["half"] = np.where(d.t < med, "early", "late")
    print(f"{d.ev.nunique()} matches; tiers: {d.groupby('tier').ev.nunique().to_dict()}\n")
    for phase in ["pre", "inplay"]:
        g0 = d[d.phase == phase]
        print(f"===== {phase.upper()} =====")
        print(f"{'price bin':14s} {'n':>6s} {'win':>7s} {'price':>7s} {'gap':>6s} | "
              f"{'P&L buy@ask':>12s} {'t':>6s} | {'early':>14s} {'late':>14s} | "
              f"{'main':>7s} {'chall':>7s} {'itf':>7s}")
        for b, g in g0.groupby("bin", observed=True):
            e, l = g[g.half == "early"], g[g.half == "late"]
            tiers = [g[g.tier == k].pnl.mean() * 100 if (g.tier == k).sum() > 20 else np.nan
                     for k in ["main", "challenger", "itf"]]
            print(f"{str(b):14s} {len(g):6d} {g.y.mean():7.3f} {g.p.mean():7.3f} "
                  f"{100*(g.y.mean()-g.p.mean()):+5.1f}c | {100*g.pnl.mean():+10.2f}c {tstat(g.pnl):+6.2f} | "
                  f"{100*e.pnl.mean():+6.2f}c(t{tstat(e.pnl):+4.1f}) {100*l.pnl.mean():+6.2f}c(t{tstat(l.pnl):+4.1f}) | "
                  + " ".join(f"{x:+6.2f}c" if not np.isnan(x) else "    n/a" for x in tiers))
        print()
    # discovery on early half, check on late half
    print("===== Discovery (EARLY half, t>2.5, n>=100) -> Confirmation (LATE half) =====")
    for phase in ["pre", "inplay"]:
        g0 = d[d.phase == phase]
        for b, g in g0.groupby("bin", observed=True):
            e, l = g[g.half == "early"], g[g.half == "late"]
            if len(e) >= 100 and tstat(e.pnl) > 2.5:
                print(f"  {phase:6s} {str(b):14s} early {100*e.pnl.mean():+.2f}c (t {tstat(e.pnl):+.1f}, n {len(e)})"
                      f"  ->  late {100*l.pnl.mean():+.2f}c (t {tstat(l.pnl):+.1f}, n {len(l)})")
    d.to_pickle("data/calib_snapshots.pkl")


if __name__ == "__main__":
    main()
