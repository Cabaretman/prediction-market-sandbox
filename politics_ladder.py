"""Deep resting-order ('spike catcher') backtest on Kalshi politics trade tapes.

Idea: thin politics books get swept by impatient market orders that pay far above
the prevailing price, which then snaps back. A maker who keeps small resting orders
X cents away from the recent price on BOTH sides collects those sweeps.

Rules (no look-ahead):
  ref    = median YES price of the previous 10 trades that happened >= 60s before
           (our orders are assumed placed/updated off that, with a 60s lag)
  orders = sell side S at ref_S + X  (i.e. buy the other side at 1 - (ref_S + X)),
           for both S = yes and S = no
  fill   = only if a trade printed at least 1c THROUGH our price (level fully
           consumed => queue position irrelevant); filled at OUR price
  limit  = at most one fill per market per side per 6 hours
  hold   = to settlement; maker fee 0.0175 p(1-p)
Reported per X: fills, P&L per contract, by half, ex-midterm, by family,
and scaled to the whole universe per year.
Usage: python3 politics_ladder.py
"""
import json

import numpy as np
import pandas as pd

MK = 0.0175


def prep():
    d = pd.read_pickle("data/politics/tape.pkl").sort_values(["ticker", "t"]).reset_index(drop=True)
    out = []
    for tk, g in d.groupby("ticker", sort=False):
        t, yp = g.t.values, g.yp.values
        # index of last trade with time <= t - 60
        j = np.searchsorted(t, t - 60, side="right") - 1
        ref = np.full(len(g), np.nan)
        for i in range(len(g)):
            if j[i] >= 4:
                ref[i] = np.median(yp[max(0, j[i] - 9):j[i] + 1])
        g = g.assign(ref=ref)
        out.append(g)
    d = pd.concat(out)
    d["refq"] = np.where(d.side == "yes", d.ref, 1 - d.ref)
    return d[d.ref.notna()]


def run(d, X, gap=6 * 3600):
    g = d[d.q >= d.refq + X + 0.01].copy()
    g["L"] = (g.refq + X).round(2)
    g = g[g.L < 0.99]
    g = g.sort_values("t")
    keep, last = [], {}
    for r in g[["ticker", "side", "t"]].itertuples():
        k = (r.ticker, r.side)
        if r.t - last.get(k, -1e18) >= gap:
            keep.append(r.Index)
            last[k] = r.t
    f = g.loc[keep]
    # we sold side S at L  => P&L = L - win(S) - fee
    f["pnl"] = f.L - f.win - MK * f.L * (1 - f.L)
    return f


def rep(f, label, scale, yrs):
    if len(f) < 20:
        return
    pe = f.groupby("event").pnl.mean()
    t = pe.mean() / pe.std() * np.sqrt(len(pe)) if pe.std() > 0 else 0
    med = f.close.median()
    e, l = f[f.close < med].pnl.mean(), f[f.close >= med].pnl.mean()
    x = f[f.qtr != "2026Q4"].pnl.mean()
    rec = f[f.t >= pd.Timestamp("2025-01-01").timestamp()]
    per_yr = len(rec) * scale / yrs
    print(f"{label:22s} fills {len(f):6d} ev {len(pe):5d} | sold at {f.L.mean():.3f}, side won {f.win.mean():.3f} | "
          f"{100 * f.pnl.mean():+6.2f}c t{t:+5.1f} | early {100 * e:+6.2f} late {100 * l:+6.2f} ex-mid {100 * x:+6.2f} | "
          f"~{per_yr:,.0f} fills/yr universe")


def main():
    d = prep()
    n_univ = len({json.loads(l)["ticker"] for l in open("data/politics/markets.jsonl")
                  if json.loads(l)["volume"] >= 5000})
    scale = n_univ / d.ticker.nunique()
    yrs = (d.t.max() - pd.Timestamp("2025-01-01").timestamp()) / 86400 / 365
    print(f"{d.ticker.nunique()} sample markets, universe {n_univ}, scale x{scale:.1f}, {yrs:.2f} yrs since 2025\n")
    for X in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
        f = run(d, X)
        rep(f, f"X = {100 * X:.0f}c", scale, yrs)
    print()
    for X in [0.05, 0.10, 0.20]:
        f = run(d, X)
        print(f"--- X = {100 * X:.0f}c: by price we sold at / days to close / family")
        f["Lb"] = pd.cut(f.L, [0, .2, .5, .8, .9, .95, 1])
        for k, h in f.groupby("Lb", observed=True):
            rep(h, f"  sold@{k}", scale, yrs)
        f["db"] = pd.cut(f.dtc, [-1, 1, 7, 30, 1e9])
        for k, h in f.groupby("db", observed=True):
            rep(h, f"  dtc {k}", scale, yrs)
        for k, h in f.groupby("fam"):
            rep(h, f"  {k}", scale, yrs)
        print()


if __name__ == "__main__":
    main()


def capacity(d, f):
    """Contracts a sweep traded at prices at/through our level L within 5s (what we could have absorbed)."""
    idx = d.set_index(["ticker", "side"]).sort_index()
    cap = []
    for r in f.itertuples():
        g = idx.loc[(r.ticker, r.side)]
        w = g[(g.t >= r.t - 5) & (g.t <= r.t + 5) & (g.q >= r.L)]
        cap.append(w.n.sum())
    return np.array(cap)


def money():
    d = prep()
    n_univ = len({json.loads(l)["ticker"] for l in open("data/politics/markets.jsonl")
                  if json.loads(l)["volume"] >= 5000})
    scale = n_univ / d.ticker.nunique()
    t25 = pd.Timestamp("2025-01-01").timestamp()
    yrs = (d.t.max() - t25) / 86400 / 365
    print(f"\n===== Capacity: $/yr for the whole universe at order size S (fills since 2025, x{scale:.1f}) =====")
    for X in [0.15, 0.20, 0.30]:
        f = run(d, X)
        f = f[f.dtc > 1]
        f["cap"] = capacity(d, f)
        rec = f[f.t >= t25]
        print(f"X={100 * X:.0f}c, >1d to close: fills {len(f)}, median sweep size through our level "
              f"{np.median(f.cap):.0f} contracts")
        for S in [10, 50, 100, 250, 1000]:
            q = np.minimum(rec.cap, S)
            pnl = (q * rec.pnl).sum() * scale / yrs
            risk = (q * (1 - rec.L)).sum() * scale / yrs
            print(f"   S={S:5d}: filled {q.sum() * scale / yrs:10,.0f} ct/yr, P&L ${pnl:10,.0f}/yr, "
                  f"capital deployed ${risk:10,.0f}/yr, {100 * (q * rec.pnl).sum() / max((q * rec.q).sum(), 1):+.1f}c/ct")


def checks():
    d = prep()
    print("\n===== Fill rule check: count fills AT our level too (q >= L, not 1c through) =====")
    for X in [0.20, 0.30]:
        g = d[(d.q >= d.refq + X) & (d.dtc > 1)].copy()
        g["L"] = (g.refq + X).round(2)
        g = g[g.L < 0.99].sort_values("t")
        keep, last = [], {}
        for r in g[["ticker", "side", "t"]].itertuples():
            k = (r.ticker, r.side)
            if r.t - last.get(k, -1e18) >= 6 * 3600:
                keep.append(r.Index)
                last[k] = r.t
        f = g.loc[keep]
        f["pnl"] = f.L - f.win - MK * f.L * (1 - f.L)
        at = f[(f.q - f.L).abs() < 0.005]
        rep(f, f"X={100*X:.0f}c inclusive", 1, 1)
        rep(at, f"X={100*X:.0f}c exactly-at-L", 1, 1)
    print("\n===== Collateral: resting orders on Kalshi lock their cost =====")
    m = d.groupby("ticker").agg(op=("t", "min"), cl=("close", "first"))
    t25 = pd.Timestamp("2025-01-01").timestamp()
    m = m[m.cl >= t25]
    span = d.t.max() - t25
    conc = ((m.cl - m.op.clip(lower=t25)).clip(lower=0)).sum() / span
    print(f"sample: on average {conc:.0f} markets open at once (x3.9 for the universe = {conc * 3.9:.0f})")


if __name__ == "__main__":
    pass
