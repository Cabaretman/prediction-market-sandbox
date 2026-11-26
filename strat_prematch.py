"""Pre-match strategies on Kalshi (ATP, WTA, Challenger), no look-ahead.

1. Elo built ONLY from Kalshi match results, in time order (each match's
   rating uses only earlier results). Question: does Elo add information
   to Kalshi's pre-match price? Fit a blend on the first half, test on
   the second half. Bet when blended prob beats the ask by > fee + margin.
2. Line movement: mid at ~T-6h, T-2h, and just before start.
   - Does early movement continue (momentum) or revert?
   - Is the closing price better than the early one (then early prices are
     beatable if you can forecast the close)?
"""
import json
import glob

import numpy as np
import pandas as pd

import kalshi_analyze as ka

TAKER = 0.07


def logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def ll(p, y):
    p = np.clip(p, 0.01, 0.99)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def build():
    meta = {}
    for fn in glob.glob("data/kalshi/KX*MATCH.jsonl"):
        for line in open(fn):
            r = json.loads(line)
            if None in r["names"]:
                continue
            meta[r["event"]] = (r["names"], r["close_time"])
    rows = []
    for m in ka.load("data/kalshi/KX*MATCH.jsonl"):
        if m["event"] not in meta:
            continue
        s = ka.match_start(m)
        if s is None or s < 30:
            continue
        bid, ask = m["bid"], m["ask"]
        mid = (bid + ask) / 2
        spr = ask - bid

        def at(i):
            i = max(0, min(i, len(mid) - 1))
            return mid[i] if spr[i] <= 0.06 else np.nan
        names, ct = meta[m["event"]]
        rows.append(dict(ev=m["event"], series=m["series"], close=ct,
                         a=names[0], b=names[1], y=m["y"],
                         q=at(s - 1), bid=bid[s - 1], ask=ask[s - 1],
                         q2h=at(s - 120), q6h=at(s - 360),
                         ask6h=ask[max(s - 360, 0)], bid6h=bid[max(s - 360, 0)],
                         prevol=m["vol"][:s].sum()))
    d = pd.DataFrame(rows).sort_values("close").reset_index(drop=True)
    # Elo from Kalshi results only, online
    elo, n = {}, {}
    pe, na, nb = [], [], []
    for r in d.itertuples():
        ra, rb = elo.get(r.a, 1500.0), elo.get(r.b, 1500.0)
        p = 1 / (1 + 10 ** ((rb - ra) / 400))
        pe.append(p); na.append(n.get(r.a, 0)); nb.append(n.get(r.b, 0))
        ka_, kb_ = 250 / (n.get(r.a, 0) + 5) ** 0.4, 250 / (n.get(r.b, 0) + 5) ** 0.4
        elo[r.a] = ra + ka_ * (r.y - p)
        elo[r.b] = rb - kb_ * (r.y - p)
        n[r.a] = n.get(r.a, 0) + 1; n[r.b] = n.get(r.b, 0) + 1
    d["pelo"], d["na"], d["nb"] = pe, na, nb
    return d


def main():
    d = build()
    d = d.dropna(subset=["q"])
    d = d[(d.ask - d.bid) <= 0.06]
    print(f"{len(d)} matches with a clean pre-match quote\n")
    half = len(d) // 2
    tr, te = d.iloc[:half], d.iloc[half:]
    exp_te = te[(te.na >= 10) & (te.nb >= 10)]
    print("== 1. Elo (built from Kalshi results) vs Kalshi pre-match price ==")
    print(f"  test half, both players >=10 prior matches: {len(exp_te)} matches")
    print(f"  log-loss  Kalshi {ll(exp_te.q, exp_te.y):.4f}   Elo {ll(exp_te.pelo, exp_te.y):.4f}")
    # fit blend on train (experienced only)
    exp_tr = tr[(tr.na >= 10) & (tr.nb >= 10)]
    from numpy.linalg import lstsq
    best = None
    for w in np.linspace(0, 0.6, 13):
        z = (1 - w) * logit(exp_tr.q) + w * logit(exp_tr.pelo)
        l = ll(1 / (1 + np.exp(-z)), exp_tr.y)
        if best is None or l < best[1]:
            best = (w, l)
    w = best[0]
    z = (1 - w) * logit(exp_te.q) + w * logit(exp_te.pelo)
    pb = 1 / (1 + np.exp(-z))
    print(f"  best Elo weight on train half: {w:.2f}; test log-loss blend {ll(pb, exp_te.y):.4f}")
    for ser, g in exp_te.groupby("series"):
        print(f"    [{ser}] n={len(g)}  Kalshi {ll(g.q, g.y):.4f}  Elo {ll(g.pelo, g.y):.4f}")
    # betting: blended prob vs ask on both sides
    t = exp_te.assign(pb=pb)
    for margin in [0.0, 0.02, 0.04]:
        bets = []
        for r in t.itertuples():
            for side in (0, 1):
                p = r.pb if side == 0 else 1 - r.pb
                px = r.ask if side == 0 else 1 - r.bid
                yy = r.y if side == 0 else 1 - r.y
                if p - px - TAKER * px * (1 - px) > margin:
                    bets.append((r.ev, yy - px - TAKER * px * (1 - px)))
        b = pd.DataFrame(bets, columns=["ev", "pnl"])
        ka.fmt(f"bet blend>ask+fee+{100*margin:.0f}c (test half)", b.pnl, b.ev)

    print("\n== 2. Pre-match line movement ==")
    x = d.dropna(subset=["q6h", "q2h", "q"])
    x = x[(x.prevol > 0)]
    m1 = x.q2h - x.q6h
    m2 = x.q - x.q2h
    print(f"  {len(x)} matches with quotes at T-6h, T-2h, start")
    print(f"  corr(move 6h->2h, move 2h->start) = {np.corrcoef(m1, m2)[0,1]:+.3f}")
    print(f"  log-loss at T-6h {ll(x.q6h, x.y):.4f}  T-2h {ll(x.q2h, x.y):.4f}  start {ll(x.q, x.y):.4f}")
    # bet at T-6h ask in direction Elo disagrees with the market (experienced)
    xe = x[(x.na >= 10) & (x.nb >= 10)].iloc[len(x) // 2:]
    for thr in [0.03, 0.06, 0.10]:
        bets = []
        for r in xe.itertuples():
            diff = r.pelo - r.q6h
            if abs(diff) < thr:
                continue
            side = 0 if diff > 0 else 1
            px = r.ask6h if side == 0 else 1 - r.bid6h
            if not (0 < px < 1):
                continue
            yy = r.y if side == 0 else 1 - r.y
            cl = r.q if side == 0 else 1 - r.q  # closing value (CLV)
            bets.append((r.ev, yy - px - TAKER * px * (1 - px), cl - px))
        b = pd.DataFrame(bets, columns=["ev", "pnl", "clv"])
        if len(b):
            ka.fmt(f"Elo vs T-6h price, |diff|>={100*thr:.0f}c: P&L", b.pnl, b.ev)
            ka.fmt(f"   ... closing line value (close mid - entry)", b.clv, b.ev)


if __name__ == "__main__":
    main()
