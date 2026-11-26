"""Systematising "buy heavy favourites" on Kalshi (ATP, WTA, Challenger).

Entry event per match (mimics the discretionary strategy):
  IN-PLAY: first minute after the detected start where a player's mid is in
           [LO, HI] (default 88-97c) with spread <= 3c.
  PRE    : last clean quote before start, favourite priced in [LO, HI].
Context features (all known at entry time, no look-ahead):
  series, pre-match price of this player (chalk / comeback), minutes since
  start, 10-min price change into the zone (surge vs creep), spread, volume
  traded so far.
Execution:
  TAKER: buy at ask + 0.07*p(1-p)
  MAKER: bid at the best bid; filled only if the ask trades down to it within
         5 minutes (adverse selection baked in); fee 0.0175*p(1-p).
Segment search on the TRAIN half (earlier matches), confirmation on TEST.
"""
import itertools

import numpy as np
import pandas as pd

import kalshi_analyze as ka

LO, HI = 0.88, 0.97
TK, MK = 0.07, 0.0175


def events():
    rows = []
    for m in ka.load("data/kalshi/KX*MATCH.jsonl"):
        s = ka.match_start(m)
        if s is None or s < 2:
            continue
        bid, ask, y, vol = m["bid"], m["ask"], m["y"], m["vol"]
        n = len(bid)
        mid = (bid + ask) / 2
        pre0 = mid[s - 1]  # pre-match price of side 0
        # ---- pre-match entry
        if ask[s - 1] - bid[s - 1] <= 0.03:
            for side in (0, 1):
                fm = pre0 if side == 0 else 1 - pre0
                if LO <= fm <= HI:
                    rows.append(entry(m, side, s - 1, "pre", pre0, s, bid, ask, y, vol))
        # ---- first in-play entry into the zone (either player)
        for i in range(s + 3, n - 6):
            if ask[i] - bid[i] > 0.03:
                continue
            for side in (0, 1):
                fm = mid[i] if side == 0 else 1 - mid[i]
                if LO <= fm <= HI:
                    rows.append(entry(m, side, i, "inplay", pre0, s, bid, ask, y, vol))
                    break
            else:
                continue
            break
    return pd.DataFrame([r for r in rows if r])


def entry(m, side, i, phase, pre0, s, bid, ask, y, vol):
    n = len(bid)
    if side == 0:
        b, a, yy = bid[i], ask[i], y
        fut_ask = ask[i + 1:min(i + 6, n)]
        p_pre = pre0
        mid_10 = (bid[max(i - 10, 0)] + ask[max(i - 10, 0)]) / 2
    else:
        b, a, yy = 1 - ask[i], 1 - bid[i], 1 - y
        fut_ask = 1 - bid[i + 1:min(i + 6, n)]
        p_pre = 1 - pre0
        mid_10 = 1 - (bid[max(i - 10, 0)] + ask[max(i - 10, 0)]) / 2
    if np.isnan(a) or np.isnan(b):
        return None
    filled = len(fut_ask) > 0 and np.nanmin(fut_ask) <= b + 1e-9
    return dict(ev=m["event"], series=m["series"].replace("KX", "").replace("MATCH", ""),
                t=m["t0"] + 60 * i, phase=phase, price=(a + b) / 2, spread=a - b,
                p_pre=p_pre, mins=i - s, move10=(a + b) / 2 - mid_10,
                volsofar=vol[:i + 1].sum(), y=yy,
                taker=yy - a - TK * a * (1 - a),
                maker=(yy - b - MK * b * (1 - b)) if filled else np.nan, filled=filled)


def seg_cols(d):
    d = d.copy()
    d["who"] = pd.cut(d.p_pre, [-1, 0.5, 0.75, 2], labels=["comeback(pre<50)", "mild fav(50-75)", "big fav(>75)"])
    d["when"] = pd.cut(d.mins, [-10, 0, 45, 90, 1e9], labels=["pre", "<45m", "45-90m", ">90m"])
    d["how"] = pd.cut(d.move10, [-2, 0.03, 0.10, 2], labels=["creep(<3c/10m)", "move(3-10c)", "surge(>10c)"])
    d["lvl"] = pd.cut(d.price, [0, 0.92, 0.95, 1], labels=["88-92", "92-95", "95-97"])
    return d


def main():
    d = seg_cols(events())
    med = d.t.median()
    d["half"] = np.where(d.t < med, "train", "test")
    print(f"{len(d)} entry events ({d.ev.nunique()} matches); "
          f"pre {sum(d.phase=='pre')}, in-play {sum(d.phase=='inplay')}\n")
    print("== Overall (all entries) ==")
    for ph, g in d.groupby("phase"):
        print(f" [{ph}] win {g.y.mean():.3f} vs price {g.price.mean():.3f}")
        ka.fmt("   TAKER", g.taker, g.ev)
        f = g[g.filled]
        ka.fmt(f"   MAKER (fill rate {g.filled.mean():.0%})", f.maker, f.ev)
    print()
    print("== Single-factor breakdown (TRAIN | TEST), taker / maker ==")
    for col in ["series", "who", "when", "how", "lvl"]:
        print(f" -- {col}")
        for v, g in d.groupby(col, observed=True):
            tr, te = g[g.half == "train"], g[g.half == "test"]
            def c(x, k):
                x = x.dropna(subset=[k]) if k == "maker" else x
                return f"{100*x[k].mean():+5.2f}c(n={len(x)})" if len(x) else "  n/a"
            print(f"    {str(v):18s} taker {c(tr,'taker')} | {c(te,'taker')}    "
                  f"maker {c(tr,'maker')} | {c(te,'maker')}")
    print()
    # ---- combo search on train
    keys = ["series", "who", "when", "how"]
    res = []
    for combo in itertools.chain(itertools.combinations(keys, 2), itertools.combinations(keys, 3)):
        for vals, g in d[d.half == "train"].groupby(list(combo), observed=True):
            for ex in ["taker", "maker"]:
                gg = g.dropna(subset=[ex])
                if len(gg) < 150:
                    continue
                mu = gg[ex].mean()
                se = gg[ex].std() / np.sqrt(len(gg))
                res.append(dict(combo=combo, vals=vals, ex=ex, n=len(gg), mu=mu, t=mu / se))
    res = pd.DataFrame(res).sort_values("t", ascending=False)
    print(f"== Segment search on TRAIN: {len(res)} candidate segments (n>=150) ==")
    print(f"   expected false positives at t>2 by chance: ~{0.023*len(res):.0f}; "
          f"found {(res.t > 2).sum()}")
    top = res[res.t > 2].head(10)
    print("   Top segments on TRAIN -> TEST:")
    for r in top.itertuples():
        mask = np.ones(len(d), bool)
        for k, v in zip(r.combo, r.vals):
            mask &= (d[k] == v).to_numpy()
        te = d[mask & (d.half == "test").to_numpy()].dropna(subset=[r.ex])
        se = te[r.ex].std() / np.sqrt(max(len(te), 1))
        print(f"   {dict(zip(r.combo, r.vals))} [{r.ex}]  TRAIN {100*r.mu:+.2f}c t={r.t:.1f} n={r.n}"
              f"  ->  TEST {100*te[r.ex].mean():+.2f}c ± {196*se:.2f} n={len(te)}")


if __name__ == "__main__":
    main()
