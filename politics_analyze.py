"""Mispricing map for Kalshi politics/elections markets (daily candles).

Snapshots (no look-ahead):
  ALL markets   : price at open + 1d / +7d / +30d (only markets still open then)
  ELECTIONS only: price at close - 30d / -7d / -1d (election day is fixed)
For each snapshot, both sides: buy YES at ask, buy NO at 1-bid (spread <= 10c),
held to settlement, fee 0.07*p(1-p). Report by price bin, early/late split by
close date, P&L per contract and per $-day of capital locked.
Partisan test: outcomes labelled Democrat/Republican in their titles.
"""
import json
import re

import numpy as np
import pandas as pd

TK = 0.07
DAY = 86400
BINS = [0, .03, .05, .10, .20, .35, .50, .65, .80, .90, .95, .97, 1.0]
ELECTION_TAGS = {"US Elections", "International elections", "Primaries", "Senate", "House",
                 "Governor", "Other US Elections", "State legislatures", "Foreign Elections"}


def ts(s):
    return pd.Timestamp(s).timestamp()


def party(m):
    txt = f"{m.get('yes_sub') or ''} {m.get('title') or ''}".lower()
    d = bool(re.search(r"\bdemocrat|\bdemocratic\b|\bdem\b", txt))
    r = bool(re.search(r"\brepublican|\bgop\b|\brep\b", txt))
    return "D" if d and not r else "R" if r and not d else None


def load():
    meta = {}
    for l in open("data/politics/markets.jsonl"):
        m = json.loads(l)
        meta[m["ticker"]] = m
    rows = []
    seen = set()
    for l in open("data/politics/candles.jsonl"):
        c = json.loads(l)
        if c["ticker"] in seen:   # two collectors may have written the same market
            continue
        seen.add(c["ticker"])
        m = meta.get(c["ticker"])
        if m is None or not c["candles"]:
            continue
        a = np.array([[x[0], np.nan if x[1] is None else x[1], np.nan if x[2] is None else x[2]]
                      for x in c["candles"]], dtype=float)
        op, cl = ts(m["open"]), ts(m["close"])
        y = 1.0 if m["result"] == "yes" else 0.0
        is_elec = bool(ELECTION_TAGS & set(m["tags"]))
        snaps = [(f"open+{k}d", op + k * DAY) for k in (1, 7, 30)]
        if is_elec:
            snaps += [(f"close-{k}d", cl - k * DAY) for k in (30, 7, 1)]
        for name, t in snaps:
            if t >= cl or t <= op:
                continue
            i = np.searchsorted(a[:, 0], t, side="right") - 1
            if i < 0:
                continue
            b, k = a[i, 1], a[i, 2]
            if np.isnan(b) or np.isnan(k) or k - b > 0.10 or k <= 0 or b >= 1:
                continue
            mid = (b + k) / 2
            hold = max((cl - t) / DAY, 1)
            for side, p, px, yy in [("YES", mid, k, y), ("NO", 1 - mid, 1 - b, 1 - y)]:
                pnl = yy - px - TK * px * (1 - px)
                rows.append(dict(ticker=m["ticker"], event=m["event"], series=m["series"],
                                 elec=is_elec, snap=name, side=side, p=p, px=px, y=yy, pnl=pnl,
                                 hold=hold, close=cl, party=party(m), vol=m["volume"]))
    d = pd.DataFrame(rows)
    d["bin"] = pd.cut(d.p, BINS)
    return d


def tstat(x):
    # cluster by event: average within event first
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0


def table(g0, label):
    med = g0.close.median()
    print(f"--- {label}: {g0.ticker.nunique()} markets, {g0.event.nunique()} events")
    print(f"{'price':13s} {'n':>6s} {'win':>6s} {'price':>6s} | {'P&L':>8s} {'t(ev)':>6s} | "
          f"{'early':>7s} {'late':>7s} | {'hold d':>6s} {'%/yr':>7s}")
    for b, g in g0.groupby("bin", observed=True):
        ev = g.groupby("event").pnl.mean()
        e, l = g[g.close < med], g[g.close >= med]
        ann = (g.pnl.sum() / g.px.sum()) / g.hold.mean() * 365 * 100
        print(f"{str(b):13s} {len(g):6d} {g.y.mean():6.3f} {g.p.mean():6.3f} | {100*g.pnl.mean():+7.2f}c "
              f"{tstat(ev):+6.2f} | {100*e.pnl.mean():+6.2f}c {100*l.pnl.mean():+6.2f}c | "
              f"{g.hold.median():6.0f} {ann:+6.0f}%")


def main():
    d = load()
    print(f"{d.ticker.nunique()} markets with usable daily prices\n")
    for snap in ["open+1d", "open+7d", "open+30d"]:
        table(d[d.snap == snap], f"ALL politics, snapshot {snap}")
        print()
    for snap in ["close-30d", "close-7d", "close-1d"]:
        table(d[d.snap == snap], f"ELECTIONS, snapshot {snap}")
        print()
    print("===== Partisan test: party-labelled outcomes, buying YES on that party =====")
    p = d[(d.party.notna()) & (d.side == "YES")]
    for snap in ["open+7d", "close-30d", "close-7d", "close-1d"]:
        for pt, g in p[p.snap == snap].groupby("party"):
            ev = g.groupby("event").pnl.mean()
            print(f"  {snap:10s} {pt}: n={len(g):5d} won {g.y.mean():.3f} vs price {g.p.mean():.3f}  "
                  f"P&L {100*g.pnl.mean():+.2f}c  t={tstat(ev):+.2f}")
    d.to_pickle("data/politics/snapshots.pkl")


if __name__ == "__main__":
    main()
