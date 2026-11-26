"""Dirty tests on Kalshi tennis match markets (1-min bid/ask candles).

Tests
  A. Calibration / favourite-longshot: is the price an unbiased probability?
  B. In-play overreaction: after a sharp move in the mid price, does the
     outcome (settlement) or the later price revert? Strategy P&L for
     fading vs following the move, paying the ask/bid + Kalshi taker fee.

All P&L is per $1-payout contract, in cents. Standard errors are clustered
by match (bootstrap over matches), because shocks within a match are
correlated.
"""
import glob
import json
import sys

import numpy as np
import pandas as pd

MAX_SPREAD = 0.06
rng = np.random.default_rng(0)


def fee(p):
    """Kalshi taker fee per contract (dollars): 0.07*p*(1-p), rounded up to
    the cent per order; we ignore rounding (assumes larger orders)."""
    return 0.07 * p * (1 - p)


def load(pattern="data/kalshi/*.jsonl"):
    mats = []
    for fn in glob.glob(pattern):
        with open(fn) as f:
            for line in f:
                r = json.loads(line)
                c = r["candles"]
                if len(c) < 30:
                    continue
                a = np.array([[x[0], np.nan if x[1] is None else x[1],
                               np.nan if x[2] is None else x[2], x[4]] for x in c],
                             dtype=float)
                t0, t1 = int(a[0, 0]), int(a[-1, 0])
                idx = (a[:, 0] - t0) // 60
                n = int(idx[-1]) + 1
                bid = np.full(n, np.nan); ask = np.full(n, np.nan); vol = np.zeros(n)
                bid[idx.astype(int)] = a[:, 1]; ask[idx.astype(int)] = a[:, 2]
                vol[idx.astype(int)] = a[:, 3]
                bid = pd.Series(bid).ffill().to_numpy()
                ask = pd.Series(ask).ffill().to_numpy()
                y = 1.0 if r["results"][0] == "yes" else 0.0
                mats.append(dict(event=r["event"], series=r["series"], y=y,
                                 bid=bid, ask=ask, vol=vol,
                                 totvol=sum(r["volume"]), t0=t0))
    return mats


def match_start(m):
    """Heuristic in-play start: first minute of the first 10-minute stretch
    whose volume exceeds 3x the match's pre-period median 10-min volume and
    after which the mid moves. Dirty but serviceable."""
    v = pd.Series(m["vol"]).rolling(10, min_periods=1).sum().to_numpy()
    mid = (m["bid"] + m["ask"]) / 2
    n = len(v)
    # in-play markets trade much more; use the cumulative volume midpoint of
    # the session and walk back to the first sustained high-volume minute
    thr = np.nanquantile(v, 0.6)
    hi = v > max(thr, 1)
    for i in range(n - 30):
        if hi[i:i + 30].mean() > 0.8 and np.nanstd(mid[i:i + 60]) > 0.01:
            return max(i - 10, 0)
    return None


def boot(values, groups, B=500):
    df = pd.DataFrame({"v": values, "g": groups})
    gs = df.groupby("g")["v"].agg(["sum", "count"])
    s, c = gs["sum"].to_numpy(), gs["count"].to_numpy()
    k = len(s)
    est = s.sum() / c.sum()
    bs = []
    for _ in range(B):
        i = rng.integers(0, k, k)
        bs.append(s[i].sum() / c[i].sum())
    return est, np.std(bs)


def fmt(name, v, g):
    if len(v) == 0:
        print(f"  {name:48s} n=0"); return
    e, se = boot(np.asarray(v), np.asarray(g))
    print(f"  {name:48s} n={len(v):6d} matches={len(set(g)):5d}  "
          f"{100*e:+6.2f}c ± {196*se:4.2f}c  t={e/se if se else 0:+5.2f}")


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "data/kalshi/*.jsonl"
    ms = load(pat)
    print(f"{len(ms)} matches loaded\n")

    # ---------- A. calibration ----------
    pre_rows, inplay_rows = [], []
    for m in ms:
        s = match_start(m)
        m["start"] = s
        mid = (m["bid"] + m["ask"]) / 2
        spr = m["ask"] - m["bid"]
        if s is not None and s >= 5:
            j = s - 1
            if spr[j] <= MAX_SPREAD:
                for side, p, y in [(0, mid[j], m["y"]), (1, 1 - mid[j], 1 - m["y"])]:
                    ask = m["ask"][j] if side == 0 else 1 - m["bid"][j]
                    pre_rows.append(dict(ev=m["event"], series=m["series"], p=p,
                                         ask=ask, y=y))
        if s is not None:
            # sample in-play every 15 min
            for j in range(s + 10, len(mid) - 5, 15):
                if spr[j] <= MAX_SPREAD and 0.03 < mid[j] < 0.97:
                    inplay_rows.append(dict(ev=m["event"], p=mid[j], y=m["y"]))
    pre = pd.DataFrame(pre_rows)
    found = sum(m["start"] is not None for m in ms)
    print(f"match start detected for {found}/{len(ms)} matches\n")

    print("== A1. Pre-match calibration (mid just before detected start) ==")
    print("   bucket        n   avg price  win rate   P&L buying at ask incl. fee")
    pre["b"] = pd.cut(pre.p, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    for bk, g in pre.groupby("b", observed=True):
        pnl = g.y - g.ask - fee(g.ask)
        e, se = boot(pnl.to_numpy(), g.ev.to_numpy(), 200)
        print(f"   {str(bk):12s} {len(g):5d}   {g.p.mean():.3f}     {g.y.mean():.3f}"
              f"     {100*e:+5.2f}c ± {196*se:.2f}c")
    for ser, g in pre.groupby("series"):
        fav = g[g.p > 0.5]
        pnl = fav.y - fav.ask - fee(fav.ask)
        fmt(f"back pre-match favourite [{ser}]", pnl, fav.ev)

    ip = pd.DataFrame(inplay_rows)
    print("\n== A2. In-play calibration (sampled every 15 min) ==")
    ip["b"] = pd.cut(ip.p, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    for bk, g in ip.groupby("b", observed=True):
        print(f"   {str(bk):12s} n={len(g):6d}  price {g.p.mean():.3f}  "
              f"win {g.y.mean():.3f}  gap {100*(g.y.mean()-g.p.mean()):+.1f}c")

    # ---------- B. shocks ----------
    print("\n== B. In-play shocks: mid moves >= D within W minutes ==")
    print("   'fade' = bet on the side the move went against; 'follow' = opposite.")
    print("   Entry at the ask one minute after the shock (+fee). Exit at settlement")
    print("   or at the bid after H minutes (+fee on exit).\n")
    for D, W in [(0.10, 5), (0.15, 5), (0.20, 10), (0.25, 10)]:
        rows = []
        for m in ms:
            s = m["start"]
            if s is None:
                continue
            bid, ask, y = m["bid"], m["ask"], m["y"]
            mid = (bid + ask) / 2
            n = len(mid)
            j = s + W
            while j < n - 2:
                d = mid[j] - mid[j - W]
                if abs(d) >= D and 0.15 <= mid[j] <= 0.85:
                    k = j + 1  # 1-minute latency
                    if ask[k] - bid[k] <= MAX_SPREAD:
                        # fade: if price fell (d<0) buy YES; if rose, buy NO
                        if d < 0:
                            f_ask, f_y = ask[k], y
                            f_exit = {H: bid[min(k + H, n - 1)] for H in (5, 15, 30)}
                            g_ask, g_y = 1 - bid[k], 1 - y
                            g_exit = {H: 1 - ask[min(k + H, n - 1)] for H in (5, 15, 30)}
                        else:
                            f_ask, f_y = 1 - bid[k], 1 - y
                            f_exit = {H: 1 - ask[min(k + H, n - 1)] for H in (5, 15, 30)}
                            g_ask, g_y = ask[k], y
                            g_exit = {H: bid[min(k + H, n - 1)] for H in (5, 15, 30)}
                        row = dict(ev=m["event"], series=m["series"], d=d,
                                   pre=mid[j - W], post=mid[j],
                                   fade_settle=f_y - f_ask - fee(f_ask),
                                   follow_settle=g_y - g_ask - fee(g_ask),
                                   # outcome vs mid: >0 means price was too low
                                   # on the fade side (overreaction)
                                   fade_gap=f_y - (mid[k] if d < 0 else 1 - mid[k]))
                        for H in (5, 15, 30):
                            row[f"fade_{H}"] = f_exit[H] - f_ask - fee(f_ask) - fee(f_exit[H])
                            row[f"follow_{H}"] = g_exit[H] - g_ask - fee(g_ask) - fee(g_exit[H])
                        rows.append(row)
                    j += 30  # cooldown to limit overlapping samples
                else:
                    j += 1
        r = pd.DataFrame(rows)
        print(f" D={D:.2f} W={W}min: {len(r)} shocks in {r.ev.nunique() if len(r) else 0} matches")
        if len(r) == 0:
            continue
        fmt("outcome minus mid on fade side (no costs)", r.fade_gap, r.ev)
        fmt("FADE, hold to settlement", r.fade_settle, r.ev)
        fmt("FOLLOW, hold to settlement", r.follow_settle, r.ev)
        for H in (5, 15, 30):
            fmt(f"FADE, exit after {H}m", r[f"fade_{H}"], r.ev)
        for H in (5, 15, 30):
            fmt(f"FOLLOW, exit after {H}m", r[f"follow_{H}"], r.ev)
        print()


if __name__ == "__main__":
    main()
