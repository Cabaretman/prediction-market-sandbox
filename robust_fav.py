"""Robustness checks for: buy the PRE-MATCH favourite priced 95-97c."""
import json, math
import numpy as np, pandas as pd
import kalshi_analyze as ka

TK = 0.07

def entries(leads=(1, 10, 30, 60), lo=0.90, hi=0.995):
    rows = []
    for m in ka.load("data/kalshi/KX*MATCH.jsonl"):
        s = ka.match_start(m)
        if s is None: continue
        bid, ask = m["bid"], m["ask"]
        for L in leads:
            i = s - L
            if i < 0 or np.isnan(bid[i]) or np.isnan(ask[i]) or ask[i] - bid[i] > 0.03: continue
            mid = (bid[i] + ask[i]) / 2
            for side in (0, 1):
                fm = mid if side == 0 else 1 - mid
                if not (lo <= fm <= hi): continue
                a = ask[i] if side == 0 else 1 - bid[i]
                y = m["y"] if side == 0 else 1 - m["y"]
                rows.append(dict(ev=m["event"], series=m["series"], lead=L, t=m["t0"], mid=fm, ask=a,
                                 spread=ask[i]-bid[i], y=y, pnl=y - a - TK*a*(1-a),
                                 prevol=m["vol"][max(0,s-60):s].sum()))
    return pd.DataFrame(rows)

def binom_cdf(k, n, p):
    return sum(math.comb(n, j) * p**j * (1-p)**(n-j) for j in range(k+1))

def main():
    d = entries()
    print("== 1. Entry timing (band 95-97c, mid) ==")
    for L, g in d.groupby("lead"):
        g = g[(g.mid >= .95) & (g.mid <= .97)]
        ka.fmt(f"T-{L:2d} min  won {g.y.mean():.3%} vs {g.mid.mean():.3%}", g.pnl, g.ev)
    print("\n== 2. Neighbouring bands (T-10 min) ==")
    g10 = d[d.lead == 10]
    for lo, hi in [(.93,.95),(.94,.97),(.95,.96),(.96,.97),(.95,.97),(.95,.98),(.97,.99)]:
        g = g10[(g10.mid >= lo) & (g10.mid <= hi)]
        ka.fmt(f"{int(lo*100)}-{int(hi*100)}c won {g.y.mean():.3%} vs {g.mid.mean():.3%}", g.pnl, g.ev)
    print("\n== 3. Luck: losses vs fair-price expectation (T-10, 95-97c) ==")
    g = g10[(g10.mid >= .95) & (g10.mid <= .97)]
    n, k = len(g), int((g.y == 0).sum()); p_loss = 1 - g.mid.mean()
    print(f"  n={n}, losses={k}, expected at fair price {n*p_loss:.1f}; "
          f"P(<= {k} losses | fair) = {binom_cdf(k, n, p_loss):.4f}")
    print(f"  break-even loss count after costs: ~{(n*(1-g.ask.mean()-TK*g.ask.mean()*(1-g.ask.mean()))):.1f}")
    print("\n  losses (check by hand):")
    print(g[g.y == 0][["ev", "mid", "ask"]].to_string())
    print("\n== 4. By tier and time half (T-10, 95-97c) ==")
    med = g.t.median()
    ka.fmt("earlier half", g[g.t < med].pnl, g[g.t < med].ev)
    ka.fmt("later half", g[g.t >= med].pnl, g[g.t >= med].ev)
    for s_, gg in g.groupby("series"):
        print(f"   {s_:24s} n={len(gg):4d} losses={(gg.y==0).sum():2d} won {gg.y.mean():.3f} vs {gg.mid.mean():.3f} P&L {100*gg.pnl.mean():+.2f}c")
    g.to_csv("data/fav9597_entries.csv", index=False)

if __name__ == "__main__":
    main()
