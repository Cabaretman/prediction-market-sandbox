"""Pre-game mispricing map for other Kalshi sports (2-outcome game markets)."""
import sys, numpy as np, pandas as pd
import calib_map as cm
def ts(x): return x.mean()/(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>2 and x.std()>0 else 0
for lead in (10, 60):
    d = cm.snapshots("data/kalshi/KX*GAME.jsonl", lead=lead); d = d[d.phase=="pre"]
    med = d.t.median()
    print(f"===== PRE-GAME, T-{lead} min: {d.ev.nunique()} games; {d.groupby('tier').ev.nunique().to_dict()} =====")
    for b, g in d.groupby("bin", observed=True):
        e, l = g[g.t<med], g[g.t>=med]
        per = " ".join(f"{k.replace('KX','').replace('GAME','')}:{100*gg.pnl.mean():+.1f}c({len(gg)})" for k, gg in g.groupby("tier") if len(gg) >= 15)
        print(f"{str(b):14s} n={len(g):5d} win {g.y.mean():.3f} vs {g.p.mean():.3f} | P&L {100*g.pnl.mean():+6.2f}c t={ts(g.pnl):+5.2f} | early {100*e.pnl.mean():+5.2f} late {100*l.pnl.mean():+5.2f} | {per}")
    print()
