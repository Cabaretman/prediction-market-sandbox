"""Out-of-sample test of the heavy-favourite bias on LOW-TIER tennis
(ITF men, ITF women, WTA Challenger) -- data not used to form the ideas.

Hypotheses fixed beforehand (from ATP/WTA/ATP-Challenger data):
  H1: pre-match favourites (88-97c) in thin matches are profitable after costs
  H2: the favourite bias is larger in thinner markets
Also reports a widened band (80-97c) and the time split (early/late half).
"""
import numpy as np
import pandas as pd

import kalshi_analyze as ka
import strat_fav as sf

LOW = ["ITF", "ITFW", "WTACHALLENGER"]


def run(lo, hi):
    sf.LO, sf.HI = lo, hi
    d = sf.seg_cols(sf.events())
    d = d[d.series.isin(LOW)]
    d["vq"] = pd.qcut(d.volsofar.rank(method="first"), 3, labels=["thin", "mid", "thick"])
    med = d.t.median()
    print(f"===== favourite band {lo:.2f}-{hi:.2f}: {len(d)} entries, {d.ev.nunique()} matches =====")
    for ph in ["pre", "inplay"]:
        g0 = d[d.phase == ph]
        print(f"-- {ph}: win {g0.y.mean():.3f} vs price {g0.price.mean():.3f}")
        ka.fmt("ALL, taker", g0.taker, g0.ev)
        for col in ["series", "vq", "lvl"]:
            for v, g in g0.groupby(col, observed=True):
                print(f"   {col}={str(v):14s} win {g.y.mean():.3f} vs {g.price.mean():.3f}", end="")
                ka.fmt("", g.taker, g.ev)
        ka.fmt("first half (time)", g0[g0.t < med].taker, g0[g0.t < med].ev)
        ka.fmt("second half (time)", g0[g0.t >= med].taker, g0[g0.t >= med].ev)
    print()
    return d


if __name__ == "__main__":
    run(0.88, 0.97)
    run(0.80, 0.97)
