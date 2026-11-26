"""Dirty v0 backtest: ATP pre-match strategies vs Pinnacle closing odds.

Data: tennis-data.co.uk ATP files (mirrored in data/atp/). Pinnacle closing
odds (PSW/PSL) are treated as the "sharp" market price we would trade against.

Strategies tested (all bet 1 unit flat, settled at the Pinnacle closing price):
  1. Elo value     - surface-blended Elo vs no-vig Pinnacle probability.
  2. Price buckets - favourite/longshot bias: ROI by implied probability.
  3. Form reversion - back players whose recent results were much worse than
     their Elo expected (the "market overreacts to bad form" hunch), and the
     opposite (momentum).

Elo is updated online in date order, so every prediction uses only past
matches. Elo warms up on 2000-2011; evaluation is 2012 onward.
"""
import glob
from collections import defaultdict, deque

import numpy as np
import pandas as pd

ROUND_ORDER = {"1st Round": 1, "2nd Round": 2, "3rd Round": 3, "4th Round": 4,
               "Quarterfinals": 5, "Semifinals": 6, "The Final": 7,
               "Round Robin": 4}
EVAL_START = "2012-01-01"


def load():
    frames = []
    for f in sorted(glob.glob("data/atp/[0-9]*.csv")):
        d = pd.read_csv(f, low_memory=False)
        d["_row"] = np.arange(len(d))
        d["_file"] = f
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Comment"].astype(str).str.strip() != "Walkover"]
    df = df.dropna(subset=["Date", "Winner", "Loser"])
    df["_r"] = df["Round"].map(ROUND_ORDER).fillna(0)
    df = df.sort_values(["Date", "_file", "_row"]).reset_index(drop=True)
    for c in ["PSW", "PSL", "MaxW", "MaxL"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def k_factor(n):
    return 250 / (n + 5) ** 0.4  # FiveThirtyEight-style decaying K


def run(df):
    elo = defaultdict(lambda: 1500.0)
    selo = defaultdict(lambda: 1500.0)
    n = defaultdict(int)
    sn = defaultdict(int)
    resid = defaultdict(lambda: deque(maxlen=5))  # recent (result - expected)
    rows = []
    for r in df.itertuples(index=False):
        w, l, s = r.Winner, r.Loser, r.Surface
        ew = 1 / (1 + 10 ** ((elo[l] - elo[w]) / 400))
        sew = 1 / (1 + 10 ** ((selo[(l, s)] - selo[(w, s)]) / 400))
        pw = 0.5 * ew + 0.5 * sew
        rows.append(dict(
            Date=r.Date, W=w, L=l, p_w=pw, nW=n[w], nL=n[l],
            formW=sum(resid[w]), formL=sum(resid[l]),
            lenW=len(resid[w]), lenL=len(resid[l]),
            PSW=r.PSW, PSL=r.PSL, MaxW=r.MaxW, MaxL=r.MaxL))
        # updates
        elo[w] += k_factor(n[w]) * (1 - ew)
        elo[l] -= k_factor(n[l]) * (1 - ew)
        selo[(w, s)] += k_factor(sn[(w, s)]) * (1 - sew)
        selo[(l, s)] -= k_factor(sn[(l, s)]) * (1 - sew)
        n[w] += 1; n[l] += 1; sn[(w, s)] += 1; sn[(l, s)] += 1
        resid[w].append(1 - pw)
        resid[l].append(0 - (1 - pw))
    return pd.DataFrame(rows)


def to_bets(m):
    """Expand each match into two rows (one per player) so the winner/loser
    column layout cannot leak into the strategy."""
    a = pd.DataFrame(dict(Date=m.Date, p=m.p_w, odds=m.PSW, opp_odds=m.PSL,
                          best=m.MaxW, n=m.nW, n_opp=m.nL, form=m.formW,
                          flen=m.lenW, won=1))
    b = pd.DataFrame(dict(Date=m.Date, p=1 - m.p_w, odds=m.PSL, opp_odds=m.PSW,
                          best=m.MaxL, n=m.nL, n_opp=m.nW, form=m.formL,
                          flen=m.lenL, won=0))
    x = pd.concat([a, b], ignore_index=True)
    x = x[(x.Date >= EVAL_START) & x.odds.gt(1) & x.opp_odds.gt(1)].copy()
    inv = 1 / x.odds + 1 / x.opp_odds
    x["q"] = (1 / x.odds) / inv  # no-vig market prob (multiplicative)
    x["margin"] = inv - 1
    x["pnl"] = np.where(x.won == 1, x.odds - 1, -1.0)
    x["pnl_best"] = np.where(x.won == 1, x.best.fillna(x.odds) - 1, -1.0)
    return x


def summ(x, col="pnl"):
    if len(x) == 0:
        return "n=0"
    roi = x[col].mean()
    se = x[col].std(ddof=1) / np.sqrt(len(x))
    return f"n={len(x):6d}  ROI={roi:+.2%}  ±{1.96*se:.2%} (95%)  t={roi/se:+.2f}"


def main():
    df = load()
    m = run(df)
    x = to_bets(m)
    exp = x[(x.n >= 20) & (x.n_opp >= 20)]
    print(f"Eval period {x.Date.min().date()} → {x.Date.max().date()}, "
          f"{len(x)//2} matches with Pinnacle odds; median margin "
          f"{x.margin.median():.2%}\n")

    # Calibration sanity: Elo vs market log-loss
    def ll(p, y):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    print("Log-loss (lower=better), experienced players:")
    print(f"  Elo model     {ll(exp.p, exp.won):.4f}")
    print(f"  Pinnacle novig{ll(exp.q, exp.won):.4f}")
    print(f"  Blend 30/70   {ll(0.3*exp.p+0.7*exp.q, exp.won):.4f}\n")

    print("== 1. Elo value bets (bet when p*odds-1 > edge) ==")
    for thr in [0.0, 0.05, 0.10, 0.20]:
        b = exp[exp.p * exp.odds - 1 > thr]
        print(f"  edge>{thr:4.0%}  Pinnacle: {summ(b)}")
        print(f"              best-of-market: {summ(b, 'pnl_best')}")

    print("\n== 2. Favourite/longshot bias (Pinnacle price, all bets in bucket) ==")
    bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    x["bucket"] = pd.cut(x.q, bins)
    for bk, g in x.groupby("bucket", observed=True):
        print(f"  q in {str(bk):12s} {summ(g)}")

    print("\n== 3. Form reversion / momentum (sum of last-5 Elo residuals) ==")
    f = exp[exp.flen == 5]
    lo, hi = f.form.quantile([0.1, 0.9])
    print(f"  Back cold players (bottom 10% form): {summ(f[f.form <= lo])}")
    print(f"  Back hot players  (top 10% form)   : {summ(f[f.form >= hi])}")
    print(f"  Baseline, all experienced bets     : {summ(f)}")
    # Does the market already price form? compare to Elo-only model
    for name, g in [("cold", f[f.form <= lo]), ("hot", f[f.form >= hi])]:
        print(f"    {name}: actual win {g.won.mean():.3f}  market {g.q.mean():.3f}"
              f"  elo {g.p.mean():.3f}")

    print("\n== By year: Elo edge>5% at Pinnacle ==")
    b = exp[exp.p * exp.odds - 1 > 0.05]
    for yr, g in b.groupby(b.Date.dt.year):
        print(f"  {yr}: {summ(g)}")


if __name__ == "__main__":
    main()
