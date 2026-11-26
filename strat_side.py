"""Kalshi side markets priced off the liquid match-winner anchor.

Layers
 1. Model-free arbitrage (any time):
      sum over exact scores = 1 ;  A20+A21 (+A30+A31+A32) = P(A wins match)
    -> locked arb if buying a complete set at the asks costs < $1 after fees.
 2. Exact score vs independent-sets model from the anchor (pre-match, T-10m).
 3. Set-1 winner vs anchor (pre-match), mapping fitted on TRAIN half only.
Bets are taker at the side market's ask (+0.07*p(1-p)), held to settlement.
Split train/test by time (first half of matches by date = train).
"""
import glob
import json
import unicodedata

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import kalshi_analyze as ka

TAKER = 0.07
LEAD = 10  # minutes before detected start for the pre-match snapshot


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return s


def fee(p):
    return TAKER * p * (1 - p)


def series_at(candles, t_grid):
    """bid/ask of a side market at given timestamps (ffill, None if no data)."""
    if not candles:
        return np.full(len(t_grid), np.nan), np.full(len(t_grid), np.nan)
    a = np.array([[c[0], np.nan if c[1] is None else c[1],
                   np.nan if c[2] is None else c[2]] for c in candles], dtype=float)
    b = pd.Series(a[:, 1]).ffill().to_numpy()
    k = pd.Series(a[:, 2]).ffill().to_numpy()
    idx = np.searchsorted(a[:, 0], t_grid, side="right") - 1
    ok = idx >= 0
    bid = np.where(ok, b[np.clip(idx, 0, None)], np.nan)
    ask = np.where(ok, k[np.clip(idx, 0, None)], np.nan)
    return bid, ask


# ---------- independent-sets model ----------
def match_from_s(s, bo):
    if bo == 3:
        return s * s * (3 - 2 * s)
    t = 1 - s
    return s ** 3 * (1 + 3 * t + 6 * t * t)


def s_from_match(M, bo):
    M = min(max(M, 1e-4), 1 - 1e-4)
    return brentq(lambda s: match_from_s(s, bo) - M, 1e-6, 1 - 1e-6)


def score_probs(s, bo):
    t = 1 - s
    if bo == 3:
        return {(2, 0): s * s, (2, 1): 2 * s * s * t, (0, 2): t * t, (1, 2): 2 * s * t * t}
    return {(3, 0): s ** 3, (3, 1): 3 * s ** 3 * t, (3, 2): 6 * s ** 3 * t * t,
            (0, 3): t ** 3, (1, 3): 3 * t ** 3 * s, (2, 3): 6 * t ** 3 * s * s}


def load_anchor():
    meta = {}
    for fn in ["data/kalshi/KXATPMATCH.jsonl", "data/kalshi/KXWTAMATCH.jsonl"]:
        for line in open(fn):
            r = json.loads(line)
            if None in r["names"]:
                continue
            meta[r["event"].split("-", 1)[1]] = r
    ms = {}
    for m in ka.load("data/kalshi/KXATPMATCH.jsonl") + ka.load("data/kalshi/KXWTAMATCH.jsonl"):
        code = m["event"].split("-", 1)[1]
        if code in meta:
            m["start"] = ka.match_start(m)
            m["names"] = meta[code]["names"]
            ms[code] = m
    return ms


def parse_exact(title, names):
    """'Jannik Sinner wins 3-1' -> (side_index_of_winner, (w, l))"""
    t = norm(title)
    try:
        sc = t.rsplit(" ", 1)[1]
        w, l = int(sc[0]), int(sc[2])
    except Exception:
        return None
    last = [norm(n).split()[-1] for n in names]
    hits = [i for i, ln in enumerate(last) if ln in t]
    if len(hits) != 1:
        return None
    return hits[0], (w, l)


def exact_analysis(anchor):
    rows, arbs = [], []
    for fn in glob.glob("data/kalshi/side_K*EXACTMATCH.jsonl"):
        for line in open(fn):
            r = json.loads(line)
            code = r["event"].split("-", 1)[1]
            m = anchor.get(code)
            if m is None or m["start"] is None:
                continue
            mk = []
            for x in r["markets"]:
                p = parse_exact(x["title"], m["names"])
                if p:
                    mk.append((p, x))
            if len(mk) not in (4, 6):
                continue
            bo = 3 if len(mk) == 4 else 5
            # outcome in side-0 orientation: (a_sets, b_sets)
            n = len(m["bid"])
            tg = m["t0"] + 60 * np.arange(n)
            K = (m["bid"] + m["ask"]) / 2
            q = {}
            for (side, (w, l)), x in mk:
                key = (w, l) if side == 0 else (l, w)
                bid, ask = series_at(x["candles"], tg)
                q[key] = (bid, ask, 1.0 if x["result"] == "yes" else 0.0, x["volume"])
            if sum(v[2] for v in q.values()) != 1:
                continue
            # ---- layer 1: model-free arbitrage over the whole window ----
            a_keys = [k for k in q if k[0] > k[1]]
            b_keys = [k for k in q if k[0] < k[1]]
            for i in range(max(m["start"] - 60, 0), n - 2):
                asks = {k: q[k][1][i] for k in q}
                if any(np.isnan(v) for v in asks.values()):
                    continue
                full = sum(a + fee(a) for a in asks.values())  # buy all scores
                # A scores + Kalshi NO on side 0 (= side-1 wins) via match market
                no_px = 1 - m["bid"][i]
                a_plus_no = sum(asks[k] + fee(asks[k]) for k in a_keys) + no_px + fee(no_px)
                yes_px = m["ask"][i]
                b_plus_yes = sum(asks[k] + fee(asks[k]) for k in b_keys) + yes_px + fee(yes_px)
                best = min(full, a_plus_no, b_plus_yes)
                if best < 1:
                    arbs.append(dict(ev=r["event"], i=i, inplay=i >= m["start"],
                                     profit=1 - best,
                                     kind=["all scores", "A-scores + match NO",
                                           "B-scores + match YES"][[full, a_plus_no, b_plus_yes].index(best)]))
            # ---- layer 2: model vs exact-score price, pre-match ----
            i = m["start"] - LEAD
            if i < 0 or m["ask"][i] - m["bid"][i] > 0.03:
                continue
            s = s_from_match(K[i], bo)
            model = score_probs(s, bo)
            for k, (bid, ask, y, vol) in q.items():
                if np.isnan(bid[i]) or np.isnan(ask[i]) or ask[i] - bid[i] > 0.10:
                    continue
                rows.append(dict(ev=r["event"], t=m["t0"], bo=bo, score=k, K=K[i],
                                 bid=bid[i], ask=ask[i], mid=(bid[i] + ask[i]) / 2,
                                 model=model[k], y=y, vol=vol))
    return pd.DataFrame(rows), pd.DataFrame(arbs)


def set1_analysis(anchor):
    rows = []
    for fn in glob.glob("data/kalshi/side_K*SETWINNER.jsonl"):
        for line in open(fn):
            r = json.loads(line)
            code = r["event"].split("-", 1)[1].rsplit("-", 1)[0]
            m = anchor.get(code)
            if m is None or m["start"] is None or len(r["markets"]) != 2:
                continue
            last0 = norm(m["names"][0]).split()[-1]
            x = [x for x in r["markets"] if last0 in norm(x["title"])]
            if len(x) != 1:
                continue
            x = x[0]
            n = len(m["bid"])
            tg = m["t0"] + 60 * np.arange(n)
            bid, ask = series_at(x["candles"], tg)
            i = m["start"] - LEAD
            if i < 0 or np.isnan(bid[i]) or np.isnan(ask[i]):
                continue
            if m["ask"][i] - m["bid"][i] > 0.03 or ask[i] - bid[i] > 0.10:
                continue
            bo = 5 if "3-" in "" else 3
            K = (m["bid"][i] + m["ask"][i]) / 2
            rows.append(dict(ev=r["event"], t=m["t0"], K=K, bid=bid[i], ask=ask[i],
                             mid=(bid[i] + ask[i]) / 2, y=1.0 if x["result"] == "yes" else 0.0,
                             s3=s_from_match(K, 3)))
    return pd.DataFrame(rows)


def ll(p, y):
    p = np.clip(p, 0.01, 0.99)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def bet_report(d, fair_col, label):
    med = d.t.median()
    for part, g in [("TRAIN (first half)", d[d.t < med]), ("TEST (second half)", d[d.t >= med])]:
        print(f"  -- {part}: n={len(g)}  log-loss market mid {ll(g.mid, g.y):.4f}  "
              f"{label} {ll(g[fair_col], g.y):.4f}")
        for thr in [0.02, 0.04, 0.07]:
            buy_yes = g[g[fair_col] - g.ask - fee(g.ask) > thr]
            buy_no = g[(1 - g[fair_col]) - (1 - g.bid) - fee(1 - g.bid) > thr]
            pnl = pd.concat([buy_yes.y - buy_yes.ask - fee(buy_yes.ask),
                             (1 - buy_no.y) - (1 - buy_no.bid) - fee(1 - buy_no.bid)])
            evs = pd.concat([buy_yes.ev, buy_no.ev])
            ka.fmt(f"     bet when {label} edge > {100*thr:.0f}c after fee", pnl, evs)


def main():
    anchor = load_anchor()
    print(f"anchor matches: {len(anchor)}\n")
    ex, arbs = exact_analysis(anchor)
    print("===== 1. Model-free arbitrage (exact score vs match market) =====")
    if len(arbs):
        a = arbs.groupby(["ev", "kind"]).agg(minutes=("i", "count"), best=("profit", "max"),
                                             inplay=("inplay", "mean")).reset_index()
        print(f"  events with any arb minute: {a.ev.nunique()} "
              f"(minutes: {len(arbs)}), by kind:")
        print(a.groupby("kind").agg(events=("ev", "nunique"), minutes=("minutes", "sum"),
                                    median_best_profit=("best", "median")).to_string())
    else:
        print("  none found")
    print()
    print("===== 2. Exact score: market vs independent-sets model (T-10m) =====")
    if len(ex):
        print(f"  {ex.ev.nunique()} events, {len(ex)} score-markets; "
              f"mean sum of mids per event {ex.groupby('ev').mid.sum().mean():.3f}")
        for bo, g in ex.groupby("bo"):
            print(f" best-of-{bo}:")
            # calibration of model vs market for straight-sets vs deciders
            g = g.assign(straight=g.score.map(lambda k: min(k) == 0))
            for st, gg in g.groupby("straight"):
                print(f"   {'straight sets' if st else 'went the distance'}: n={len(gg)} "
                      f"win {gg.y.mean():.3f}  market {gg.mid.mean():.3f}  model {gg.model.mean():.3f}")
            bet_report(g, "model", "model")
            # blended: shrink model toward market (fit nothing; fixed 50/50)
            g2 = g.assign(blend=0.5 * g.model + 0.5 * g.mid)
            bet_report(g2, "blend", "50/50 blend")
    print()
    print("===== 3. Set-1 winner vs anchor (T-10m) =====")
    s1 = set1_analysis(anchor)
    if len(s1):
        med = s1.t.median()
        tr = s1[s1.t < med]
        # fit logit(set1) = a + b*logit(s) on TRAIN outcomes
        from numpy.polynomial import polynomial as P
        lx = np.log(np.clip(tr.s3, .01, .99) / (1 - np.clip(tr.s3, .01, .99)))
        best = (0, 1, 9)
        for a in np.linspace(-0.2, 0.2, 21):
            for b in np.linspace(0.6, 1.2, 31):
                p = 1 / (1 + np.exp(-(a + b * lx)))
                l_ = ll(p, tr.y)
                if l_ < best[2]:
                    best = (a, b, l_)
        a, b, _ = best
        lx_all = np.log(np.clip(s1.s3, .01, .99) / (1 - np.clip(s1.s3, .01, .99)))
        s1["fair"] = 1 / (1 + np.exp(-(a + b * lx_all)))
        print(f"  {len(s1)} events; fitted on train: logit(set1) = {a:+.2f} + {b:.2f}*logit(s)")
        print(f"  avg set-1 price {s1.mid.mean():.3f}, model {s1.fair.mean():.3f}, "
              f"win {s1.y.mean():.3f}; mean spread {100*(s1.ask-s1.bid).mean():.1f}c")
        bet_report(s1, "fair", "anchor-model")


if __name__ == "__main__":
    main()
