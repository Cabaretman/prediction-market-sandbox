"""Who wins in Kalshi politics? Trade-tape settlement P&L.

Every trade: taker bought side S at price q (q = yes_price if S=yes else 1-yes_price).
Taker P&L/contract = outcome(S) - q - 0.07 q(1-q). The resting maker holds the other
side: maker P&L = -(outcome(S) - q) - 0.0175 q(1-q) (conservative maker fee).
Broken down by price of the side the taker bought, market age, days to close,
trade size, market family. Clustered t by market. Midterm (2026-Q4) excluded
in the 'ex-midterm' rows; early/late = halves by close date.
"""
import json
import re

import numpy as np
import pandas as pd

TK, MK = 0.07, 0.0175


def family(series):
    s = series.upper()
    for k in ["MENTION", "APPROV", "APRPOTUS", "TRUTH", "TWEET", "PRIMARY", "MIDTERM", "HOUSE",
              "SENATE", "GOV", "MAYOR", "NOMINEE", "NOM", "CABINET", "SEC", "EO", "BILL",
              "SCOTUS", "COURT", "TARIFF", "POLL", "POPVOTE", "ELECT", "PRES", "SHUTDOWN"]:
        if k in s:
            return k
    return "OTHER"


def load():
    meta = {}
    for l in open("data/politics/markets.jsonl"):
        m = json.loads(l)
        meta[m["ticker"]] = m
    nmk = pd.Series([m["event"] for m in meta.values()]).value_counts()
    rows = []
    for l in open("data/politics/trades.jsonl"):
        r = json.loads(l)
        m = meta.get(r["ticker"])
        if m is None or not r["trades"]:
            continue
        op = pd.Timestamp(m["open"]).timestamp()
        cl = pd.Timestamp(m["close"]).timestamp()
        y = 1.0 if m["result"] == "yes" else 0.0
        a = pd.DataFrame(r["trades"], columns=["t", "yp", "n", "side"])
        a = a[(a.yp > 0) & (a.yp < 1) & a.side.isin(["yes", "no"])]
        if a.empty:
            continue
        a["q"] = np.where(a.side == "yes", a.yp, 1 - a.yp)
        a["win"] = np.where(a.side == "yes", y, 1 - y)
        a["ticker"], a["event"], a["series"] = m["ticker"], m["event"], m["series"]
        a["fam"] = family(m["series"])
        a["multi"] = nmk.get(m["event"], 1) > 2
        a["age"] = (a.t - op) / max(cl - op, 1)
        a["dtc"] = (cl - a.t) / 86400
        a["close"] = cl
        a["elec"] = bool({"US Elections", "Primaries", "Senate", "House", "Governor"} & set(m["tags"]))
        rows.append(a)
    d = pd.concat(rows, ignore_index=True)
    d["gross"] = d.win - d.q                      # taker gross per contract
    d["taker"] = d.gross - TK * d.q * (1 - d.q)
    d["maker"] = -d.gross - MK * d.q * (1 - d.q)
    d["qbin"] = pd.cut(d.q, [0, .03, .05, .10, .20, .35, .50, .65, .80, .90, .95, .97, 1])
    d["agebin"] = pd.cut(d.age, [-1, .02, .1, .3, .6, .9, 2])
    d["dtcbin"] = pd.cut(d.dtc, [-1, 1, 7, 30, 90, 1e5])
    d["size"] = pd.cut(d.n, [0, 10, 100, 1000, 1e9], labels=["<10", "10-100", "100-1k", "1k+"])
    d["qtr"] = pd.to_datetime(d.close, unit="s").dt.to_period("Q").astype(str)
    return d


def tstat(g, col):
    # contract-weighted mean; t clustered by market
    w = g.n
    per = (g[col] * w).groupby(g.ticker).sum() / w.groupby(g.ticker).sum()
    pw = w.groupby(g.ticker).sum()
    mean = (g[col] * w).sum() / w.sum()
    if len(per) < 3:
        return mean, 0.0
    # cluster-robust SE for weighted mean
    resid = ((per - mean) * pw)
    se = np.sqrt((resid ** 2).sum()) / pw.sum()
    return mean, mean / se if se > 0 else 0.0


def table(d, by, label):
    med = d.close.median()
    print(f"--- {label} (by {by}) ---")
    print(f"{by:16s} {'trades':>8s} {'mkts':>5s} {'contracts':>10s} | {'taker':>8s} {'t':>6s} | {'maker':>8s} "
          f"{'t':>6s} | {'early':>7s} {'late':>7s} | {'ex-mid':>7s}")
    for k, g in d.groupby(by, observed=True):
        if g.ticker.nunique() < 15:
            continue
        tm, tt = tstat(g, "taker")
        mm, mt = tstat(g, "maker")
        e, l, x = g[g.close < med], g[g.close >= med], g[g.qtr != "2026Q4"]
        f = lambda z: f"{100 * tstat(z, 'maker')[0]:+6.2f}c" if z.ticker.nunique() >= 5 else "    n/a"
        print(f"{str(k):16s} {len(g):8d} {g.ticker.nunique():5d} {g.n.sum():10.0f} | {100 * tm:+7.2f}c {tt:+6.1f} | "
              f"{100 * mm:+7.2f}c {mt:+6.1f} | {f(e)} {f(l)} | {f(x)}")
    print()


def main():
    d = load()
    print(f"{len(d)} trades, {d.ticker.nunique()} markets, {d.n.sum():,.0f} contracts\n")
    table(d, "qbin", "ALL: price of the side the TAKER bought")
    table(d, "agebin", "ALL: market age when traded (0=open, 1=close)")
    table(d, "dtcbin", "ALL: days to close")
    table(d, "size", "ALL: trade size")
    table(d, "fam", "ALL: market family")
    table(d, "qtr", "ALL: quarter")
    lo = d[d.q < 0.35]
    table(lo, "fam", "Takers buying the CHEAP side (<35c), by family")
    table(lo, "agebin", "Takers buying the CHEAP side (<35c), by age")
    hi = d[d.q > 0.65]
    table(hi, "fam", "Takers buying the EXPENSIVE side (>65c), by family")
    d.to_pickle("data/politics/tape.pkl")


if __name__ == "__main__":
    main()
