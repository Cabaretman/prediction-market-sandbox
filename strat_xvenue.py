"""Strategy #3: Kalshi vs Polymarket on the same tennis matches.

Per minute (in-play, Kalshi-detected start -> Kalshi close):
  K = Kalshi mid (bid/ask known), P = Polymarket price (prices-history; no
  bid/ask, so we ASSUME a Polymarket half-spread of POLY_HALF).
Tests
  a) How big / how persistent are K-P gaps?
  b) Lead-lag: does one venue's move predict the other's next move?
  c) Locked arbitrage: buy YES on the cheap venue + NO on the other when the
     all-in cost < $1 (both taker, incl. fees).
  d) Directional: when |P-K| is large, take Kalshi towards P (Kalshi ask/bid
     + fee); mark at +5 min Kalshi mid and at settlement.
  e) Which venue is better calibrated (log-loss at sampled minutes)?
"""
import json
import glob
import unicodedata

import numpy as np
import pandas as pd

import kalshi_analyze as ka

K_TAKER, P_TAKER = 0.07, 0.05  # Polymarket sports taker (0.03-0.05; use 0.05)
POLY_HALF = 0.01


def norm(name):
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return n.replace("-", " ").split()[-1]


def pairs():
    raw = {}
    for fn in glob.glob("data/kalshi/KX*MATCH.jsonl"):
        for line in open(fn):
            r = json.loads(line)
            if None in r["names"]:
                continue
            raw[r["event"]] = r
    kidx = {}
    for ev, r in raw.items():
        ct = int(pd.Timestamp(r["close_time"]).timestamp())
        key = frozenset(norm(x) for x in r["names"])
        for dd in (ct // 86400 - 1, ct // 86400, ct // 86400 + 1):
            kidx[(dd, key)] = ev
    kms = {m["event"]: m for m in ka.load("data/kalshi/KX*MATCH.jsonl")}
    out = []
    for line in open("data/poly/tennis.jsonl"):
        p = json.loads(line)
        key = (p["closed"] // 86400, frozenset(norm(o) for o in p["outcomes"]))
        ev = kidx.get(key)
        if ev is None or ev not in kms or len(p["hist"]) < 20:
            continue
        m, r = kms[ev], raw[ev]
        # orient Polymarket outcome to Kalshi side 0
        flip = norm(p["outcomes"][0]) != norm(r["names"][0])
        pw = 1 - p["winner"] if flip else p["winner"]
        y_poly = 1.0 if pw == 0 else 0.0
        if y_poly != m["y"]:
            continue  # settlement disagreement / mismatch: skip
        h = np.array(p["hist"], dtype=float)
        n = len(m["bid"])
        tgrid = m["t0"] + 60 * np.arange(n)
        idx = np.searchsorted(h[:, 0], tgrid, side="right") - 1
        P = np.where(idx >= 0, h[np.clip(idx, 0, None), 1], np.nan)
        # stale if last poly print older than 3 minutes
        age = tgrid - np.where(idx >= 0, h[np.clip(idx, 0, None), 0], -1e18)
        P[age > 180] = np.nan
        if flip:
            P = 1 - P
        m["P"] = P
        m["start"] = ka.match_start(m)
        if m["start"] is not None:
            out.append(m)
    return out


def main():
    ms = pairs()
    print(f"{len(ms)} matched matches (Kalshi + Polymarket)\n")
    rows = []
    for m in ms:
        s = m["start"]
        bid, ask, P, y = m["bid"], m["ask"], m["P"], m["y"]
        K = (bid + ask) / 2
        n = len(K)
        for i in range(s, n - 6):
            if np.isnan(P[i]) or ask[i] - bid[i] > ka.MAX_SPREAD:
                continue
            if not (0.05 < K[i] < 0.95):
                continue
            rows.append(dict(
                ev=m["event"], K=K[i], P=P[i], bid=bid[i], ask=ask[i], y=y,
                dK_next=K[i + 1] - K[i], dP_prev=P[i] - P[i - 1] if not np.isnan(P[i - 1]) else np.nan,
                dK_prev=K[i] - K[i - 1], dP_next=P[i + 1] - P[i] if not np.isnan(P[i + 1]) else np.nan,
                K5=K[i + 5], gap_next=(P[i + 1] - K[i + 1]) if not np.isnan(P[i + 1]) else np.nan))
    d = pd.DataFrame(rows)
    d["gap"] = d.P - d.K
    print(f"{len(d)} in-play minutes with both prices\n")

    print("== a) Gap P-K distribution (cents) ==")
    print((100 * d.gap.abs()).describe(percentiles=[.5, .75, .9, .95, .99]).round(2).to_string())
    big = d[d.gap.abs() >= 0.04]
    print(f"  minutes with |gap|>=4c: {len(big)} ({len(big)/len(d):.1%}); "
          f"still >=4c one minute later: {(big.gap_next.abs() >= 0.04).mean():.0%}\n")

    print("== b) Lead-lag (corr of one venue's last-minute move with the other's next move) ==")
    x = d.dropna(subset=["dP_prev", "dP_next"])
    print(f"  corr(dP[t-1->t], dK[t->t+1]) = {x.dP_prev.corr(x.dK_next):+.3f}   (Poly leads Kalshi?)")
    print(f"  corr(dK[t-1->t], dP[t->t+1]) = {x.dK_prev.corr(x.dP_next):+.3f}   (Kalshi leads Poly?)")
    print(f"  corr(gap_t, dK[t->t+1])      = {x.gap.corr(x.dK_next):+.3f}   (Kalshi closes gap?)")
    print(f"  corr(-gap_t, dP[t->t+1])     = {(-x.gap).corr(x.dP_next):+.3f}   (Poly closes gap?)\n")

    print("== c) Locked arbitrage, both legs taker (Poly half-spread assumed "
          f"{100*POLY_HALF:.0f}c) ==")
    # buy YES on Kalshi + NO on Poly, or YES on Poly + NO on Kalshi
    pyes = d.P + POLY_HALF
    pno = 1 - d.P + POLY_HALF
    c1 = d.ask + K_TAKER * d.ask * (1 - d.ask) + pno + P_TAKER * pno * (1 - pno)
    kno = 1 - d.bid
    c2 = pyes + P_TAKER * pyes * (1 - pyes) + kno + K_TAKER * kno * (1 - kno)
    edge = 1 - np.minimum(c1, c2)
    for thr in [0, 0.01, 0.02]:
        e = edge[edge > thr]
        print(f"  minutes with locked profit > {100*thr:.0f}c: {len(e)} "
              f"({len(e)/len(d):.2%}) in {d.ev[edge > thr].nunique()} matches, "
              f"avg {100*e.mean() if len(e) else 0:.2f}c")
    print()

    print("== d) Directional: take Kalshi towards Polymarket when |gap| big ==")
    for thr in [0.02, 0.03, 0.05, 0.08]:
        g = d[d.gap.abs() >= thr].copy()
        g = g.groupby("ev").head(10)  # limit to first 10 signals per match
        up = g.gap > 0
        px = np.where(up, g.ask, 1 - g.bid)
        fee = K_TAKER * px * (1 - px)
        yy = np.where(up, g.y, 1 - g.y)
        k5 = np.where(up, g.K5, 1 - g.K5)
        print(f" |gap|>={100*thr:.0f}c:")
        ka.fmt("  mark at Kalshi mid +5m", k5 - px - fee, g.ev.to_numpy())
        ka.fmt("  hold to settlement", yy - px - fee, g.ev.to_numpy())
    print()

    print("== e) Calibration: log-loss at sampled minutes (lower = better) ==")
    s = d.groupby("ev").sample(frac=0.1, random_state=0)
    def ll(p, yv):
        p = np.clip(p, 0.01, 0.99)
        return -np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p))
    print(f"  Kalshi mid {ll(s.K, s.y):.4f}   Polymarket {ll(s.P, s.y):.4f}   "
          f"avg {ll((s.K + s.P) / 2, s.y):.4f}")


if __name__ == "__main__":
    main()
