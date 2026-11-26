"""Event-driven market-making simulator for Kalshi SET-1 winner markets.

Replays the historical trade tape (to the microsecond) of each set-1 market
and of its match-winner market (the anchor). The same quoting logic can later
be driven by live data.

Book reconstruction (Kalshi publishes no historical order books):
  - at each 1-minute candle close, best bid/ask = candle values
  - between candles, each trade updates it: a YES-buying taker at p means
    best ask was p; a YES-selling taker at p means best bid was p.
Quoting (decided at each event, ACTIVE only after LATENCY seconds; cancels
are delayed the same way):
  - no quotes if spread < MIN_SPREAD, mid outside [5c, 95c], inventory cap
    hit on that side, or within PULL_SECS after the anchor (match-winner)
    price moved >= PULL_MOVE inside 15s ("a point just happened")
  - spread >= 3c: improve both sides by 1c (we are alone at our price)
  - spread == 2c: improve the side that reduces inventory, JOIN the other
Fills (our orders never existed, so we bracket reality):
  optimistic : a taker trade at our price or better fills us
  harsh      : the trade must print >= 1c THROUGH our price
  very_harsh : >= 2c through
  joined (not improved) quotes always need >= 1c through (back of queue).
  Fill size = min(trade size, our remaining size). Remaining size resets when
  our quote price changes.
Accounting: maker fee ceil(0.0175*c*p*(1-p)) cents per fill; positions
settle at the set result; markouts at +60s/+300s vs reconstructed mid.
"""
import json
import math
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

MAKER = 0.0175


@dataclass
class Params:
    min_spread: float = 0.02
    pull_move: float = 0.02      # anchor move that triggers a pull (None = never)
    pull_secs: float = 20.0
    latency: float = 1.0
    size: int = 50
    cap: int = 300
    fill_rule: str = "harsh"     # optimistic | harsh | very_harsh
    band: float = 0.05           # quote only while mid in [band, 1-band]
    flatten_at: int = 10**9      # if |inv| > flatten_at, take liquidity down to flatten_to
    flatten_to: int = 0
    front_join: bool = False     # incumbent model: quote AT best price, first in queue
    exit_band: float = 0.0       # flatten everything once mid leaves [exit_band, 1-exit_band]


def norm(s):
    import unicodedata
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def fee(c, p):
    return math.ceil(MAKER * c * p * (1 - p) * 100 - 1e-9) / 100


def to_sec(strings):
    ts = pd.to_datetime(strings, format="ISO8601", utc=True)
    return ((ts - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy()


def load_events(series):
    side = {}
    for l in open(f"data/kalshi/side_{series}.jsonl"):
        r = json.loads(l)
        if len(r["markets"]) == 2:
            side[r["event"]] = r["markets"][0]
    trades = {}
    for l in open(f"data/kalshi/trades_side_{series}.jsonl"):
        r = json.loads(l)
        trades[r["event"]] = r
    anchors = {}
    try:
        for l in open(f"data/kalshi/anchor_trades_{series}.jsonl"):
            r = json.loads(l)
            anchors[r["event"]] = r
    except FileNotFoundError:
        pass
    evs = []
    for ev, tr in trades.items():
        mk = side.get(ev)
        if mk is None or not mk["candles"] or not tr["trades"]:
            continue
        t = to_sec([x[0] for x in tr["trades"]])
        tt = sorted(zip(t, [x[1] for x in tr["trades"]], [x[2] for x in tr["trades"]],
                        [x[3] for x in tr["trades"]]))
        an = anchors.get(ev)
        at = []
        if an and an["trades"]:
            flip = norm(an["match_name0"]).split()[-1] not in norm(mk["title"])
            ta = to_sec([x[0] for x in an["trades"]])
            at = sorted((t_, (1 - p) if flip else p) for t_, p in zip(ta, [x[1] for x in an["trades"]]))
        evs.append(dict(ev=ev, y=1.0 if mk["result"] == "yes" else 0.0,
                        candles=[c for c in mk["candles"] if c[1] is not None and c[2] is not None],
                        trades=tt, anchor=at, volume=sum(x[2] for x in tt)))
    evs.sort(key=lambda e: e["candles"][0][0])
    return evs


def simulate(e, P):
    """Returns dict with fills list and settled P&L for one set-1 market."""
    # merged timeline: (time, kind, payload); kind 0 = candle reset, 1 = anchor, 2 = trade
    tl = [(c[0], 0, (c[1], c[2])) for c in e["candles"]]
    tl += [(t, 1, p) for t, p in e["anchor"]]
    tl += [(t, 2, (p, c, s)) for t, p, c, s in e["trades"]]
    tl.sort(key=lambda x: (x[0], x[1]))
    bid = ask = None
    anchor_hist = deque()
    pulled_until = -1.0
    inv, cash = 0, 0.0
    versions = []          # (t_active, quote) quote = dict(side -> (price, kind)) or {}
    remaining = {}         # (side, price) -> remaining size
    mids_t, mids = [], []
    fills = []
    taker_c = [0]
    through = {"optimistic": 0.0, "harsh": 0.01, "very_harsh": 0.02}[P.fill_rule]

    def desired(t):
        if bid is None or ask is None:
            return {}
        spread = round(ask - bid, 4)
        mid = (bid + ask) / 2
        if spread < P.min_spread - 1e-9 or not (P.band <= mid <= 1 - P.band) or t < pulled_until:
            return {}
        q = {}
        if P.front_join:
            q["bid"] = (round(bid, 2), "front"); q["ask"] = (round(ask, 2), "front")
        elif spread >= 0.03 - 1e-9:
            q["bid"] = (round(bid + 0.01, 2), "imp")
            q["ask"] = (round(ask - 0.01, 2), "imp")
        else:
            if inv >= 0:
                q["ask"] = (round(ask - 0.01, 2), "imp"); q["bid"] = (round(bid, 2), "join")
            else:
                q["bid"] = (round(bid + 0.01, 2), "imp"); q["ask"] = (round(ask, 2), "join")
        if inv >= P.cap:
            q.pop("bid", None)
        if inv <= -P.cap:
            q.pop("ask", None)
        return q

    def active(t):
        i = bisect_right([v[0] for v in versions], t) - 1
        return versions[i][1] if i >= 0 else {}

    last_q = None
    for t, kind, x in tl:
        if kind == 0:
            bid, ask = x
        elif kind == 1:
            anchor_hist.append((t, x))
            while anchor_hist and anchor_hist[0][0] < t - 15:
                anchor_hist.popleft()
            if P.pull_move is not None:
                ps = [p for _, p in anchor_hist]
                if max(ps) - min(ps) >= P.pull_move - 1e-9:
                    pulled_until = t + P.pull_secs
        else:
            p, c, s = x
            # 1) fills against our ACTIVE quote (state before this trade)
            q = active(t)
            if s == "no" and "bid" in q:      # taker sold YES -> hits bids
                px, k = q["bid"]
                need = max(through, 0.01) if k == "join" else (0.0 if k == "front" else through)
                if p <= px - need + 1e-9:
                    rem = remaining.get(("bid", px), P.size)
                    f = min(rem, c, P.cap - inv)
                    if f > 0:
                        remaining[("bid", px)] = rem - f
                        inv += f; cash -= f * px + fee(f, px)
                        fills.append((t, +1, px, f))
            if s == "yes" and "ask" in q:     # taker bought YES -> lifts asks
                px, k = q["ask"]
                need = max(through, 0.01) if k == "join" else (0.0 if k == "front" else through)
                if p >= px + need - 1e-9:
                    rem = remaining.get(("ask", px), P.size)
                    f = min(rem, c, P.cap + inv)
                    if f > 0:
                        remaining[("ask", px)] = rem - f
                        inv -= f; cash += f * px - fee(f, px)
                        fills.append((t, -1, px, f))
            # 2) book update from the print
            if s == "yes":
                ask = p
                if bid is not None and bid >= p:
                    bid = round(p - 0.01, 2)
            else:
                bid = p
                if ask is not None and ask <= p:
                    ask = round(p + 0.01, 2)
        if bid is not None and ask is not None:
            mids_t.append(t); mids.append((bid + ask) / 2)
            mid_now = (bid + ask) / 2
            target = None
            if abs(inv) > P.flatten_at:
                target = P.flatten_to
            if P.exit_band > 0 and not (P.exit_band <= mid_now <= 1 - P.exit_band) and inv != 0:
                target = 0
            if target is not None and kind == 2:   # act on a trade event (book is live)
                if inv > target:          # sell YES at the bid as taker
                    q_ = inv - target
                    cash += q_ * bid - math.ceil(0.07 * q_ * bid * (1 - bid) * 100 - 1e-9) / 100
                    inv -= q_; taker_c[0] += q_
                elif inv < -target:       # buy YES at the ask as taker
                    q_ = -target - inv
                    cash -= q_ * ask + math.ceil(0.07 * q_ * ask * (1 - ask) * 100 - 1e-9) / 100
                    inv += q_; taker_c[0] += q_
        dq = desired(t)
        if dq != last_q:
            # price change resets remaining size for that side
            for side_, (px, _) in dq.items():
                if last_q is None or side_ not in last_q or last_q[side_][0] != px:
                    remaining[(side_, px)] = P.size
            versions.append((t + P.latency, dq))
            last_q = dq

    pnl = cash + inv * e["y"]
    mk = []
    for t, d, px, f in fills:
        out = []
        for h in (60, 300):
            j = bisect_right(mids_t, t + h) - 1
            m = mids[j] if j >= 0 else px
            out.append(d * (m - px) * f)
        mk.append(out)
    contracts = sum(f for *_, f in fills)
    return dict(ev=e["ev"], pnl=pnl, contracts=contracts, n_fills=len(fills), taker=taker_c[0],
                fees=sum(fee(f, px) for _, _, px, f in fills),
                m60=sum(m[0] for m in mk), m300=sum(m[1] for m in mk),
                max_inv=max([0] + [abs(v) for v in np.cumsum([d * f for _, d, _, f in fills])]),
                volume=e["volume"], t0=e["candles"][0][0])


def run(evs, P):
    return pd.DataFrame([simulate(e, P) for e in evs])


def summarize(r, label):
    c = r.contracts.sum()
    per = r.pnl.sum() / c * 100 if c else 0
    # bootstrap per-event P&L for CI of total-per-contract
    rng = np.random.default_rng(0)
    pn, cn = r.pnl.to_numpy(), r.contracts.to_numpy()
    bs = []
    for _ in range(400):
        i = rng.integers(0, len(r), len(r))
        bs.append(pn[i].sum() / max(cn[i].sum(), 1) * 100)
    se = np.std(bs)
    cum = r.sort_values("t0").pnl.cumsum()
    dd = (cum.cummax() - cum).max()
    print(f"  {label:34s} events={len(r):4d} fills={r.n_fills.sum():6d} "
          f"contracts={c:9.0f} share={c / r.volume.sum():5.1%} | "
          f"P&L ${r.pnl.sum():8.0f} = {per:+.2f}c/contract (±{1.96*se:.2f}) "
          f"| +1m {r.m60.sum()/max(c,1)*100:+.2f}c +5m {r.m300.sum()/max(c,1)*100:+.2f}c "
          f"| win-events {(r.pnl > 0).mean():.0%} maxDD ${dd:.0f} max|inv| {r.max_inv.max():.0f}")
