"""Paper watcher: pre-match heavy favourites (95-97c) on Kalshi tennis.

One `tick` (run every ~10 min, e.g. by GitHub Actions):
  1. list open match-winner markets in all tennis series
  2. for matches 5-60 min before their real scheduled start (Kalshi milestone
     start_date; the market's occurrence_datetime is ~3h LATER than the start)
     and whose milestone status is still "not_started", read the order book
  3. log a snapshot of the favourite (any price >= 80c) to snapshots.jsonl
  4. if favourite mid in [95c, 97c] and spread <= 3c, and no entry yet for
     this match, log a PAPER BUY at the real ask with the depth offered there
  5. settle logged entries whose markets have resolved
  6. rewrite REPORT.md
No account, no API key, no orders: read-only public endpoints.
Usage: python3 paper/watcher.py tick        (single pass)
       python3 paper/watcher.py loop        (tick every 10 min, forever)
"""
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = ["KXATPMATCH", "KXWTAMATCH", "KXATPCHALLENGERMATCH",
          "KXWTACHALLENGERMATCH", "KXITFMATCH", "KXITFWMATCH"]
LO, HI, MAX_SPREAD = 0.95, 0.97, 0.03
PREMATCH_STATUSES = {"not_started", "SCH", "scheduled"}
WIN_EARLY, WIN_LATE = 60 * 60, 5 * 60      # seconds before the milestone start_date
STAKE_DOLLARS = 200                        # paper stake cap per match
TAKER = 0.07
HERE = os.path.dirname(os.path.abspath(__file__))
ENTRIES = os.path.join(HERE, "entries.jsonl")
SNAPS = os.path.join(HERE, "snapshots.jsonl")
REPORT = os.path.join(HERE, "REPORT.md")


def get(path, params=None):
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "paper-watcher/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return None


def now():
    return time.time()


def iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def fee(c, p):
    return math.ceil(TAKER * c * p * (1 - p) * 100 - 1e-9) / 100


def read_jsonl(fn):
    if not os.path.exists(fn):
        return []
    with open(fn) as f:
        return [json.loads(l) for l in f if l.strip()]


def open_events():
    ev = {}
    for s in SERIES:
        cursor = None
        while True:
            p = {"series_ticker": s, "status": "open", "limit": 1000}
            if cursor:
                p["cursor"] = cursor
            d = get("/markets", p) or {}
            for m in d.get("markets", []):
                ev.setdefault(m["event_ticker"], []).append(m)
            cursor = d.get("cursor")
            if not cursor or not d.get("markets"):
                break
    return {e: mk for e, mk in ev.items() if len(mk) == 2}


def milestone(event):
    d = get("/milestones", {"related_event_ticker": event, "limit": 5}) or {}
    for m in d.get("milestones", []):
        if m.get("type") == "tennis_tournament_singles":
            return iso(m["start_date"]), (m.get("details") or {}).get("status")
    return None, None


def book(ticker):
    """Best YES bid/ask and the size offered at the ask (and within +1c)."""
    d = get(f"/markets/{ticker}/orderbook", {"depth": 10}) or {}
    ob = d.get("orderbook_fp") or {}
    yes = [(float(p), float(q)) for p, q in (ob.get("yes_dollars") or [])]
    no = [(float(p), float(q)) for p, q in (ob.get("no_dollars") or [])]
    if not yes or not no:
        return None
    best_bid = max(p for p, _ in yes)
    best_no = max(p for p, _ in no)
    ask = round(1 - best_no, 4)
    # YES asks are NO bids: NO bid at q  <=>  YES offered at 1-q
    at_ask = sum(q for p, q in no if abs((1 - p) - ask) < 1e-6)
    within_1c = sum(q for p, q in no if (1 - p) <= ask + 0.01 + 1e-6)
    return dict(bid=best_bid, ask=ask, size_at_ask=at_ask, size_within_1c=within_1c)


def tick():
    t = now()
    entries = read_jsonl(ENTRIES)
    have = {e["event"] for e in entries}
    snapped = {s["event"] for s in read_jsonl(SNAPS)}
    evs = open_events()
    n_window = n_new = 0
    with open(SNAPS, "a") as fs, open(ENTRIES, "a") as fe:
        for e, mk in evs.items():
            occ = mk[0].get("occurrence_datetime")
            # cheap prefilter: occurrence is ~3h after the real start
            if not occ or not (0 <= iso(occ) - t <= 6 * 3600):
                continue
            start, status = milestone(e)
            if start is None:
                continue
            dt = start - t
            if not (WIN_LATE <= dt <= WIN_EARLY) or status not in PREMATCH_STATUSES:
                continue
            n_window += 1
            # favourite = market with the higher last/bid price
            mk.sort(key=lambda m: float(m.get("yes_bid_dollars") or 0), reverse=True)
            fav = mk[0]
            b = book(fav["ticker"])
            if not b:
                continue
            mid = (b["bid"] + b["ask"]) / 2
            snap = dict(t=round(t), event=e, series=fav["ticker"].split("-")[0],
                        ticker=fav["ticker"], player=fav.get("yes_sub_title"),
                        mins_to_start=round(dt / 60, 1), **b, mid=mid)
            if mid >= 0.80 and e not in snapped:
                fs.write(json.dumps(snap) + "\n")
                snapped.add(e)
            if e in have:
                continue
            if LO <= mid <= HI and b["ask"] - b["bid"] <= MAX_SPREAD:
                want = int(STAKE_DOLLARS / b["ask"])
                contracts = int(min(want, b["size_at_ask"]))
                if contracts <= 0:
                    continue
                entry = dict(snap, contracts=contracts, cost=round(contracts * b["ask"], 2),
                             fee=fee(contracts, b["ask"]), status="open")
                fe.write(json.dumps(entry) + "\n")
                have.add(e)
                n_new += 1
    settle()
    report()
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC: {len(evs)} open matches, "
          f"{n_window} in pre-match window, {n_new} new paper entries")


def settle():
    entries = read_jsonl(ENTRIES)
    changed = False
    for e in entries:
        if e["status"] != "open":
            continue
        d = get(f"/markets/{e['ticker']}") or {}
        m = d.get("market") or {}
        res = m.get("result")
        if res in ("yes", "no"):
            y = 1.0 if res == "yes" else 0.0
            e["status"] = "won" if y == 1 else "lost"
            e["pnl"] = round(e["contracts"] * (y - e["ask"]) - e["fee"], 2)
            e["settled_t"] = round(now())
            changed = True
        elif m.get("status") in ("finalized", "settled") and res not in ("yes", "no"):
            e["status"] = "void"   # walkover / fair-price settlement
            e["pnl"] = 0.0
            changed = True
    if changed:
        with open(ENTRIES, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")


def report():
    E = read_jsonl(ENTRIES)
    S = read_jsonl(SNAPS)
    done = [e for e in E if e["status"] in ("won", "lost")]
    won = sum(e["status"] == "won" for e in done)
    pnl = sum(e["pnl"] for e in done)
    cost = sum(e["cost"] for e in done)
    contracts = sum(e["contracts"] for e in done)
    lines = [
        "# Paper watcher: pre-match favourites at 95-97c",
        "",
        f"_Updated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC._ "
        "Rule: favourite mid 95-97c, spread <= 3c, 5-60 min before the official start, match not started; "
        f"paper-buy at the real ask, up to ${STAKE_DOLLARS} or the size offered at the ask.",
        "",
        "| | |",
        "|---|---|",
        f"| Entries logged | {len(E)} ({len(E) - len(done)} open) |",
        f"| Settled | {len(done)}: **{won} won / {len(done) - won} lost** |",
        f"| Win rate vs avg price | {won / len(done):.1%} vs {sum(e['ask'] for e in done) / len(done):.1%} |" if done else "| Win rate | n/a |",
        f"| Paper P&L | **${pnl:,.2f}** on ${cost:,.2f} staked ({100 * pnl / cost:+.2f}%) |" if cost else "| Paper P&L | n/a |",
        f"| P&L per contract | {100 * pnl / contracts:+.2f}c (backtest: about +2c) |" if contracts else "| P&L per contract | n/a |",
        f"| Median size offered at the ask | {sorted(e['size_at_ask'] for e in E)[len(E) // 2]:,.0f} contracts |" if E else "| Size at ask | n/a |",
        f"| Pre-match favourite snapshots (>=80c) logged | {len(S)} |",
        "",
        "## Recent entries",
        "",
        "| time (UTC) | match | player | ask | size at ask | contracts | result | P&L |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in sorted(E, key=lambda x: x["t"], reverse=True)[:30]:
        lines.append(f"| {datetime.fromtimestamp(e['t'], timezone.utc):%m-%d %H:%M} | {e['event']} | "
                     f"{e.get('player')} | {e['ask']:.2f} | {e['size_at_ask']:,.0f} | {e['contracts']} | "
                     f"{e['status']} | {'' if 'pnl' not in e else f'${e['pnl']:,.2f}'} |")
    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "tick"
    if mode == "loop":
        while True:
            try:
                tick()
            except Exception as ex:
                print("tick error:", ex, flush=True)
            time.sleep(600)
    else:
        tick()
