"""Live order-book recorder: Kalshi vs Polymarket, same tennis matches.

Every REFRESH seconds: list open Kalshi match markets and open Polymarket
tennis moneylines, pair them by (player last names). For pairs whose
scheduled start is within the past 6h / next 15min, poll both top-of-book
every POLL seconds and append to data/live/books_<date>.jsonl:
  {t, ev, k: [yes_bid, yes_bid_sz, yes_ask, yes_ask_sz],
          p: [bid, bid_sz, ask, ask_sz]  (oriented to Kalshi side-0 player)}
This lets us test whether Polymarket's lag is tradeable at REAL quotes.
"""
import json
import os
import time
from datetime import datetime, timezone

import poly_collect as pc
from kalshi_collect import get as kget

SERIES = ["KXATPMATCH", "KXWTAMATCH", "KXATPCHALLENGERMATCH", "KXITFMATCH",
          "KXWTACHALLENGERMATCH", "KXITFWMATCH"]
REFRESH, POLL = 120, 3


def now():
    return time.time()


def iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def kalshi_open():
    ev = {}
    for s in SERIES:
        cursor = None
        while True:
            p = {"series_ticker": s, "status": "open", "limit": 1000}
            if cursor:
                p["cursor"] = cursor
            d = kget("/markets", p) or {}
            for m in d.get("markets", []):
                ev.setdefault(m["event_ticker"], []).append(m)
            cursor = d.get("cursor")
            if not cursor or not d.get("markets"):
                break
    out = {}
    for e, mk in ev.items():
        if len(mk) != 2 or any(m.get("yes_sub_title") is None for m in mk):
            continue
        mk.sort(key=lambda m: m["ticker"])
        t = mk[0].get("occurrence_datetime") or mk[0].get("expected_expiration_time")
        out[frozenset(pc.norm(m["yes_sub_title"]) for m in mk)] = dict(
            ev=e, ticker=mk[0]["ticker"], name0=mk[0]["yes_sub_title"],
            start=iso(t) if t else None)
    return out


def poly_open():
    out, cursor = {}, None
    while True:
        p = {"tag_slug": "tennis", "closed": "false", "limit": 100,
             "order": "startTime", "ascending": "false"}
        if cursor:
            p["after_cursor"] = cursor
        d = pc.get(pc.GAMMA + "/events/keyset", p) or {}
        evs = d.get("events", [])
        stop = False
        for e in evs:
            st = e.get("startTime")
            if st and iso(st) < now() - 8 * 3600:
                stop = True
                break
            if "/" in e["title"]:
                continue
            ml = [m for m in e.get("markets", []) if m.get("sportsMarketType") == "moneyline"]
            if len(ml) != 1:
                continue
            try:
                outs = json.loads(ml[0]["outcomes"])
                toks = json.loads(ml[0]["clobTokenIds"])
            except Exception:
                continue
            out[frozenset(pc.norm(o) for o in outs)] = dict(
                title=e["title"], outcomes=outs, tokens=toks)
        cursor = d.get("next_cursor")
        if stop or not cursor or not evs:
            break
    return out


def kbook(ticker):
    d = kget(f"/markets/{ticker}/orderbook", {"depth": 1}) or {}
    ob = d.get("orderbook_fp") or {}
    yes, no = ob.get("yes_dollars") or [], ob.get("no_dollars") or []
    yb = max(yes, key=lambda x: float(x[0])) if yes else None
    nb = max(no, key=lambda x: float(x[0])) if no else None
    return [float(yb[0]) if yb else None, float(yb[1]) if yb else 0,
            1 - float(nb[0]) if nb else None, float(nb[1]) if nb else 0]


def pbook(token):
    d = pc.get(pc.CLOB + "/book", {"token_id": token}) or {}
    bids, asks = d.get("bids") or [], d.get("asks") or []
    b = max(bids, key=lambda x: float(x["price"])) if bids else None
    a = min(asks, key=lambda x: float(x["price"])) if asks else None
    return [float(b["price"]) if b else None, float(b["size"]) if b else 0,
            float(a["price"]) if a else None, float(a["size"]) if a else 0]


def main():
    os.makedirs("data/live", exist_ok=True)
    pairs, last = [], 0
    while True:
        if now() - last > REFRESH:
            try:
                k, p = kalshi_open(), poly_open()
                pairs = []
                for key, km in k.items():
                    if key in p and km["start"] and \
                            km["start"] - 15 * 60 < now() < km["start"] + 6 * 3600:
                        pm = p[key]
                        idx = 0 if pc.norm(pm["outcomes"][0]) == pc.norm(km["name0"]) else 1
                        pairs.append((km, pm["tokens"][idx]))
                print(f"{datetime.now(timezone.utc):%H:%M} kalshi={len(k)} "
                      f"poly={len(p)} paired-live={len(pairs)}", flush=True)
            except Exception as e:
                print("refresh error", e, flush=True)
            last = now()
        if not pairs:
            time.sleep(30)
            continue
        t0 = now()
        fn = f"data/live/books_{datetime.now(timezone.utc):%Y%m%d}.jsonl"
        with open(fn, "a") as f:
            for km, tok in pairs:
                try:
                    rec = {"t": round(now(), 2), "ev": km["ev"],
                           "k": kbook(km["ticker"]), "p": pbook(tok)}
                    f.write(json.dumps(rec) + "\n")
                except Exception as e:
                    print("poll error", km["ev"], e, flush=True)
        time.sleep(max(0, POLL - (now() - t0)))


if __name__ == "__main__":
    main()
