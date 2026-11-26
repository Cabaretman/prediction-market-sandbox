"""Download closed Polymarket tennis singles moneyline markets + 1-min prices.

Output: data/poly/tennis.jsonl (one line per match). Resumable.
Price history is the CLOB 'prices-history' series (mid/last, no bid/ask).
"""
import json
import os
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
SINCE = "2025-06-01"
_last = [0.0]


def get(url, params):
    url = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(8):
        w = 0.15 - (time.time() - _last[0])
        if w > 0:
            time.sleep(w)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=40) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    import gzip
                    raw = gzip.decompress(raw)
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            if e.code in (429,) or e.code >= 500:
                time.sleep(2 ** attempt); continue
            return None
        except Exception:
            time.sleep(2 ** attempt)
    return None


def ts(s):
    s = s.replace(" ", "T").replace("+00", "+00:00") if s.endswith("+00") else s.replace(" ", "T")
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def norm(name):
    import unicodedata
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return n.replace("-", " ").split()[-1] if n.split() else n


def kalshi_index():
    """(day, frozenset(last names)) for every settled Kalshi tennis match,
    built from Kalshi's market listings (cached to data/kalshi/index.json)."""
    from kalshi_collect import list_markets
    cache = "data/kalshi/index.json"
    if os.path.exists(cache):
        rows = json.load(open(cache))
    else:
        rows, ev = [], {}
        for ser in ["KXATPMATCH", "KXWTAMATCH", "KXATPCHALLENGERMATCH"]:
            for m in list_markets(ser):
                ev.setdefault(m["event_ticker"], []).append(m)
        for e, mk in ev.items():
            names = [m.get("yes_sub_title") for m in mk]
            if len(mk) == 2 and None not in names:
                rows.append([e, ts(mk[0]["close_time"]), names])
        json.dump(rows, open(cache, "w"))
    idx = set()
    for e, ct, names in rows:
        d = ct // 86400
        key = frozenset(norm(x) for x in names)
        for dd in (d - 1, d, d + 1):
            idx.add((dd, key))
    return idx


def main():
    os.makedirs("data/poly", exist_ok=True)
    kidx = kalshi_index()
    print(f"{len(kidx)//3} Kalshi matches indexed", flush=True)
    fn = "data/poly/tennis.jsonl"
    done = set()
    if os.path.exists(fn):
        with open(fn) as f:
            done = {json.loads(l)["id"] for l in f}
    cursor, n_ev, n_new = None, 0, 0
    with open(fn, "a") as out:
        while True:
            p = {"tag_slug": "tennis", "closed": "true", "limit": 50,
                 "order": "startTime", "ascending": "false"}
            if cursor:
                p["after_cursor"] = cursor
            d = get(GAMMA + "/events/keyset", p)
            if not d or not d.get("events"):
                break
            stop = False
            for e in d["events"]:
                n_ev += 1
                st = e.get("startTime") or ""
                if st and st[:10] < SINCE:
                    stop = True
                    break
                if "/" in e["title"] or e["id"] in done:
                    continue
                ml = [m for m in e.get("markets", [])
                      if m.get("sportsMarketType") == "moneyline"]
                if len(ml) != 1:
                    continue
                m = ml[0]
                try:
                    outcomes = json.loads(m["outcomes"])
                    prices = [float(x) for x in json.loads(m["outcomePrices"])]
                    toks = json.loads(m["clobTokenIds"])
                    gs = ts(m.get("gameStartTime") or st)
                    ct = ts(m["closedTime"])
                except Exception:
                    continue
                if sorted(prices) != [0.0, 1.0]:
                    continue  # unresolved / 50-50
                if (ct // 86400, frozenset(norm(o) for o in outcomes)) not in kidx:
                    continue  # only matches we also have on Kalshi
                h = get(CLOB + "/prices-history",
                        {"market": toks[0], "startTs": gs - 3 * 3600,
                         "endTs": min(ct, gs + 12 * 3600), "fidelity": 1}) or {}
                hist = [[x["t"], x["p"]] for x in h.get("history", [])]
                out.write(json.dumps({
                    "id": e["id"], "title": e["title"], "outcomes": outcomes,
                    "winner": int(prices.index(1.0)), "start": gs, "closed": ct,
                    "volume": float(m.get("volume") or 0),
                    "fee": [m.get("makerBaseFee"), m.get("takerBaseFee"), m.get("feeType")],
                    "hist": hist}) + "\n")
                n_new += 1
                if n_new % 100 == 0:
                    out.flush()
                    print(f"{n_new} saved, at {st}", flush=True)
            cursor = d.get("next_cursor")
            if stop or not cursor:
                break
    print(f"done: scanned {n_ev} events, saved {n_new}", flush=True)


if __name__ == "__main__":
    main()
