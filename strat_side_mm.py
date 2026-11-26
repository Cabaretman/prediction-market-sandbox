"""Market making in the thin SET-1 winner market: maker P&L after fee,
marked at the set market's own mid +1m/+5m and at set-1 settlement."""
import json
from datetime import datetime
import numpy as np, pandas as pd
from strat_mm_state import wfmt
MAKER = 0.0175

def main(series="KXATPSETWINNER"):
    side = {}
    for l in open(f"data/kalshi/side_{series}.jsonl"):
        r = json.loads(l)
        if len(r["markets"]) == 2: side[r["event"]] = r["markets"][0]
    rows = []
    for l in open(f"data/kalshi/trades_side_{series}.jsonl"):
        r = json.loads(l); mk = side.get(r["event"])
        if not mk or not mk["candles"]: continue
        c = np.array([[x[0], np.nan if x[1] is None else x[1], np.nan if x[2] is None else x[2]] for x in mk["candles"]], float)
        t0 = c[0, 0]; n = int((c[-1, 0] - t0) // 60) + 1
        bid = np.full(n, np.nan); ask = np.full(n, np.nan)
        ix = ((c[:, 0] - t0) // 60).astype(int); bid[ix] = c[:, 1]; ask[ix] = c[:, 2]
        bid = pd.Series(bid).ffill().to_numpy(); ask = pd.Series(ask).ffill().to_numpy()
        mid = (bid + ask) / 2
        for created, p, cnt, taker in r["trades"]:
            t = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
            i = int((t - t0) // 60)
            if i < 1 or i >= n or not (0.05 < mid[i - 1] < 0.95): continue
            sgn = 1 if taker == "no" else -1
            f = MAKER * p * (1 - p)
            rows.append(dict(ev=r["event"], c=cnt, spread=ask[i - 1] - bid[i - 1],
                             half=sgn * (mid[i - 1] - p),
                             h1=sgn * (mid[min(i + 1, n - 1)] - p) - f,
                             h5=sgn * (mid[min(i + 5, n - 1)] - p) - f,
                             settle=sgn * (r["y"] - p) - f))
    d = pd.DataFrame(rows)
    print(f"{d.ev.nunique()} set-1 markets, {len(d):,} trades, {d.c.sum()/1e6:.1f}M contracts\n")
    def block(g, name):
        print(f"--- {name}")
        wfmt("spread captured (no fee)", g, "half"); wfmt("maker +1m", g, "h1")
        wfmt("maker +5m", g, "h5"); wfmt("maker to settlement", g, "settle")
    block(d, "ALL")
    d["sz"] = pd.cut(d.c, [0, 10, 100, 1000, 1e12], labels=["<10", "10-100", "100-1000", ">1000"])
    for s, g in d.groupby("sz", observed=True): block(g, f"taker size {s}")
    d["sp"] = pd.cut(d.spread, [-1, .0101, .0301, 1], labels=["1c", "2-3c", "4c+"])
    for s, g in d.groupby("sp", observed=True): block(g, f"quoted spread {s}")

if __name__ == "__main__":
    main()
