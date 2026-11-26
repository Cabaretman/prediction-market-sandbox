"""Pre-match mispricing map for tennis SIDE markets (exact score, set winner).
Each contract at T-10 min before the match's detected start; rows for buying
YES (at ask) and buying NO (at 1-bid). Fee 0.07*p(1-p). Early/late halves."""
import glob, json
import numpy as np, pandas as pd
import strat_side as ss, kalshi_analyze as ka
TK=0.07
BINS=[0,.03,.05,.10,.20,.40,.60,.80,.90,.95,.97,1.0]
anchor=ss.load_anchor()
rows=[]
for fn in glob.glob("data/kalshi/side_K*.jsonl"):
    kind="exact" if "EXACT" in fn else "set1"
    for l in open(fn):
        r=json.loads(l)
        code=r["event"].split("-",1)[1]
        if kind=="set1": code=code.rsplit("-",1)[0]
        m=anchor.get(code)
        if m is None or m["start"] is None: continue
        i=m["start"]-10
        if i<0: continue
        tg=np.array([m["t0"]+60*i])
        for x in r["markets"]:
            b,a=ss.series_at(x["candles"],tg); b,a=b[0],a[0]
            if np.isnan(b) or np.isnan(a) or a-b>0.05 or a<=0 or b>=1: continue
            y=1.0 if x["result"]=="yes" else 0.0
            mid=(a+b)/2
            for side,p,px,yy in [("yes",mid,a,y),("no",1-mid,1-b,1-y)]:
                rows.append((kind,r["event"],m["t0"],side,p,px,yy,yy-px-TK*px*(1-px)))
d=pd.DataFrame(rows,columns=["kind","ev","t","side","p","px","y","pnl"])
d["bin"]=pd.cut(d.p,BINS)
med=d.t.median()
def ts(x): return x.mean()/(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>2 and x.std()>0 else 0
for kind,g0 in d.groupby("kind"):
    print(f"===== {kind}: {g0.ev.nunique()} events =====")
    print(f"{'bin':14s} {'n':>6s} {'win':>6s} {'price':>6s} | {'P&L':>8s} {'t':>6s} | early / late")
    for b,g in g0.groupby("bin",observed=True):
        e,l=g[g.t<med],g[g.t>=med]
        print(f"{str(b):14s} {len(g):6d} {g.y.mean():6.3f} {g.p.mean():6.3f} | {100*g.pnl.mean():+7.2f}c {ts(g.pnl):+6.2f} | "
              f"{100*e.pnl.mean():+6.2f}c ({len(e)}) / {100*l.pnl.mean():+6.2f}c ({len(l)})")
    print()
