"""Choose market-making settings on the EARLY half of events (harsh fills),
then evaluate the chosen setting on the LATE half (out of sample), under all
fill rules and a slower 3s latency.
Usage: python3 sim_grid.py KXATPSETWINNER [KXWTASETWINNER]
"""
import itertools
import sys
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
import pandas as pd

import mm_sim as s

GRID = dict(size=[20, 50], cap=[50, 150], band=[0.10, 0.20],
            flatten=[False, True], exit_band=[0.0, 0.10], min_spread=[0.02, 0.03])


def make(cfg):
    size, cap, band, flat, exit_band, ms = cfg
    return s.Params(size=size, cap=cap, band=band, exit_band=exit_band, min_spread=ms,
                    flatten_at=int(cap * 0.6) if flat else 10**9,
                    flatten_to=int(cap * 0.3) if flat else 0, fill_rule="harsh")


EVS = None


def _run(P):
    r = s.run(EVS, P)
    return r.pnl.sum(), r.contracts.sum()


def main(series_list):
    global EVS
    for series in series_list:
        evs = s.load_events(series)
        n_anchor = sum(1 for e in evs if e["anchor"])
        half = len(evs) // 2
        train, test = evs[:half], evs[half:]
        print(f"===== {series}: {len(evs)} set-1 markets ({n_anchor} with anchor tape); "
              f"train {len(train)} / test {len(test)} =====")
        cfgs = list(itertools.product(*GRID.values()))
        params = [make(c) for c in cfgs]
        EVS = train
        with Pool(4) as pool:
            res = pool.map(_run, params)
        tab = pd.DataFrame([dict(cfg=c, pnl=p, contracts=k) for c, (p, k) in zip(cfgs, res)])
        tab["c_per"] = tab.pnl / tab.contracts.clip(lower=1) * 100
        tab = tab.sort_values("pnl", ascending=False)
        print("Top 5 settings on TRAIN (size, cap, band, flatten, exit_band, min_spread):")
        for r in tab.head(5).itertuples():
            print(f"   {r.cfg}  P&L ${r.pnl:8.0f}  {r.c_per:+.2f}c/contract")
        print(f"   ...worst: {tab.iloc[-1].cfg} ${tab.iloc[-1].pnl:.0f}; "
              f"share of settings profitable on train: {(tab.pnl > 0).mean():.0%}")
        best = make(tab.iloc[0].cfg)
        print(f"\nChosen: {best}")
        print("OUT-OF-SAMPLE (test half):")
        for rule in ["optimistic", "harsh", "very_harsh"]:
            s.summarize(s.run(test, replace(best, fill_rule=rule)), f"{rule}, 1s latency")
        s.summarize(s.run(test, replace(best, latency=3.0)), "harsh, 3s latency")
        s.summarize(s.run(test, replace(best, pull_move=None)), "harsh, no anchor pull")
        print("For reference, TRAIN with chosen setting:")
        s.summarize(s.run(train, best), "harsh, 1s latency (train)")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
