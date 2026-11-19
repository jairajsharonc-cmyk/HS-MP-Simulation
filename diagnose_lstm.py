#!/usr/bin/env python3
"""Diagnostics for the LSTM caching result:
  (1) online next-request prediction accuracy of the LSTM vs. trivial predictors
  (2) prefetch ablation: none vs frequency vs LSTM  (cache-hit rate, H-ABR fixed)

Usage: python diagnose_lstm.py [--seeds 10]
"""
import argparse
from dataclasses import replace
import numpy as np
from hsmp import Config, run_simulation
from hsmp.workload import Workload
from hsmp.lstm import MLPopularCaching, FrequencyPredictor
from hsmp.simulator import _streams

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10); a = ap.parse_args()
cfg = Config()

acc = {"LSTM top-1": [], "LSTM top-5": [], "Frequency top-5": [], "Oracle 'next chunk' top-1": []}
for seed in range(a.seeds):
    r_churn, r_work, r_seed, r_traffic, r_net, r_abr, r_ml = _streams(seed)
    wl = Workload(cfg, r_work); lstm = MLPopularCaching(cfg, r_ml); freq = FrequencyPredictor(cfg)
    h1 = h5 = hf = ho = n = 0; prev = None; pred5 = []; predf = []; predo = None
    for t in range(1, cfg.total_steps + 1):
        v, c = wl.next(); cid = MLPopularCaching.encode(v, c, cfg.chunks_per_video)
        if prev is not None and lstm.trained:                      # score the previous prediction
            n += 1; h1 += (len(pred5) > 0 and pred5[0] == cid); h5 += cid in pred5
            hf += cid in predf; ho += (predo == cid)
        lstm.record(cid); freq.record(cid)
        if t % cfg.train_every == 0: lstm.train()
        pred5 = lstm.predict_top_k(5); predf = freq.predict_top_k(5); predo = cid + 1; prev = cid
    for k, x in zip(acc, [h1, h5, hf, ho]): acc[k].append(x / max(n, 1))
print("Online next-request prediction accuracy (mean over seeds):")
for k, v in acc.items(): print(f"  {k:28s} {np.mean(v):.3f}")

print("\nPrefetch ablation (H-ABR fixed, mean over seeds):")
for name, cfg2, strat in [("no prefetch (pure FIFO)", replace(cfg, heuristic_prefetch="none"), "HH"),
                          ("frequency prefetch      ", cfg, "HH"),
                          ("LSTM prefetch           ", cfg, "MH")]:
    R = [run_simulation(cfg2, s, strat)["summary"] for s in range(a.seeds)]
    print(f"  {name}: cache_hit={np.mean([r['cache_hit'] for r in R]):.4f}  latency={np.mean([r['latency_ms'] for r in R]):.2f} ms")
