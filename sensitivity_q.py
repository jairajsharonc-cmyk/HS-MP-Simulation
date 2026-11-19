#!/usr/bin/env python3
"""Sensitivity of the H-Q configuration to the Q-learning hyper-parameters that the
manuscript does not report (alpha, gamma) and to epsilon. Not part of the paper's
pipeline; use it to justify / report the values you finally choose.

Usage: python sensitivity_q.py [--seeds 6] [--out results/sensitivity_q.csv]
"""
import argparse, itertools
from dataclasses import replace
import numpy as np, pandas as pd
from hsmp import Config, run_simulation

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=6)
ap.add_argument("--out", default="results/sensitivity_q.csv")
a = ap.parse_args()

base = Config()
hh = [run_simulation(base, s, "HH")["summary"] for s in range(a.seeds)]
ref = {k: np.mean([r[k] for r in hh]) for k in ["latency_ms", "energy_j", "switch", "cache_hit"]}
rows = []
for alpha, gamma, eps in itertools.product([0.05, 0.1, 0.3, 0.5], [0.0, 0.5, 0.9, 0.99], [0.1]):
    cfg = replace(base, q_alpha=alpha, q_gamma=gamma, q_epsilon=eps)
    R = [run_simulation(cfg, s, "HQ")["summary"] for s in range(a.seeds)]
    m = lambda k: np.mean([r[k] for r in R])
    rows.append(dict(alpha=alpha, gamma=gamma, epsilon=eps,
                     latency_ms=m("latency_ms"), d_latency_ms=m("latency_ms") - ref["latency_ms"],
                     energy_j=m("energy_j"), energy_saving_pct=100 * (1 - m("energy_j") / ref["energy_j"]),
                     switch=m("switch"), bitrate_mbps=m("bitrate_mbps")))
df = pd.DataFrame(rows)
df.to_csv(a.out, index=False)
print(f"H-H reference: latency={ref['latency_ms']:.2f} ms  energy={ref['energy_j']:.4f} J  switch={ref['switch']:.3f}")
print("Manuscript H-Q: latency +0.52 ms, energy saving ~53%, switch 0.49\n")
print(df.round(3).to_string(index=False))
