#!/usr/bin/env python3
"""Run the full HS-MP experiment: 4 strategies x N seeds (Alg. 1 step 4).

Usage:
    python run_experiments.py                    # manuscript setting: 10 seeds, 300 steps
    python run_experiments.py --seeds 3 --steps 100 --out results_quick     # smoke test
    python run_experiments.py --lstm-backend torch                          # use PyTorch LSTM
    python run_experiments.py --heuristic-prefetch none                     # purely reactive FIFO
"""
import argparse, os, time, json
from dataclasses import replace, asdict
import numpy as np
import pandas as pd
from hsmp import Config, CONFIGS, run_simulation
from hsmp.simulator import METRICS, EXTRA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default="results")
    ap.add_argument("--lstm-backend", default="numpy", choices=["numpy", "torch"])
    ap.add_argument("--heuristic-prefetch", default="none", choices=["frequency", "none"])
    ap.add_argument("--no-local-energy-discount", action="store_true")
    ap.add_argument("--deterministic-lstm", action="store_true")
    a = ap.parse_args()

    cfg = Config(total_steps=a.steps, n_seeds=a.seeds, lstm_backend=a.lstm_backend,
                 heuristic_prefetch=a.heuristic_prefetch,
                 local_energy_discount=not a.no_local_energy_discount,
                 sample_predictions=not a.deterministic_lstm)
    os.makedirs(a.out, exist_ok=True)
    seeds = list(range(a.first_seed, a.first_seed + a.seeds))

    rows, series = [], {}
    t0 = time.time()
    for strat in CONFIGS:
        arr = np.zeros((len(seeds), a.steps, len(METRICS) + len(EXTRA)))
        for j, seed in enumerate(seeds):
            r = run_simulation(cfg, seed, strat)
            rows.append(dict(strategy=strat, seed=seed, **r["summary"]))
            arr[j] = np.stack([r["series"][k] for k in METRICS + EXTRA], axis=1)
            print(f"[{time.time()-t0:6.1f}s] {strat} seed={seed:2d} "
                  f"lat={r['summary']['latency_ms']:.2f}ms E={r['summary']['energy_j']:.4f}J "
                  f"hit={r['summary']['cache_hit']:.3f} sw={r['summary']['switch']:.3f}")
        series[strat] = arr

    pd.DataFrame(rows).to_csv(os.path.join(a.out, "per_seed_summary.csv"), index=False)
    np.savez_compressed(os.path.join(a.out, "timeseries.npz"),
                        columns=np.array(METRICS + EXTRA), seeds=np.array(seeds), **series)
    with open(os.path.join(a.out, "config_used.json"), "w") as f:
        json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()}, f, indent=2)
    print(f"\nSaved raw results to {a.out}/   (next: python make_report.py --in {a.out})")


if __name__ == "__main__":
    main()
