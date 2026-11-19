# HS-MP testbed — implementation of the manuscript

*Machine Learning versus Heuristics in Peer-to-Peer Video Streaming: A Statistical Evaluation of
Predictive Caching and Adaptive Bitrate Control*

Clean-room implementation of the HS-MP simulator (Alg. 1), the four strategies (H–H, H–Q, M–H, M–Q),
the multi-seed protocol and the paired t-tests. Pure Python; the LSTM is hand-written NumPy
(gradient-checked), with an optional PyTorch backend.

---------------------------------------------------------------------------------------------------
## 1. Step-by-step: how to run it

```bash
# 0. Python >= 3.9 is required
unzip hsmp_project.zip && cd hsmp_project
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Verify the installation (8 tests, ~10 s)
python test_hsmp.py

# 2. Quick smoke run (3 seeds, 100 steps, ~5 s)
python run_experiments.py --seeds 3 --steps 100 --out results_quick
python make_report.py --in results_quick

# 3. FULL experiment as in the paper: 4 strategies x 10 seeds x 300 steps (~30-60 s on a laptop CPU)
python run_experiments.py --out results
python make_report.py --in results

# 4. Optional analyses
python sensitivity_q.py            # effect of the unreported Q-learning alpha / gamma
python diagnose_lstm.py            # why the LSTM adds nothing (prediction accuracy + prefetch ablation)
python run_experiments.py --heuristic-prefetch frequency --out results_freq   # optional ablation: FIFO + frequency prefetch
python run_experiments.py --lstm-backend torch --out results_torch           # needs: pip install torch
```

### What you get in `results/`
| File | Corresponds to |
|---|---|
| `per_seed_summary.csv` | one row per (strategy, seed): all metrics averaged over the 300 steps |
| `table2_performance.csv` | **Table 2** (mean ± 95 % CI half-width, t-distribution, n = 10) |
| `table3_paired_ttests.csv` | **Table 3** (t, p, mean_diff, Cohen's d_z, significance) + supplementary Holm-adjusted p |
| `fig2…fig5_timeseries_{HH,HQ,MH,MQ}.png` | **Figs 2–5** (seed 0 trace; change with `--ts-seed`) |
| `fig6_significance_counts.png` | **Fig 6** (as its caption describes: # significant tests per metric) |
| `fig_means_bar.png` | bar chart of mean metrics (what the Fig. 6 image in the manuscript actually shows) |
| `table_supplementary.csv` | mean delivered bitrate and stall ratio (not in the manuscript) |
| `timeseries.npz`, `config_used.json` | raw per-step data and the exact parameters (cite these for reproducibility) |

Everything is deterministic: same seeds -> same numbers.

---------------------------------------------------------------------------------------------------
## 2. Map: manuscript -> code

| Manuscript | File / function |
|---|---|
| Alg. 1 (workflow) | `hsmp/simulator.py::run_simulation` (steps labelled 1a … 2l in comments) |
| Sec 3.3.1–3.3.2 peers, churn (p_join .10, p_leave .08, 12–256 peers) | `hsmp/network.py::Overlay` |
| Structured overlay, multi-path routing, Hamming distance | `Overlay._route`, `hamming` (d = 8 hypercube) |
| Sec 3.3.4 latency, jitter, traffic patterns | `Overlay.latency_ms`, `update_traffic` |
| Eq. (1) energy | `Overlay.energy_j` |
| FIFO cache (Sec 3.1) | `Overlay.insert` |
| Throughput ABR (0.85 rule) | `hsmp/abr.py::ThroughputABR` |
| Q-learning ABR (state, reward, update, ε = 0.1) | `hsmp/abr.py::QLearningABR` |
| LSTM predictor (embedding→LSTM→FC, 50-history, seq 4, lr 1e-3, top-5) | `hsmp/lstm.py` |
| Multi-seed + 95 % CI + paired t-tests | `run_experiments.py`, `make_report.py` |

Design choice worth citing: for a given seed, **churn, workload, seeding and traffic noise use
independent RNG streams that are identical across the four strategies** (common random numbers), so the
seed-wise pairing in the paired t-test is genuine.

---------------------------------------------------------------------------------------------------
