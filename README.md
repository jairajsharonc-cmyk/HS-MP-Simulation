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

# 3. FULL experiment as in the paper: 4 strategies x 10 seeds x 250 steps (~30-60 s on a laptop CPU)
python run_experiments.py --out results
python make_report.py --in results

# 4. Optional analyses
python sensitivity_q.py            # effect of the unreported Q-learning alpha / gamma
python diagnose_lstm.py            # why the LSTM adds nothing (prediction accuracy + prefetch ablation)
python run_experiments.py --heuristic-prefetch none --out results_pureFIFO   # strictly reactive FIFO (Sec 3.1)
python run_experiments.py --lstm-backend torch --out results_torch           # needs: pip install torch
```

### What you get in `results/`
| File | Corresponds to |
|---|---|
| `per_seed_summary.csv` | one row per (strategy, seed): all metrics averaged over the 250 steps |
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
## 3. What the manuscript does NOT specify (my assumptions — report these!)

All are tagged `[ASSUMED]`, `[INFERRED]` or `[CALIBRATED]` in `hsmp/config.py`.

| Item | Value used | Basis |
|---|---|---|
| Bitrate ladder | 0.4 / 1.0 / 2.8 / 4.0 / 7.2 / 12 Mbps, 2 s chunks | **Inferred from the energy plateaus of your Fig. 3** (E ≈ bytes × 8 nJ). Gives H–H energy = 0.0239 J exactly as Table 2 |
| Workload | 30 videos, Zipf 0.8, P(next chunk) = 0.985 | **Calibrated** so the H–H baseline matches Table 2 (latency ≈ 11.4 ms, hit ≈ 0.11–0.14). Not tuned on ML-vs-heuristic contrasts |
| Link capacity | 10 Gbps | needed for ~11 ms latency with ~3 MB chunks (see review note 8) |
| Q-learning α, γ | 0.1, 0.9 | unreported; results are insensitive (`sensitivity_q.py`) |
| LSTM size / batch / epochs per training call | 32-d embedding, 64 hidden, 16, 1 | unreported |
| Heuristic "prediction" in Alg. 1 step 2k | frequency of recent requests (`--heuristic-prefetch none` = pure FIFO) | Alg. 1 and Sec 3.1 disagree; ablation shows it makes no difference |
| Local-cache energy | Eq. (1) applied uniformly | matches Table 2; prose says "very little" (`--local-energy-discount`) |
| Cache hit | served from the client itself or a 1-hop neighbour | Alg. 1 step 2f "(0–1 hops)" |
| Re-replication on peer departure | on (availability = 1) | paper reports availability 1.0000 |
| Proactive caching | every 5 steps, 3 nearest low-load peers, ≤ 2 replicas/chunk | "periodic", unspecified |

---------------------------------------------------------------------------------------------------
## 4. Results of this implementation vs. your Table 2 (10 seeds, mean)

| | Latency ms (mine / paper) | Energy J | Switch rate | Cache hit |
|---|---|---|---|---|
| H–H | 11.33 / 11.37 | 0.0239 / 0.0239 | 0.004 / 0.021 | 0.143 / 0.113 |
| H–Q | 10.60 / 11.89 | 0.0158 / 0.0111 | 0.234 / 0.494 | 0.141 / 0.093 |
| M–H | 11.33 / 11.97 | 0.0239 / 0.0238 | 0.004 / 0.024 | 0.144 / 0.105 |
| M–Q | 10.60 / 12.28 | 0.0158 / 0.0119 | 0.236 / 0.470 | 0.139 / 0.087 |

**Reproduced (direction & significance):** Q-learning significantly cuts energy (t ≈ 7.6, p < 0.001) and
significantly raises switching (t ≈ −17); availability ≈ 1, stalls ≈ 0; H–H baseline latency/energy match.

**Not reproduced:** (i) magnitude of energy saving (−34 % vs −50 %) and switch rate (0.23 vs 0.49), not
explained by α/γ; (ii) **latency increase under Q-learning** — here latency *falls* by 0.7 ms because smaller
chunks transmit faster; (iii) **ML caching being worse than FIFO** — here no measurable difference.

**Two findings you should understand before submitting** (`diagnose_lstm.py`, `sensitivity_q.py`):
1. *The LSTM learns nothing in this setting*: online top-1 accuracy 0.3 % (chance 0.02 %), while the rule
   "next chunk = current + 1" is right 98.5 % of the time. Prefetch ablation: none 0.1436, frequency 0.1432,
   LSTM 0.1444 cache hit. A chunk-ID-embedding LSTM cannot generalise across chunk IDs in a 250-step online run.
2. *The Q-learning "energy saving" is mostly a quality sacrifice*: energy ∝ delivered bytes. Mean delivered
   bitrate drops from 11.95 to ~7.9 Mbps. Without a QoE/bitrate metric, "saves 50 % energy" is not a like-for-like win.

> **Important:** this is an independent re-implementation from the text. If your submitted Table 2/3 came from
> your original code, keep reporting those numbers from that code; use this package as the released, documented
> reference implementation, and state any differences openly.

---------------------------------------------------------------------------------------------------
## 5. Review notes on the manuscript (things a referee may catch)

1. **Fig. 6 image ≠ Table 2.** The image shows latency 0.49–1.39 ms, cache hit 74–82 %, switch rate 0.27/0.74 —
   different from Table 2 — and it is a bar chart of means while the caption says "number of significant differences".
2. Sec 5.1 says "four key metrics" but defines six; stall *ratio* is defined but Table 2 reports stall *seconds*.
3. Energy: Sec 5.1 defines power × time (tx/rx/idle) while the simulator/Eq. (1) is per-byte + per-hop. Pick one.
4. Alg. 1 uses 250 steps; Sec 5.2 says 300.
5. Limitations says churn is not considered, but churn is modelled (Sec 3.3.2); "homogeneous" testbed vs "heterogeneous peer behavior".
6. Missing hyper-parameters: α, γ, LSTM embedding/hidden size, batch, epochs, catalogue size, workload model, link capacity, ladder bitrates.
7. "p = 0" → write p < 0.001. 24 uncorrected tests: after Holm correction `HH vs MH` cache-hit and several marginal
   latency results lose significance (see `p_holm` in Table 3). Consider also reporting effect sizes with CIs.
8. No explicit latency equation (only words); multi-Gbps links are needed to reach ~11 ms — state this.
9. References: [10], [11] appear uncited in the text; [12]/[13] are cited for two different ideas; [30] (key-value store
   caching) and [32] (VVC decoding energy) look mismatched with the claims they support; several entries still contain
   "[DOI]" / "[add remaining authors]" and the Data-availability statement has "[repository name]". (Verify against the sources.)
