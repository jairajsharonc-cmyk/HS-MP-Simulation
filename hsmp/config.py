"""All simulation parameters in one place.

Every field is tagged:
  [PAPER]   value stated explicitly in the manuscript (section / table given)
  [INFERRED] value reverse-engineered from a figure/number in the manuscript
  [ASSUMED] the manuscript is silent; a sensible default was chosen.
            These are the knobs to report in a "Reproducibility" paragraph.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # ---------------- Overlay / peers (Sec. 3.3.1-3.3.2, Table 1, Alg. 1) --------
    dimension: int = 8                 # [PAPER] d=8 -> 2^8 = 256 PIDs (Alg.1 input)
    initial_peers: int = 100           # [PAPER] Table 1
    min_peers: int = 12                # [PAPER] Sec 3.3.2
    max_peers: int = 256               # [PAPER] Sec 3.3.2
    p_join: float = 0.10               # [PAPER] Sec 3.3.2
    p_leave: float = 0.08              # [PAPER] Sec 3.3.2
    cache_capacity: int = 180          # [PAPER] Table 1 (chunks per peer)
    replicas_per_chunk: int = 2        # [PAPER] Alg.1 step 1c
    repair_on_leave: bool = True       # [ASSUMED] re-replicate chunks that lose all
                                       #   replicas when a peer leaves (paper reports
                                       #   availability = 1.0000 for heuristics)
    n_paths: int = 3                   # [ASSUMED] candidate shortest paths examined
                                       #   per request (multi-path routing)

    # ---------------- Content model (Sec. 3.3.3) -------------------------------
    chunks_per_video: int = 180        # [PAPER]
    n_videos: int = 30                 # [ASSUMED/CALIBRATED] catalogue size (LSTM vocab = 5400)
    ladder_names: Tuple[str, ...] = ("144p", "240p", "360p", "480p", "720p", "1080p")  # [PAPER]
    # [INFERRED] from the energy plateaus in Fig. 3 (E ~ bytes * 8e-9 J, 2 s chunks):
    #   240p=1.0, 360p=2.8, 480p=4.0, 720p=7.2, 1080p=12 Mbps; 144p must be
    #   <= 0.85*600 kbps (first heuristic decision) -> 0.4 Mbps.
    ladder_mbps: Tuple[float, ...] = (0.4, 1.0, 2.8, 4.0, 7.2, 12.0)
    chunk_duration_s: float = 2.0      # [ASSUMED] consistent with 2 s buffer bins

    # ---------------- Workload (not specified: "synthetic request patterns") ----
    # CALIBRATION NOTE: n_videos / zipf / p_sequential were set so that the H-H *baseline* lands
    # in the regime of Table 2 (cache hit ~0.11, latency ~11-12 ms). No ML-vs-heuristic
    # contrast was used for tuning.
    zipf_exponent: float = 0.8         # [ASSUMED/CALIBRATED] video popularity ~ Zipf(s)
    p_sequential: float = 0.985        # [ASSUMED/CALIBRATED] P(next request = next chunk)
                                       #   else: switch to a new (Zipf) video, chunk 0

    # ---------------- Playback buffer (Sec. 3.3.1, Table 1) --------------------
    buffer_initial_s: float = 2.0      # [PAPER]
    buffer_max_s: float = 20.0         # [PAPER]
    throughput_init_bps: float = 600_000.0   # [PAPER] Alg.1 step 1e
    throughput_ema: float = 0.5        # [ASSUMED] weight of newest sample ("moving average")
    heuristic_safety: float = 0.85     # [PAPER] Alg.1 step 2e

    # ---------------- Network & energy models (Sec. 3.3.4, Table 1) ------------
    hop_delay_ms: float = 2.8          # [PAPER]
    traffic_factors: Tuple[float, ...] = (1.00, 1.15, 1.30)   # [PAPER] transpose, bit-reversal, bit-complement
    traffic_names: Tuple[str, ...] = ("transpose", "bit-reversal", "bit-complement")
    load_noise_std: float = 0.05       # [ASSUMED] clipped-Gaussian noise on load factor
    load_clip: Tuple[float, float] = (0.8, 2.0)   # [ASSUMED]
    jitter_ms: float = 1.0             # [PAPER] U(-1, +1) ms per hop
    link_bw_bps: float = 10e9          # [ASSUMED] per-link capacity. Must be ~Gbps for the
                                       #   paper's ~11 ms latency to hold with 3 MB chunks.
    local_proc_ms: float = 0.1         # [ASSUMED] local-cache read latency
    e_tx_per_byte: float = 5.0e-9      # [PAPER]
    e_rx_per_byte: float = 3.0e-9      # [PAPER]
    e_hop: float = 2.0e-6              # [PAPER]
    local_energy_discount: bool = True   # [PAPER] Sec 3.3.4 "Impact of Caching": a local-cache fetch costs
                                       #   "very little energy (only local processing)"; remote fetches
                                       #   follow Eq.(1).
    e_local_per_byte: float = 1.0e-10  # [ASSUMED] local processing cost (value not given in the paper)

    # ---------------- Simulation loop (Alg. 1) ---------------------------------
    total_steps: int = 300             # [PAPER] Sec 5.2 / time-series plots (Alg.1 lists 250 as an "e.g.")
    train_every: int = 10              # [PAPER] Alg.1
    prefetch_every: int = 5            # [ASSUMED] "periodic" proactive caching
    n_prefetch_targets: int = 3        # [ASSUMED]
    top_k: int = 5                     # [PAPER] Sec 4.2.4
    heuristic_prefetch: str = "none"   # [PAPER] Sec 3.1 / 3.3.3: FIFO is purely reactive (caches only after a
                                       #   request). "frequency" kept as an optional ablation.

    # ---------------- Q-learning ABR (Sec. 4.1) --------------------------------
    q_buffer_bins: int = 10            # [PAPER] 10 bins x 2 s
    q_buffer_bin_s: float = 2.0        # [PAPER]
    q_thr_bins: int = 10               # [PAPER] logarithmic
    q_thr_log10_range: Tuple[float, float] = (5.0, 10.0)   # [ASSUMED] 1e5 .. 1e10 bps
    q_alpha: float = 0.1               # [ASSUMED] learning rate (not given)
    q_gamma: float = 0.9               # [ASSUMED] discount (not given)
    q_epsilon: float = 0.1             # [PAPER]
    w_quality: float = 1.0             # [PAPER] reward weights
    w_stall: float = 3.0               # [PAPER]
    w_switch: float = 0.5              # [PAPER]

    # ---------------- LSTM caching (Sec. 4.2) ----------------------------------
    history_len: int = 50              # [PAPER]
    seq_len: int = 4                   # [PAPER]
    lstm_lr: float = 1e-3              # [PAPER]
    lstm_embed: int = 32               # [ASSUMED]
    lstm_hidden: int = 64              # [ASSUMED]
    lstm_batch: int = 16               # [ASSUMED]
    lstm_epochs_per_train: int = 1     # [ASSUMED]
    lstm_backend: str = "numpy"        # "numpy" (tested, no dependencies) or "torch"
    sample_predictions: bool = True    # [PAPER] Sec 4.3: randomness enters via "prediction probabilities";
                                       #   Sec 6 mentions the softmax temperature. False = deterministic top-k
    softmax_temperature: float = 1.0   # only used if sample_predictions=True

    # ---------------- Experiment design (Sec. 4.3, 5) ---------------------------
    n_seeds: int = 10                  # [PAPER]
    alpha_sig: float = 0.05            # [PAPER]

    # ---- strategy switches (set by CONFIGS below) ----
    caching: str = "heuristic"         # "heuristic" | "ml"
    abr: str = "heuristic"             # "heuristic" | "q"

    @property
    def n_chunks_total(self) -> int:
        return self.n_videos * self.chunks_per_video

    @property
    def ladder_bps(self):
        return tuple(m * 1e6 for m in self.ladder_mbps)


# The four strategies evaluated in the paper
CONFIGS = {
    "HH": dict(caching="heuristic", abr="heuristic", label="Heuristic Caching + Heuristic ABR"),
    "HQ": dict(caching="heuristic", abr="q",         label="Heuristic Caching + Q-learning ABR"),
    "MH": dict(caching="ml",        abr="heuristic", label="ML Caching + Heuristic ABR"),
    "MQ": dict(caching="ml",        abr="q",         label="ML Caching + Q-learning ABR (Full Model)"),
}
