"""HS-MP simulation workflow - implements Algorithm 1 of the manuscript."""
from __future__ import annotations
from dataclasses import replace
import numpy as np

from .config import Config, CONFIGS
from .network import Overlay
from .workload import Workload
from .abr import ThroughputABR, QLearningABR
from .lstm import MLPopularCaching, FrequencyPredictor

METRICS = ["latency_ms", "energy_j", "available", "stall_s", "switch", "cache_hit"]
EXTRA = ["bitrate_mbps", "hops"]


def _streams(seed):
    """Independent RNG streams. churn/workload/seeding/traffic are identical for a given
    seed across the 4 strategies (common random numbers -> valid paired t-tests)."""
    ss = np.random.SeedSequence(seed).spawn(7)
    return [np.random.default_rng(s) for s in ss]


def run_simulation(cfg: Config, seed: int, strategy: str = "HH"):
    """Run one simulation. strategy in {"HH","HQ","MH","MQ"}.

    Returns dict(series={metric: ndarray[steps]}, summary={metric: float}).
    """
    spec = CONFIGS[strategy]
    cfg = replace(cfg, caching=spec["caching"], abr=spec["abr"])
    r_churn, r_work, r_seed, r_traffic, r_net, r_abr, r_ml = _streams(seed)

    # ---- 1. Initialisation phase -------------------------------------------------
    ov = Overlay(cfg, r_churn, r_traffic, r_net, client_pid=0)          # 1a, 1b
    ov.seed(r_seed)                                                      # 1c
    work = Workload(cfg, r_work)
    abr = QLearningABR(cfg, r_abr) if cfg.abr == "q" else ThroughputABR(cfg)   # 1d
    if cfg.caching == "ml":
        predictor = MLPopularCaching(cfg, r_ml)
    elif cfg.heuristic_prefetch == "frequency":
        predictor = FrequencyPredictor(cfg)
    else:
        predictor = None                                                 # pure reactive FIFO
    thr = cfg.throughput_init_bps                                        # 1e
    buf = cfg.buffer_initial_s
    ladder_bps = np.asarray(cfg.ladder_bps)
    top = len(ladder_bps) - 1
    last_q = None

    T = cfg.total_steps
    S = {k: np.zeros(T) for k in METRICS + EXTRA}

    # ---- 2. Main simulation loop ---------------------------------------------------
    for t in range(1, T + 1):
        i = t - 1
        ov.churn_step()                                                  # 2a
        ov.update_traffic(t)                                             # 2b
        v, c = work.next()                                               # 2c
        chunk = (v, c)
        cid = MLPopularCaching.encode(v, c, cfg.chunks_per_video)
        if predictor is not None:                                        # 2d
            predictor.record(cid)
            if cfg.caching == "ml" and t % cfg.train_every == 0:
                predictor.train()

        req_q = abr.select(buf, thr)                                     # 2e
        loc = ov.locate(chunk, req_q)                                    # 2f

        if loc is None:                                                  # A_t = 0
            avail, lat, en, hit, hops, served_q = 0.0, np.nan, 0.0, 0.0, np.nan, None
            stall = cfg.chunk_duration_s                                 # nothing to play
            buf = cfg.buffer_initial_s
            switched = False
            bitrate = 0.0
        else:
            holder, served_q, hops, path = loc
            avail = 1.0
            size = ladder_bps[served_q] * cfg.chunk_duration_s / 8.0     # bytes
            lat = ov.latency_ms(hops, path, size)                        # 2g
            en = ov.energy_j(hops, size)
            hit = float(hops <= 1)                                       # 0-1 hops = cache hit
            measured = min(size * 8.0 / (lat * 1e-3), cfg.link_bw_bps)
            thr = cfg.throughput_ema * measured + (1 - cfg.throughput_ema) * thr
            lat_s = lat * 1e-3                                           # 2h
            if lat_s > buf:
                stall = lat_s - buf
                buf = cfg.buffer_initial_s
            else:
                stall = 0.0
                buf = min(buf + cfg.chunk_duration_s - lat_s, cfg.buffer_max_s)
            switched = (last_q is not None) and (served_q != last_q)     # 2i
            last_q = served_q
            ov.insert(ov.client_pid, chunk, served_q)                    # client caches (FIFO)
            bitrate = cfg.ladder_mbps[served_q]

        if cfg.abr == "q":                                               # 2j
            rew = abr.reward(served_q if served_q is not None else 0, stall, switched)
            abr.update(rew, buf, thr)

        if predictor is not None and t % cfg.prefetch_every == 0:        # 2k
            targets = ov.pick_targets(cfg.n_prefetch_targets)
            for pc in predictor.predict_top_k(cfg.top_k):
                pv, pcidx = MLPopularCaching.decode(pc, cfg.chunks_per_video)
                placed = 0
                for pid in targets:
                    if (pv, pcidx) not in ov.peers[pid].cache:
                        ov.insert(pid, (pv, pcidx), top)
                        placed += 1
                        if placed >= cfg.replicas_per_chunk:
                            break

        S["latency_ms"][i] = lat; S["energy_j"][i] = en; S["available"][i] = avail   # 2l
        S["stall_s"][i] = stall; S["switch"][i] = float(switched); S["cache_hit"][i] = hit
        S["bitrate_mbps"][i] = bitrate; S["hops"][i] = hops

    summary = {k: float(np.nanmean(v)) for k, v in S.items()}
    summary["stall_ratio"] = float(S["stall_s"].sum() / (T * cfg.chunk_duration_s))
    return {"series": S, "summary": summary}
