"""Sanity tests. Run:  python test_hsmp.py     (or: pytest -q)"""
from dataclasses import replace
import numpy as np
from hsmp import Config, run_simulation
from hsmp.lstm import NumpyLSTM
from hsmp.network import Overlay, hamming
from hsmp.abr import ThroughputABR, QLearningABR
from hsmp.workload import Workload
from hsmp.simulator import _streams


def test_lstm_gradients():
    rng = np.random.default_rng(0)
    m = NumpyLSTM(30, 5, 6, 1e-3, rng)
    X = rng.integers(0, 30, (7, 4)); y = rng.integers(0, 30, 7)
    _, g = m.loss_and_grads(X, y)
    for k in m.p:
        flat, gf = m.p[k].reshape(-1), g[k].reshape(-1)
        for j in rng.choice(len(flat), min(8, len(flat)), replace=False):
            old = flat[j]; eps = 1e-5
            flat[j] = old + eps; lp, _ = m.loss_and_grads(X, y)
            flat[j] = old - eps; lm, _ = m.loss_and_grads(X, y)
            flat[j] = old
            num = (lp - lm) / (2 * eps)
            assert abs(num - gf[j]) < 1e-3 * max(1.0, abs(num) + abs(gf[j])), k


def test_energy_equation_1():
    cfg = Config(); ov = Overlay(cfg, *[np.random.default_rng(i) for i in range(3)])
    size = 3_000_000
    assert abs(ov.energy_j(4, size) - (size * (5e-9 + 3e-9) + 4 * 2e-6)) < 1e-12


def test_heuristic_abr_rule():
    cfg = Config(); abr = ThroughputABR(cfg)
    assert abr.select(2.0, 600_000) == 0                      # 0.85*0.6 Mbps -> 144p only
    assert abr.select(10.0, 20e6) == 5                        # plenty of throughput -> 1080p
    assert abr.select(10.0, 100.0) == 0                       # nothing fits -> lowest rung


def test_q_state_space_and_update():
    cfg = Config(); q = QLearningABR(cfg, np.random.default_rng(0))
    assert q.Q.shape == (10, 10, 6)
    assert q.state(0.0, 1e5) == (0, 0) and q.state(25.0, 1e11) == (9, 9)
    q.select(4.0, 1e8); q.update(1.0, 6.0, 1e8)
    assert q.Q.sum() > 0


def test_churn_bounds_and_cache_capacity():
    cfg = Config(p_join=0.5, p_leave=0.5)
    rc, rt, rn = [np.random.default_rng(i) for i in range(3)]
    ov = Overlay(cfg, rc, rt, rn); ov.seed(np.random.default_rng(9))
    for _ in range(600):
        ov.churn_step()
        assert cfg.min_peers <= len(ov.peers) <= cfg.max_peers
    assert all(len(p.cache) <= cfg.cache_capacity for p in ov.peers.values())
    assert 0 in ov.peers                                       # the observed client never leaves


def test_hamming_routing_hops():
    cfg = Config(); ov = Overlay(cfg, *[np.random.default_rng(i) for i in range(3)])
    hops, path = ov._route(0, 0b10110)
    assert hops == 3 == len(path) and hamming(0, 0b10110) == 3
    assert path[-1] == 0b10110


def test_common_random_numbers_across_strategies():
    cfg = Config()
    seqs = []
    for _ in range(2):
        _, r_work, *_ = _streams(3); w = Workload(cfg, r_work)
        seqs.append([w.next() for _ in range(50)])
    assert seqs[0] == seqs[1]


def test_reproducible_and_sane():
    cfg = Config(total_steps=60)
    a = run_simulation(cfg, 1, "MQ")["summary"]; b = run_simulation(cfg, 1, "MQ")["summary"]
    assert a == b
    assert 0 <= a["cache_hit"] <= 1 and 0 <= a["switch"] <= 1 and a["available"] > 0.9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
