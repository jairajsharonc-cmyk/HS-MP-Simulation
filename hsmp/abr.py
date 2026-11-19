"""Adaptive-bitrate controllers (Sec. 3.1 and Sec. 4.1)."""
from __future__ import annotations
import numpy as np


class ThroughputABR:
    """Conventional rule: highest rung whose bitrate <= 0.85 x estimated throughput
    (Sec. 3.1 / Alg. 1 step 2e). Falls back to the lowest rung."""
    def __init__(self, cfg, rng=None):
        self.ladder = np.asarray(cfg.ladder_bps)
        self.safety = cfg.heuristic_safety

    def select(self, buffer_s, thr_bps):
        ok = np.where(self.ladder <= self.safety * thr_bps)[0]
        return int(ok[-1]) if len(ok) else 0

    def update(self, *a, **k):          # stateless
        pass


class QLearningABR:
    """Tabular Q-learning agent (Sec. 4.1).

    State  s = (buffer bin, throughput bin)   10 x 10, buffer bins of 2 s,
                                              throughput log-spaced bins
    Action a = rung index of the bitrate ladder (6 actions)
    Reward R = 1.0*Q_t - 3.0*S_t - 0.5*N_switch          (Q_t = bitrate / max bitrate)
    Update Q(s,a) <- Q(s,a) + alpha [R + gamma max_a' Q(s',a') - Q(s,a)]
    Policy epsilon-greedy (eps = 0.1). Q initialised to 0; ties broken at random.
    """
    def __init__(self, cfg, rng):
        self.cfg, self.rng = cfg, rng
        self.nA = len(cfg.ladder_mbps)
        self.Q = np.zeros((cfg.q_buffer_bins, cfg.q_thr_bins, self.nA))
        self.mbps = np.asarray(cfg.ladder_mbps)
        self.max_mbps = self.mbps.max()
        self._last = None               # (state, action)

    def state(self, buffer_s, thr_bps):
        c = self.cfg
        b = min(int(buffer_s // c.q_buffer_bin_s), c.q_buffer_bins - 1)
        lo, hi = c.q_thr_log10_range
        x = (np.log10(max(thr_bps, 1.0)) - lo) / (hi - lo)
        t = min(max(int(x * c.q_thr_bins), 0), c.q_thr_bins - 1)
        return b, t

    def select(self, buffer_s, thr_bps):
        s = self.state(buffer_s, thr_bps)
        if self.rng.random() < self.cfg.q_epsilon:
            a = int(self.rng.integers(self.nA))
        else:
            q = self.Q[s]
            best = np.flatnonzero(q == q.max())
            a = int(self.rng.choice(best))
        self._last = (s, a)
        return a

    def reward(self, served_idx, stall_s, switched):
        c = self.cfg
        q_t = self.mbps[served_idx] / self.max_mbps
        return c.w_quality * q_t - c.w_stall * stall_s - c.w_switch * float(switched)

    def update(self, reward, next_buffer_s, next_thr_bps):
        s, a = self._last
        s2 = self.state(next_buffer_s, next_thr_bps)
        td = reward + self.cfg.q_gamma * self.Q[s2].max() - self.Q[s][a]
        self.Q[s][a] += self.cfg.q_alpha * td
