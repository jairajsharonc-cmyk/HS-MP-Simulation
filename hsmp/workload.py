"""Synthetic request workload (the manuscript only says 'synthetic request patterns').

A single observed client watches videos chunk by chunk. Video popularity is Zipf;
with prob. p_sequential the next request is the next chunk of the same video,
otherwise the client starts a new (Zipf-drawn) video at chunk 0.
Sequential structure is exactly what an LSTM can in principle learn.
"""
import numpy as np


class Workload:
    def __init__(self, cfg, rng):
        self.cfg, self.rng = cfg, rng
        ranks = np.arange(1, cfg.n_videos + 1)
        w = ranks ** (-cfg.zipf_exponent)
        self.p_video = w / w.sum()
        self.video = int(rng.choice(cfg.n_videos, p=self.p_video))
        self.chunk = -1

    def next(self):
        c = self.cfg
        self.chunk += 1
        if self.chunk >= c.chunks_per_video or (self.chunk > 0 and self.rng.random() > c.p_sequential):
            self.video = int(self.rng.choice(c.n_videos, p=self.p_video))
            self.chunk = 0
        return self.video, self.chunk
