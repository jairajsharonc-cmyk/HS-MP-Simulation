"""Structured P2P overlay (d-dimensional hypercube of peer IDs), churn, multi-path
routing, latency and energy models (Sec. 3.3, Eq. (1), Table 1)."""
from __future__ import annotations
from collections import OrderedDict
import numpy as np


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class Peer:
    __slots__ = ("pid", "cache", "load")

    def __init__(self, pid):
        self.pid = pid
        self.cache: "OrderedDict[tuple, int]" = OrderedDict()   # (v,c) -> max stored quality idx
        self.load = 1.0                                         # load factor, starts at 1.0


class Overlay:
    """Peers + directory ('Tokenizer' in Alg. 1 step 2f) mapping chunk -> holders."""

    def __init__(self, cfg, rng_churn, rng_traffic, rng_net, client_pid=0):
        self.cfg = cfg
        self.rng_churn, self.rng_traffic, self.rng_net = rng_churn, rng_traffic, rng_net
        self.n_pids = 2 ** cfg.dimension
        self.peers: dict[int, Peer] = {}
        self.index: dict[tuple, set] = {}            # chunk -> set(pid)
        self.client_pid = client_pid
        pids = [p for p in range(self.n_pids) if p != client_pid]
        chosen = list(rng_churn.choice(pids, size=cfg.initial_peers - 1, replace=False))
        for pid in [client_pid] + [int(x) for x in chosen]:
            self.peers[pid] = Peer(pid)

    # ------------------------------------------------------------ cache ops
    def insert(self, pid, chunk, quality_idx):
        """FIFO insertion. A chunk already held is upgraded in place (no reordering)."""
        peer = self.peers.get(pid)
        if peer is None:
            return
        if chunk in peer.cache:
            if quality_idx > peer.cache[chunk]:
                peer.cache[chunk] = quality_idx
            return
        if len(peer.cache) >= self.cfg.cache_capacity:
            old, _ = peer.cache.popitem(last=False)            # evict the oldest (FIFO)
            self.index[old].discard(pid)
        peer.cache[chunk] = quality_idx
        self.index.setdefault(chunk, set()).add(pid)

    def seed(self, rng):
        """Alg.1 step 1c: 2 replicas per chunk spread across peers (stored at full quality).
        Replicas go to the currently least-loaded peers (random tie-break) so that initial
        seeding never overflows a 180-chunk FIFO cache."""
        top = len(self.cfg.ladder_mbps) - 1
        pids = np.array(list(self.peers))
        k = min(self.cfg.replicas_per_chunk, len(pids))
        for v in range(self.cfg.n_videos):
            for c in range(self.cfg.chunks_per_video):
                fill = np.array([len(self.peers[int(p)].cache) for p in pids])
                order = np.lexsort((rng.random(len(pids)), fill))
                for pid in pids[order[:k]]:
                    self.insert(int(pid), (v, c), top)

    # ------------------------------------------------------------ churn
    def churn_step(self):
        """Alg.1 step 2a. Returns (joined, left) booleans."""
        cfg, r = self.cfg, self.rng_churn
        joined = left = False
        if r.random() < cfg.p_join and len(self.peers) < cfg.max_peers:
            free = [p for p in range(self.n_pids) if p not in self.peers]
            pid = int(r.choice(free))
            self.peers[pid] = Peer(pid)
            joined = True
        if r.random() < cfg.p_leave and len(self.peers) > cfg.min_peers:
            cand = [p for p in self.peers if p != self.client_pid]
            pid = int(r.choice(cand))
            self._remove(pid)
            left = True
        return joined, left

    def _remove(self, pid):
        peer = self.peers.pop(pid)
        orphaned = []
        for chunk in peer.cache:
            self.index[chunk].discard(pid)
            if len(self.index[chunk]) < self.cfg.replicas_per_chunk:
                orphaned.append(chunk)
        if self.cfg.repair_on_leave:                       # assumption, see Config
            top = len(self.cfg.ladder_mbps) - 1
            pids = list(self.peers)
            for chunk in orphaned:
                while len(self.index[chunk]) < self.cfg.replicas_per_chunk:
                    self.insert(int(self.rng_churn.choice(pids)), chunk, top)

    # ------------------------------------------------------------ traffic
    def update_traffic(self, t):
        """Alg.1 step 2b: pattern = PATTERNS[t mod 3]; load = factor + clipped Gaussian noise."""
        cfg = self.cfg
        k = t % len(cfg.traffic_factors)
        base = cfg.traffic_factors[k]
        lo, hi = cfg.load_clip
        for p in self.peers.values():
            p.load = float(np.clip(base + self.rng_traffic.normal(0, cfg.load_noise_std), lo, hi))
        return cfg.traffic_names[k]

    # ------------------------------------------------------------ lookup + routing
    def locate(self, chunk, req_q):
        """Find the best holder for `chunk` at quality `req_q` (fallback to lower quality).

        A replica stored at quality q can serve any request <= q (scalable-coding
        assumption). Returns (holder_pid, served_quality, hops, path_nodes) or None.
        """
        holders = self.index.get(chunk)
        if not holders:
            return None
        cl = self.client_pid
        best_q = max(self.peers[h].cache[chunk] for h in holders)
        served_q = min(req_q, best_q)
        cands = [h for h in holders if self.peers[h].cache[chunk] >= served_q]
        h_min = min(hamming(cl, h) for h in cands)
        near = [h for h in cands if hamming(cl, h) == h_min]
        holder = min(near, key=lambda h: self.peers[h].load)
        hops, path = self._route(cl, holder)
        return holder, served_q, hops, path

    def _route(self, src, dst):
        """Multi-path routing: examine up to n_paths random dimension orders (all are
        shortest paths on a hypercube) and take the one with the lowest summed load."""
        diff = [i for i in range(self.cfg.dimension) if (src ^ dst) >> i & 1]
        hops = len(diff)
        if hops == 0:
            return 0, []
        best, best_cost = None, np.inf
        for _ in range(min(self.cfg.n_paths, max(1, hops))):
            order = list(self.rng_net.permutation(diff))
            node, nodes = src, []
            for dim in order:
                node ^= (1 << int(dim))
                nodes.append(node)
            cost = sum(self.peers[n].load if n in self.peers else 1.0 for n in nodes)
            if cost < best_cost:
                best, best_cost = nodes, cost
        return hops, best

    # ------------------------------------------------------------ models (Sec 3.3.4)
    def latency_ms(self, hops, path, size_bytes):
        cfg = self.cfg
        if hops == 0:
            return cfg.local_proc_ms
        prop = 0.0
        loads = []
        for n in path:
            lf = self.peers[n].load if n in self.peers else 1.0
            loads.append(lf)
            prop += cfg.hop_delay_ms * lf + self.rng_net.uniform(-cfg.jitter_ms, cfg.jitter_ms)
        eff_bw = cfg.link_bw_bps / max(np.mean(loads), 1e-6)
        tx = size_bytes * 8.0 / eff_bw * 1e3
        return max(prop, 0.0) + tx

    def energy_j(self, hops, size_bytes):
        """Eq. (1): E = size*(Tx+Rx) + hops*routing ; local cache: processing only."""
        cfg = self.cfg
        if hops == 0 and cfg.local_energy_discount:
            return size_bytes * cfg.e_local_per_byte
        return size_bytes * (cfg.e_tx_per_byte + cfg.e_rx_per_byte) + hops * cfg.e_hop

    # ------------------------------------------------------------ proactive caching
    def pick_targets(self, n):
        """Alg.1 step 2k: choose target peers by Hamming distance to the client, then load."""
        cl = self.client_pid
        cands = [p for p in self.peers.values() if p.pid != cl]
        cands.sort(key=lambda p: (hamming(cl, p.pid), p.load))
        return [p.pid for p in cands[:n]]
