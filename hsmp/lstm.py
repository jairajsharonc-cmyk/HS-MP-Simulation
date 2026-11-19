"""LSTM popularity predictor (Sec. 4.2).

Architecture (Sec. 4.2.2):  Embedding -> LSTM -> Fully-connected (score per chunk)
Training   (Sec. 4.2.3):    online, sliding history of 50 requests, sequences of 4
                            consecutive chunks -> next chunk, cross-entropy, Adam lr=1e-3
Prediction (Sec. 4.2.4):    top-5 chunk IDs for the current request sequence

Two interchangeable backends:
  * NumpyLSTM  - dependency-free, hand-written BPTT + Adam (gradient-checked in tests)
  * TorchLSTM  - standard PyTorch implementation (used if torch is installed and
                 cfg.lstm_backend == "torch")
"""
from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------------
# NumPy backend
# ----------------------------------------------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class NumpyLSTM:
    def __init__(self, vocab, embed, hidden, lr, rng):
        self.V, self.E, self.H, self.lr = vocab, embed, hidden, lr
        s = 0.1
        self.p = {
            "emb": rng.normal(0, s, (vocab, embed)),
            "Wx": rng.normal(0, s, (embed, 4 * hidden)),
            "Wh": rng.normal(0, s, (hidden, 4 * hidden)),
            "b": np.zeros(4 * hidden),
            "Wo": rng.normal(0, s, (hidden, vocab)),
            "bo": np.zeros(vocab),
        }
        self.p["b"][hidden:2 * hidden] = 1.0          # forget-gate bias = 1
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    # ---- forward over a batch of sequences X: (B, T) int ----
    def forward(self, X):
        p, H = self.p, self.H
        B, T = X.shape
        h = np.zeros((B, H)); c = np.zeros((B, H))
        cache = []
        for t in range(T):
            x = p["emb"][X[:, t]]
            z = x @ p["Wx"] + h @ p["Wh"] + p["b"]
            i = _sigmoid(z[:, :H]); f = _sigmoid(z[:, H:2 * H])
            g = np.tanh(z[:, 2 * H:3 * H]); o = _sigmoid(z[:, 3 * H:])
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            h_new = o * tc
            cache.append((X[:, t], x, h, c, i, f, g, o, tc))
            h, c = h_new, c_new
        logits = h @ p["Wo"] + p["bo"]
        return logits, h, cache

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def loss_and_grads(self, X, y):
        p, H = self.p, self.H
        B, T = X.shape
        logits, h_last, cache = self.forward(X)
        P = self._softmax(logits)
        loss = -np.mean(np.log(P[np.arange(B), y] + 1e-12))
        dlog = P.copy(); dlog[np.arange(B), y] -= 1.0; dlog /= B
        g = {k: np.zeros_like(v) for k, v in p.items()}
        g["Wo"] = h_last.T @ dlog; g["bo"] = dlog.sum(0)
        dh = dlog @ p["Wo"].T
        dc = np.zeros((B, H))
        for t in reversed(range(T)):
            xi, x, h_prev, c_prev, i, f, gg, o, tc = cache[t]
            do = dh * tc
            dc = dc + dh * o * (1 - tc ** 2)
            di = dc * gg; dg = dc * i; df = dc * c_prev
            dz = np.concatenate([di * i * (1 - i), df * f * (1 - f),
                                 dg * (1 - gg ** 2), do * o * (1 - o)], axis=1)
            g["Wx"] += x.T @ dz; g["Wh"] += h_prev.T @ dz; g["b"] += dz.sum(0)
            dx = dz @ p["Wx"].T
            np.add.at(g["emb"], xi, dx)
            dh = dz @ p["Wh"].T
            dc = dc * f
        return loss, g

    def adam_step(self, grads, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * grads[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * grads[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + eps)

    def fit_batch(self, X, y):
        loss, grads = self.loss_and_grads(X, y)
        self.adam_step(grads)
        return loss

    def predict_proba(self, seq):
        logits, _, _ = self.forward(np.asarray(seq)[None, :])
        return self._softmax(logits)[0]


# ----------------------------------------------------------------------------
# PyTorch backend (optional)
# ----------------------------------------------------------------------------
class TorchLSTM:
    def __init__(self, vocab, embed, hidden, lr, rng):
        import torch, torch.nn as nn
        self.torch = torch
        torch.manual_seed(int(rng.integers(0, 2**31 - 1)))

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.emb = nn.Embedding(vocab, embed)
                s.lstm = nn.LSTM(embed, hidden, batch_first=True)
                s.fc = nn.Linear(hidden, vocab)

            def forward(s, x):
                out, _ = s.lstm(s.emb(x))
                return s.fc(out[:, -1, :])

        self.net = Net()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.lossf = nn.CrossEntropyLoss()

    def fit_batch(self, X, y):
        t = self.torch
        self.net.train()
        xb = t.as_tensor(X, dtype=t.long); yb = t.as_tensor(y, dtype=t.long)
        self.opt.zero_grad()
        loss = self.lossf(self.net(xb), yb)
        loss.backward(); self.opt.step()
        return float(loss.item())

    def predict_proba(self, seq):
        t = self.torch
        self.net.eval()
        with t.no_grad():
            lo = self.net(t.as_tensor(np.asarray(seq)[None, :], dtype=t.long))
            return t.softmax(lo, dim=1)[0].numpy()


# ----------------------------------------------------------------------------
# Popularity predictor used by the simulator  (Algorithm 1: "MLPopularCaching")
# ----------------------------------------------------------------------------
class MLPopularCaching:
    def __init__(self, cfg, rng):
        self.cfg, self.rng = cfg, rng
        self.history: list[int] = []
        self.trained = False
        V = cfg.n_chunks_total
        Backend = NumpyLSTM
        if cfg.lstm_backend == "torch":
            try:
                import torch  # noqa: F401
                Backend = TorchLSTM
            except ImportError:
                print("[warn] torch not installed -> falling back to NumPy LSTM")
        self.model = Backend(V, cfg.lstm_embed, cfg.lstm_hidden, cfg.lstm_lr, rng)
        self.last_loss = float("nan")

    @staticmethod
    def encode(video_id, chunk_idx, chunks_per_video):      # Sec 4.2.1
        return video_id * chunks_per_video + chunk_idx

    @staticmethod
    def decode(cid, chunks_per_video):
        return divmod(int(cid), chunks_per_video)

    def record(self, cid):
        self.history.append(int(cid))
        if len(self.history) > self.cfg.history_len:        # sliding window of 50
            self.history.pop(0)

    def _dataset(self):
        L = self.cfg.seq_len
        h = np.asarray(self.history)
        if len(h) <= L:
            return None, None
        X = np.stack([h[i:i + L] for i in range(len(h) - L)])
        y = h[L:]
        return X, y

    def train(self):
        X, y = self._dataset()
        if X is None:
            return
        bs = self.cfg.lstm_batch
        for _ in range(self.cfg.lstm_epochs_per_train):
            order = self.rng.permutation(len(X))            # batch shuffling
            for s in range(0, len(X), bs):
                idx = order[s:s + bs]
                self.last_loss = self.model.fit_batch(X[idx], y[idx])
        self.trained = True

    def predict_top_k(self, k):
        L = self.cfg.seq_len
        if not self.trained or len(self.history) < L:
            return []
        proba = self.model.predict_proba(self.history[-L:])
        if self.cfg.sample_predictions:
            lg = np.log(proba + 1e-12) / self.cfg.softmax_temperature
            q = np.exp(lg - lg.max()); q /= q.sum()
            return list(self.rng.choice(len(q), size=k, replace=False, p=q))
        return list(np.argsort(-proba)[:k])


class FrequencyPredictor:
    """Heuristic popularity estimate: most frequent chunks in the recent window."""
    def __init__(self, cfg):
        self.cfg, self.history = cfg, []

    def record(self, cid):
        self.history.append(int(cid))
        if len(self.history) > self.cfg.history_len:
            self.history.pop(0)

    def predict_top_k(self, k):
        if not self.history:
            return []
        vals, cnt = np.unique(self.history, return_counts=True)
        order = np.lexsort((vals, -cnt))
        return [int(vals[i]) for i in order[:k]]
