#!/usr/bin/env python3
"""
hybrid.py: Dressed quantum circuit (hybrid classical-quantum regressor).
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------
# batched small-circuit statevector simulator
# ---------------------------------------------------------------------

def _ry_batch(psi, nq, q, ang):
    """
    Apply RY(ang) to qubit q of a batch of statevectors.
    psi : (B, 2^nq);  ang : (B,) -- one angle per batch element.
    """
    B = psi.shape[0]
    c = np.cos(ang / 2.0)[:, None]
    s = np.sin(ang / 2.0)[:, None]
    v = psi.reshape(B, -1, 2, 1 << q)
    a = v[:, :, 0, :].copy()
    b = v[:, :, 1, :].copy()
    v[:, :, 0, :] = c[:, :, None] * a - s[:, :, None] * b
    v[:, :, 1, :] = s[:, :, None] * a + c[:, :, None] * b
    return v.reshape(B, -1)


def _cz_ring(nq):
    idx = np.arange(1 << nq)
    ph = np.ones(1 << nq)
    for q in range(nq):
        r = (q + 1) % nq
        both = (((idx >> q) & 1) & ((idx >> r) & 1)).astype(bool)
        ph[both] *= -1.0
    return ph


def _z_masks(nq):
    """z_masks[i][k] = +1 if bit i of basis state k is 0 else -1."""
    idx = np.arange(1 << nq)
    return np.stack([1.0 - 2.0 * ((idx >> i) & 1) for i in range(nq)])


class DressedCircuit:
    """
    Hybrid classical-quantum regressor: W_pre -> circuit -> W_post.

    Parameters
    ----------
    n_in   : input dimension (n component bits + 1 for normalised time)
    n_q    : number of qubits in the quantum middle
    L      : number of trainable circuit layers
    scale  : angle scale applied after tanh (keeps encoded angles in [-scale, scale])
    """

    def __init__(self, n_in, n_q=4, L=2, scale=np.pi, seed=0):
        rng = np.random.default_rng(seed)
        self.n_in, self.n_q, self.L, self.scale = n_in, n_q, L, scale
        self.W_pre = rng.normal(0, 0.5, (n_in, n_q))
        self.b_pre = np.zeros(n_q)
        self.theta = rng.normal(0, 0.4, (L, n_q))     # trainable circuit angles
        self.W_post = rng.normal(0, 0.5, (n_q, 1))
        self.b_post = np.zeros(1)
        self.cz = _cz_ring(n_q)
        self.zm = _z_masks(n_q)

    def n_params(self):
        return int(self.W_pre.size + self.b_pre.size + self.theta.size
                   + self.W_post.size + self.b_post.size)

    # ---- circuit ----
    def _run_circuit(self, enc, theta):
        """
        enc   : (B, n_q) input-dependent encoding angles
        theta : (L, n_q) trainable angles (shared across the batch)
        returns <Z_i> for each qubit: (B, n_q)
        """
        B = enc.shape[0]
        psi = np.zeros((B, 1 << self.n_q))
        psi[:, 0] = 1.0
        for q in range(self.n_q):                       # angle encoding
            psi = _ry_batch(psi, self.n_q, q, enc[:, q])
        psi = psi * self.cz[None, :]
        for l in range(self.L):                         # trainable layers
            for q in range(self.n_q):
                psi = _ry_batch(psi, self.n_q, q,
                                np.full(B, theta[l, q]))
            psi = psi * self.cz[None, :]
        p = psi ** 2
        return p @ self.zm.T                            # (B, n_q)

    def _encode(self, X):
        return self.scale * np.tanh(X @ self.W_pre + self.b_pre)

    def forward(self, X):
        enc = self._encode(X)
        z = self._run_circuit(enc, self.theta)
        return (z @ self.W_post + self.b_post).ravel(), enc, z

    def probs(self, X):
        s, _, _ = self.forward(X)
        s = s - s.max()
        e = np.exp(s)
        return e / e.sum()

    # ---- gradients ----
    def _dz_dtheta(self, enc):
        """Parameter shift w.r.t. each trainable circuit angle. (L,n_q,B,n_q)"""
        out = np.zeros((self.L, self.n_q, enc.shape[0], self.n_q))
        for l in range(self.L):
            for q in range(self.n_q):
                tp = self.theta.copy(); tp[l, q] += np.pi / 2
                tm = self.theta.copy(); tm[l, q] -= np.pi / 2
                out[l, q] = (self._run_circuit(enc, tp)
                             - self._run_circuit(enc, tm)) / 2.0
        return out

    def _dz_denc(self, enc):
        """Parameter shift w.r.t. each ENCODING angle. (n_q,B,n_q)"""
        out = np.zeros((self.n_q, enc.shape[0], self.n_q))
        for q in range(self.n_q):
            ep = enc.copy(); ep[:, q] += np.pi / 2
            em = enc.copy(); em[:, q] -= np.pi / 2
            out[q] = (self._run_circuit(ep, self.theta)
                      - self._run_circuit(em, self.theta)) / 2.0
        return out

    def fit(self, X_by_s, targets, steps=300, lr=0.05, verbose=False):
        """
        Forward KL to the exact targets, same loss as every other model here.
        X_by_s[s] : (2^n, n_in) design matrix for horizon s
        targets[s]: (2^n,) normalised exact target
        """
        st, hist = {}, []
        Ts = sorted(X_by_s)
        for it in range(1, steps + 1):
            gWp = np.zeros_like(self.W_pre); gbp = np.zeros_like(self.b_pre)
            gth = np.zeros_like(self.theta)
            gWo = np.zeros_like(self.W_post); gbo = np.zeros_like(self.b_post)
            tot = 0.0
            for s in Ts:
                X, t = X_by_s[s], targets[s]
                sc, enc, z = self.forward(X)
                sc = sc - sc.max(); e = np.exp(sc); q = e / e.sum()
                tot += float(-(t * np.log(np.maximum(q, 1e-300))).sum())
                d = (q - t)[:, None]                    # dKL/dscore, (B,1)

                gWo += z.T @ d
                gbo += d.sum(0)

                dz = d @ self.W_post.T                  # (B, n_q)
                dth = self._dz_dtheta(enc)
                for l in range(self.L):
                    for qq in range(self.n_q):
                        gth[l, qq] += float((dz * dth[l, qq]).sum())

                denc = self._dz_denc(enc)               # (n_q,B,n_q)
                dang = np.stack([(dz * denc[qq]).sum(1)
                                 for qq in range(self.n_q)], 1)   # (B,n_q)
                pre = X @ self.W_pre + self.b_pre
                dpre = dang * self.scale * (1.0 - np.tanh(pre) ** 2)
                gWp += X.T @ dpre
                gbp += dpre.sum(0)

            nT = len(Ts)
            for nm, p, g in (("Wp", self.W_pre, gWp), ("bp", self.b_pre, gbp),
                             ("th", self.theta, gth), ("Wo", self.W_post, gWo),
                             ("bo", self.b_post, gbo)):
                g = g / nT
                m, v = st.setdefault(nm, (np.zeros_like(g), np.zeros_like(g)))
                m = 0.9 * m + 0.1 * g
                v = 0.999 * v + 0.001 * g * g
                st[nm] = (m, v)
                p -= lr * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
            hist.append(tot / nT)
            if verbose and it % 50 == 0:
                print(f"    it {it:4d}  loss {hist[-1]:.5f}")
        return hist
