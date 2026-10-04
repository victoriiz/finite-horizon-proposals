#!/usr/bin/env python3
"""
qht.py: Learning the time-inhomogeneous h-transform with variational circuits.

Boundary convention used here
-----------------------------
For a T-step proposal, Q_s uses g_{s-1}.  The boundary g_0 is known
analytically: g_0=1 on F and g_0=0 on F^c.  Therefore the learned models
represent only h_1,...,h_{T-1}; h_T is needed to evaluate p_T but not to
construct the proposal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# =====================================================================
# 1. CHAIN -- state space, kernel, exact h_s, exact p_T
# =====================================================================

@dataclass
class Chain:
    """
    n binary components. bit i == 1 means component i is UP.
    F = failure set (absorbing). Given x not in F, components evolve
    conditionally independently: up components fail w.p. a(x), down
    components repair w.p. b(x).

    `gamma` couples failure rate to current degradation (the cascade).
    Rates a0,b0 are set from measured cluster availability so one point of
    the family sits at a realistic operating point; gamma is a swept
    structural parameter, not a fit.
    """

    n: int = 8
    k_min: int = 4            # k-out-of-n: fails when fewer than k are up
    caps: np.ndarray = None   # if given, weighted structure instead
    C_min: float = None
    a0: float = 1.715e-3
    b0: float = 0.284
    gamma: float = 0.30
    eta: float = 0.50
    rate_law: str = "cascade"

    # "cascade" -- overload-cascade / repair-congestion form:
    #     a(x) = min(1, a0 + gamma * (C_nom/C(x) - 1)^+)
    #     b(x) = b0 / (1 + eta * k)
    #
    # "exponent" -- earlier draft form retained only for reproduction:
    #     a(x) = a0^(1 - gamma*k), b(x) = b0^(1 + eta*k/n)

    def __post_init__(self):
        n = self.n
        idx = np.arange(1 << n, dtype=np.int64)
        self.bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
        self.up = self.bits.sum(1)

        if self.caps is not None:
            self.caps = np.asarray(self.caps, float)
            self.in_F = (self.bits @ self.caps) < self.C_min
        else:
            self.in_F = self.up < self.k_min

        self.S = 1 << n
        self.x0 = self.S - 1                     # all components up
        self.P = self._kernel()

    def _rates(self):
        k = self.n - self.up                     # components down

        if self.rate_law == "exponent":
            a = np.clip(self.a0 ** np.maximum(1e-6, 1.0 - self.gamma * k), 0, 1)
            b = np.clip(self.b0 ** (1.0 + self.eta * k / self.n), 0, 1)
            return a, b

        if self.rate_law != "cascade":
            raise ValueError(f"unknown rate_law {self.rate_law!r}")

        # Overload cascade: surviving components carry the load of lost
        # capacity, so the failure rate rises with the capacity deficit.
        if self.caps is not None:
            C = self.bits @ self.caps
            C_nom = float(self.caps.sum())
        else:
            # k-out-of-n control: unit capacities, so C(x) is the up-count.
            C = self.up.astype(float)
            C_nom = float(self.n)

        with np.errstate(divide="ignore", invalid="ignore"):
            deficit = np.where(
                C > 0,
                C_nom / np.maximum(C, 1e-300) - 1.0,
                np.inf,
            )
        a = np.clip(
            self.a0 + self.gamma * np.maximum(deficit, 0.0),
            0.0,
            1.0,
        )

        # Repair congestion: a shared repair resource serves k down components.
        b = self.b0 / (1.0 + self.eta * k)
        return a, b

    def _kernel(self):
        """One-step kernel, F absorbing. Rows are product measures."""
        a, b = self._rates()
        P = np.zeros((self.S, self.S))
        upmask = self.bits.astype(bool)

        for x in range(self.S):
            if self.in_F[x]:
                P[x, x] = 1.0
                continue

            p_up = np.where(upmask[x], 1.0 - a[x], b[x])
            row = np.ones(1)
            for i in range(self.n):
                row = np.concatenate([row * (1 - p_up[i]), row * p_up[i]])
            P[x] = row

        return P

    def exact_h(self, T):
        """
        h[s][x] = P(enter F within s more steps | currently at x).

        h[0]=0 off F and h[s]=1 on F. Returns shape (T+1, S).
        h_T(x0)=p_T is the final quantity of interest.  Proposal learning,
        however, needs only h_1,...,h_{T-1}, because Q_s uses g_{s-1} and
        g_0 is known exactly.
        """
        h = np.zeros((T + 1, self.S))
        h[:, self.in_F] = 1.0
        nf = ~self.in_F
        for s in range(1, T + 1):
            h[s, nf] = self.P[nf][:, :] @ h[s - 1]
        return h

    def p_T(self, T):
        return float(self.exact_h(T)[T, self.x0])


# =====================================================================
# 2. EXACT VARIANCE OF AN ARBITRARY PROPOSAL
# =====================================================================

def exact_variance(chain: Chain, Q_list, T):
    """
    Second-moment backward recursion, stopping at first entry to F:

        M_s(x) = sum_{y in F} R_s(x,y)
                 + sum_{y not in F} R_s(x,y) M_{s-1}(y),
        R_s(x,y) = P(x,y)^2 / Q_s(x,y),   M_0 = 0 off F.

    Var = M_T(x0) - p_T^2. Exact -- no sampling.
    Q_list[s] is the proposal kernel used when s steps remain (s=1..T).
    """
    S, inF, P = chain.S, chain.in_F, chain.P
    nf = ~inF
    M = np.zeros(S)

    for s in range(1, T + 1):
        Q = Q_list[s]
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.where(Q > 0, P ** 2 / np.maximum(Q, 1e-300), 0.0)
        R[~np.isfinite(R)] = 0.0
        M_new = R[:, inF].sum(1) + R[:, nf] @ M[nf]
        M_new[inF] = 0.0
        M = M_new

    p = chain.p_T(T)
    return float(max(M[chain.x0] - p ** 2, 0.0)), p


def kernel_from_g(chain: Chain, g, eps=1e-3):
    """
    Build a Doob-style kernel from a full value vector g:

        Q(x,y) proportional to P(x,y) * g(y), row-normalised.

    In this study, learned models represent only a normalized shape on F^c.
    Callers reconstruct the full g by restoring g=1 exactly on F and a
    learned/calibrated scale on F^c before calling this function.

    Defensive mixing (Hesterberg 1995):
        Q <- (1-eps) Q + eps P.
    This keeps likelihood ratios finite when a learned proposal assigns very
    small mass to a nominally possible transition.
    """
    W = chain.P * np.maximum(g, 1e-300)[None, :]
    Z = W.sum(1, keepdims=True)
    Q = np.where(Z > 0, W / np.maximum(Z, 1e-300), 0.0)
    dead = Z[:, 0] <= 0
    Q[dead] = chain.P[dead]

    if eps > 0:
        Q = (1.0 - eps) * Q + eps * chain.P
    return Q


# =====================================================================
# 3. STATEVECTOR BORN MACHINE + PARAMETER-SHIFT GRADIENTS
# =====================================================================

# Ansatz: L layers of [RY(theta) on every qubit] then a CZ ring.
# RY and CZ are real, so the statevector stays real -- float64 is enough.

def _apply_ry(psi, n, q, ang):
    c, s = math.cos(ang / 2), math.sin(ang / 2)
    psi = psi.reshape(-1, 2, 1 << q)
    a, b = psi[:, 0, :].copy(), psi[:, 1, :].copy()
    psi[:, 0, :] = c * a - s * b
    psi[:, 1, :] = s * a + c * b
    return psi.reshape(-1)


def _cz_ring_phases(n):
    idx = np.arange(1 << n)
    ph = np.ones(1 << n)
    for q in range(n):
        r = (q + 1) % n
        both = (((idx >> q) & 1) & ((idx >> r) & 1)).astype(bool)
        ph[both] *= -1.0
    return ph


def born_probs(theta, n, L, cz):
    """theta shape (L,n). Returns a probability vector over 2^n states."""
    psi = np.zeros(1 << n)
    psi[0] = 1.0
    for l in range(L):
        for q in range(n):
            psi = _apply_ry(psi, n, q, theta[l, q])
        psi = psi * cz
    return psi ** 2


def kl_and_grad(theta, target, n, L, cz):
    """
    Forward KL(target || q_theta), up to the target entropy constant, and its
    exact parameter-shift gradient. target must be normalized.
    """
    q = np.maximum(born_probs(theta, n, L, cz), 1e-300)
    loss = float(-(target * np.log(q)).sum())
    grad = np.zeros_like(theta)
    w = target / q

    for l in range(L):
        for j in range(n):
            tp = theta.copy()
            tp[l, j] += math.pi / 2
            tm = theta.copy()
            tm[l, j] -= math.pi / 2
            dq = (
                born_probs(tp, n, L, cz) - born_probs(tm, n, L, cz)
            ) / 2.0
            grad[l, j] = -float((w * dq).sum())

    return loss, grad


# =====================================================================
# 4. TIME ENCODINGS
# =====================================================================

# A T-step proposal needs learned h_1,...,h_{T-1}.  g_0 is exact.
#
# PER_S:   independent block for each learned horizon 1,...,T-1.
# REUPLOAD: s/T is injected as an additive rotation at every layer.
# STATIC:  no s dependence; ablation for the value of the clock.

class VQCProposal:
    def __init__(self, n, T, L=3, encoding="reupload", seed=0, sym=False):
        self.n, self.T, self.L, self.enc, self.sym = n, T, L, encoding, sym
        self.cz = _cz_ring_phases(n)
        w = 1 if sym else n
        rng = np.random.default_rng(seed)

        if encoding == "per_s":
            # Learned horizons are s=1,...,T-1. Index block s by s-1.
            self.theta = rng.normal(0, 0.4, (T - 1, L, w))
        elif encoding == "reupload":
            self.theta = rng.normal(0, 0.4, (L, w))
            self.phi = rng.normal(0, 0.4, (L, w))
        elif encoding == "static":
            self.theta = rng.normal(0, 0.4, (L, w))
        else:
            raise ValueError(encoding)

    def _expand(self, th):
        return np.repeat(th, self.n, axis=-1) if self.sym else th

    def _fold(self, g):
        return g.sum(-1, keepdims=True) if self.sym else g

    def n_params(self):
        return int(
            self.theta.size
            + (self.phi.size if self.enc == "reupload" else 0)
        )

    def _theta_for(self, s):
        if self.enc == "per_s":
            if not (1 <= s < self.T):
                raise ValueError(
                    f"PER-s represents learned horizons s=1,...,{self.T - 1}; "
                    f"got s={s}"
                )
            return self.theta[s - 1]
        if self.enc == "reupload":
            return self.theta + self.phi * (s / max(self.T, 1))
        return self.theta

    def probs(self, s):
        if not (1 <= s < self.T):
            raise ValueError(
                f"learned proposal shape requested at s={s}; expected 1,...,{self.T - 1}"
            )
        return born_probs(
            self._expand(self._theta_for(s)), self.n, self.L, self.cz
        )

    def _adam(self, name, g, lr, t, state):
        m, v = state.setdefault(name, (np.zeros_like(g), np.zeros_like(g)))
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        state[name] = (m, v)
        mh = m / (1 - 0.9 ** t)
        vh = v / (1 - 0.999 ** t)
        return lr * mh / (np.sqrt(vh) + 1e-8)

    def fit(self, targets, steps=200, lr=0.25, verbose=False):
        """
        targets[s] is the exact normalized h_s shape on F^c for learned
        horizons s=1,...,T-1. PER-s trains one block per horizon; the shared
        encodings minimize the average forward-KL loss across those horizons.
        """
        hist, st = [], {}
        horizons = sorted(targets)
        if horizons != list(range(1, self.T)):
            raise ValueError(
                f"expected target horizons 1,...,{self.T - 1}; got {horizons}"
            )
        n_h = len(horizons)

        for it in range(1, steps + 1):
            if self.enc == "per_s":
                tot = 0.0
                for s in horizons:
                    block = self._theta_for(s)
                    l, g = kl_and_grad(
                        self._expand(block), targets[s], self.n, self.L, self.cz
                    )
                    idx = s - 1
                    self.theta[idx] -= self._adam(
                        f"t{s}", self._fold(g), lr, it, st
                    )
                    tot += l
                hist.append(tot / n_h)

            elif self.enc == "reupload":
                gt = np.zeros_like(self.theta)
                gp = np.zeros_like(self.phi)
                tot = 0.0
                for s in horizons:
                    th = self._expand(self._theta_for(s))
                    l, g = kl_and_grad(th, targets[s], self.n, self.L, self.cz)
                    g = self._fold(g)
                    gt += g
                    gp += g * (s / max(self.T, 1))
                    tot += l
                self.theta -= self._adam("t", gt / n_h, lr, it, st)
                self.phi -= self._adam("p", gp / n_h, lr, it, st)
                hist.append(tot / n_h)

            else:  # STATIC
                gt = np.zeros_like(self.theta)
                tot = 0.0
                for s in horizons:
                    l, g = kl_and_grad(
                        self._expand(self.theta), targets[s], self.n, self.L, self.cz
                    )
                    gt += self._fold(g)
                    tot += l
                self.theta -= self._adam("t", gt / n_h, lr, it, st)
                hist.append(tot / n_h)

            if verbose and it % 50 == 0:
                print(f"    it {it:4d}  loss {hist[-1]:.5f}")

        return hist


# =====================================================================
# 5. CLASSICAL BASELINES
# =====================================================================

def tilt_family_optimum(chain: Chain, T, grid=241):
    """
    Strong classical baseline: odds-ratio tilting of the per-component failure
    probability with scalar parameter nu. We report the best value on the fixed
    nu grid; every candidate is scored by exact downstream variance.
    """
    a, b = chain._rates()
    upmask = chain.bits.astype(bool)
    best = (np.inf, None)

    for nu in np.linspace(0.0, 8.0, grid):
        a_t = np.clip(
            a * math.exp(nu) / (1 - a + a * math.exp(nu)),
            1e-12,
            1 - 1e-12,
        )
        Q = np.zeros((chain.S, chain.S))
        for x in range(chain.S):
            if chain.in_F[x]:
                Q[x, x] = 1.0
                continue
            p_up = np.where(upmask[x], 1.0 - a_t[x], b[x])
            row = np.ones(1)
            for i in range(chain.n):
                row = np.concatenate([row * (1 - p_up[i]), row * p_up[i]])
            Q[x] = row

        v, _ = exact_variance(
            chain, {s: Q for s in range(1, T + 1)}, T
        )
        if v < best[0]:
            best = (v, nu)

    return best


class MLP:
    """
    Classical statewise-regressive control.

    It receives the same state/time inputs, exact targets, and forward-KL
    objective as the dressed circuit. Parameter count is approximately matched
    at the reference configuration, while optimizer budgets are class-specific.
    """

    def __init__(self, n, hidden, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.5, (n + 1, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.5, (hidden, 1))
        self.b2 = np.zeros(1)

    def n_params(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def _forward(self, X):
        H = np.tanh(X @ self.W1 + self.b1)
        return (H @ self.W2 + self.b2).ravel(), H

    def probs(self, bits, s, T):
        if not (1 <= s < T):
            raise ValueError(f"MLP learned horizon must be in 1,...,{T - 1}; got {s}")
        X = np.concatenate(
            [bits, np.full((bits.shape[0], 1), s / max(T, 1))], 1
        )
        z, _ = self._forward(X)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def fit(self, bits, targets, T, steps=400, lr=0.15):
        hist, st = [], {}
        horizons = sorted(targets)
        if horizons != list(range(1, T)):
            raise ValueError(
                f"expected target horizons 1,...,{T - 1}; got {horizons}"
            )
        n_h = len(horizons)

        for it in range(1, steps + 1):
            gW1 = np.zeros_like(self.W1)
            gb1 = np.zeros_like(self.b1)
            gW2 = np.zeros_like(self.W2)
            gb2 = np.zeros_like(self.b2)
            tot = 0.0

            for s in horizons:
                X = np.concatenate(
                    [bits, np.full((bits.shape[0], 1), s / max(T, 1))], 1
                )
                z, H = self._forward(X)
                z = z - z.max()
                e = np.exp(z)
                q = e / e.sum()
                t = targets[s]

                tot += float(-(t * np.log(np.maximum(q, 1e-300))).sum())
                d = (q - t)[:, None]                  # dKL/dz for softmax
                gW2 += H.T @ d
                gb2 += d.sum(0)
                dh = (d @ self.W2.T) * (1 - H ** 2)
                gW1 += X.T @ dh
                gb1 += dh.sum(0)

            for nm, p, g in (
                ("W1", self.W1, gW1),
                ("b1", self.b1, gb1),
                ("W2", self.W2, gW2),
                ("b2", self.b2, gb2),
            ):
                g = g / n_h
                m, v = st.setdefault(nm, (np.zeros_like(g), np.zeros_like(g)))
                m = 0.9 * m + 0.1 * g
                v = 0.999 * v + 0.001 * g * g
                st[nm] = (m, v)
                p -= lr * (m / (1 - 0.9 ** it)) / (
                    np.sqrt(v / (1 - 0.999 ** it)) + 1e-8
                )

            hist.append(tot / n_h)

        return hist
