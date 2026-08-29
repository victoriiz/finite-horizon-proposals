from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

# =====================================================================
# 1. CHAIN 
# =====================================================================

@dataclass
class Chain:
    """
    n binary components. bit i == 1 means component i is UP.
    F = failure set (absorbing). Given x not in F, components evolve
    conditionally independently: up components fail w.p. a(x), down
    components repair w.p. b(x).

    `structure` selects the failure rule; `gamma` couples failure rate to
    current degradation (the cascade).  Rates a0,b0 are set from measured
    cluster availability so one point of the family sits at a realistic
    operating point; gamma is a swept structural parameter, not a fit.
    """
    n: int = 8
    k_min: int = 4            # k-out-of-n: fails when fewer than k are up
    caps: np.ndarray = None   # if given, weighted structure instead
    C_min: float = None
    a0: float = 1.715e-3
    b0: float = 0.284
    gamma: float = 0.30
    eta: float = 0.50

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
        a = np.clip(self.a0 ** np.maximum(1e-6, 1.0 - self.gamma * k), 0, 1)
        b = np.clip(self.b0 ** (1.0 + self.eta * k / self.n), 0, 1)
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
        h[s][x] = P(enter F within s more steps | currently at x, x not in F).
        h[0] = 0 off F. Returns array of shape (T+1, S) with h[s][x]=1 on F.
        THIS IS THE OBJECT WE ARE LEARNING.
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

        M_s(x) = sum_{y in F} R_s(x,y) + sum_{y not in F} R_s(x,y) M_{s-1}(y),
        R_s(x,y) = P(x,y)^2 / Q_s(x,y),   M_0 = 0 off F.

    Var = M_T(x0) - p_T^2.  Exact -- no sampling.
    Q_list[s] is the proposal kernel used when s steps remain (s = 1..T).
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
    Build the h-transform kernel from a value function g over states:
        Q(x,y) proportional to P(x,y) * g(y),  row-normalised.
    g == exact g_s recovers the zero-variance proposal.
    Only RATIOS of g matter -- across ALL states, F included -- which is why a
    normalised Born machine suffices PROVIDED the target it is trained on also
    includes the F states at their true value of 1.

    DEFENSIVE MIXING (Hesterberg, Technometrics 37:185, 1995): we return
        (1-eps) Q_learned + eps P.
    Without it a learned proposal that assigns near-zero mass where P has mass
    produces unbounded likelihood ratios and the estimator's variance can
    exceed naive Monte Carlo by orders. With it the ratio is bounded by 1/eps.
    This is a property of the ESTIMATOR, not a fudge: it is what any deployed
    learned proposal would have to do.
    """
    W = chain.P * np.maximum(g, 1e-300)[None, :]
    Z = W.sum(1, keepdims=True)
    Q = np.where(Z > 0, W / np.maximum(Z, 1e-300), 0.0)
    dead = (Z[:, 0] <= 0)
    Q[dead] = chain.P[dead]
    if eps > 0:
        Q = (1.0 - eps) * Q + eps * chain.P
    return Q


# =====================================================================
# 3. STATEVECTOR BORN MACHINE + PARAMETER-SHIFT GRADIENTS
# =====================================================================

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
    """theta shape (L, n). Returns probability vector over 2^n states."""
    psi = np.zeros(1 << n); psi[0] = 1.0
    for l in range(L):
        for q in range(n):
            psi = _apply_ry(psi, n, q, theta[l, q])
        psi = psi * cz
    return psi ** 2


def kl_and_grad(theta, target, n, L, cz):
    """
    Forward KL( target || q_theta ) up to a constant, and its exact gradient
    by parameter shift.  target must be a normalised probability vector.
    dKL/dtheta_j = - sum_y target(y)/q(y) * dq(y)/dtheta_j
    dq/dtheta_j   = [ q(theta_j + pi/2) - q(theta_j - pi/2) ] / 2
    """
    q = np.maximum(born_probs(theta, n, L, cz), 1e-300)
    loss = float(-(target * np.log(q)).sum())
    grad = np.zeros_like(theta)
    w = target / q
    for l in range(L):
        for j in range(n):
            tp = theta.copy(); tp[l, j] += math.pi / 2
            tm = theta.copy(); tm[l, j] -= math.pi / 2
            dq = (born_probs(tp, n, L, cz) - born_probs(tm, n, L, cz)) / 2.0
            grad[l, j] = -float((w * dq).sum())
    return loss, grad


# =====================================================================
# 4. TIME ENCODINGS -- the actual design question
# =====================================================================
#
# How does a circuit read "how many steps are left"? 
#
#   PER_S   separate parameter block per horizon s.  Most parameters,
#           no sharing, an upper bound on what this ansatz family can do.
#   REUPLOAD  s injected as a rotation angle at every layer, parameters
#           shared across horizons (Perez-Salinas et al., Quantum 4:226).
#   STATIC  no s dependence at all.  ABLATION: isolates the value of
#           the clock, which is the whole point of the paper.

class VQCProposal:
    """
    `sym=True` ties every qubit in a layer to one shared angle, making the
    circuit permutation-symmetric (up to the CZ ring). For a failure rule that
    depends only on how many components are up -- k-out-of-n, and any threshold
    on an exchangeable load -- the exact h_s is CONSTANT within each Hamming
    weight, so the optimal proposal lies in the symmetric sector and a tied
    ansatz reaches it with O(L) parameters instead of O(Ln).

    This is the mechanism behind the empirical finding in the QCE work that
    structural alignment beats circuit capacity: it is not that symmetry is
    a useful prior, it is that the target has no asymmetric component at all.
    """
    def __init__(self, n, T, L=3, encoding="reupload", seed=0, sym=False):
        self.n, self.T, self.L, self.enc, self.sym = n, T, L, encoding, sym
        self.cz = _cz_ring_phases(n)
        w = 1 if sym else n
        rng = np.random.default_rng(seed)
        if encoding == "per_s":
            self.theta = rng.normal(0, 0.4, (T + 1, L, w))
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
        return int(self.theta.size + (self.phi.size if self.enc == "reupload" else 0))

    def _theta_for(self, s):
        if self.enc == "per_s":
            return self.theta[s]
        if self.enc == "reupload":
            return self.theta + self.phi * (s / max(self.T, 1))
        return self.theta

    def probs(self, s):
        return born_probs(self._expand(self._theta_for(s)), self.n, self.L, self.cz)

    def _adam(self, name, g, lr, t, state):
        m, v = state.setdefault(name, (np.zeros_like(g), np.zeros_like(g)))
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        state[name] = (m, v)
        mh = m / (1 - 0.9 ** t); vh = v / (1 - 0.999 ** t)
        return lr * mh / (np.sqrt(vh) + 1e-8)

    def fit(self, targets, steps=200, lr=0.25, verbose=False):
        """
        targets[s] = exact normalised h_s restricted to the support we care
        about.  Trained per horizon for per_s; jointly (summed loss) for the
        shared-parameter encodings, which is what makes the comparison fair.
        """
        hist, st = [], {}
        for it in range(1, steps + 1):
            if self.enc == "per_s":
                tot = 0.0
                for s in range(1, self.T + 1):
                    l, g = kl_and_grad(self._expand(self.theta[s]), targets[s],
                                       self.n, self.L, self.cz)
                    self.theta[s] -= self._adam(f"t{s}", self._fold(g), lr, it, st)
                    tot += l
                hist.append(tot / self.T)
            elif self.enc == "reupload":
                gt = np.zeros_like(self.theta); gp = np.zeros_like(self.phi); tot = 0.0
                for s in range(1, self.T + 1):
                    th = self._expand(self._theta_for(s))
                    l, g = kl_and_grad(th, targets[s], self.n, self.L, self.cz)
                    g = self._fold(g)
                    gt += g; gp += g * (s / max(self.T, 1)); tot += l
                self.theta -= self._adam("t", gt / self.T, lr, it, st)
                self.phi -= self._adam("p", gp / self.T, lr, it, st)
                hist.append(tot / self.T)
            else:
                gt = np.zeros_like(self.theta); tot = 0.0
                for s in range(1, self.T + 1):
                    l, g = kl_and_grad(self._expand(self.theta), targets[s],
                                       self.n, self.L, self.cz)
                    gt += self._fold(g); tot += l
                self.theta -= self._adam("t", gt / self.T, lr, it, st)
                hist.append(tot / self.T)
            if verbose and it % 50 == 0:
                print(f"    it {it:4d}  loss {hist[-1]:.5f}")
        return hist


# =====================================================================
# 5. CLASSICAL BASELINES
# =====================================================================

def tilt_family_optimum(chain: Chain, T, grid=241):
    """
    Strong classical baseline: odds-ratio tilting of the per-component failure
    probability, scalar parameter nu.  We report its EXACT family optimum,
    found by scanning the exact variance -- not the achieved variance of one
    trained member.  This is the comparator the quantum literature usually
    omits (cf. Rubinstein & Kroese 2004 for the CE alternative).
    """
    a, b = chain._rates()
    upmask = chain.bits.astype(bool)
    best = (np.inf, None)
    for nu in np.linspace(0.0, 8.0, grid):
        a_t = np.clip(a * math.exp(nu) / (1 - a + a * math.exp(nu)), 1e-12, 1 - 1e-12)
        Q = np.zeros((chain.S, chain.S))
        for x in range(chain.S):
            if chain.in_F[x]:
                Q[x, x] = 1.0; continue
            p_up = np.where(upmask[x], 1.0 - a_t[x], b[x])
            row = np.ones(1)
            for i in range(chain.n):
                row = np.concatenate([row * (1 - p_up[i]), row * p_up[i]])
            Q[x] = row
        v, _ = exact_variance(chain, {s: Q for s in range(1, T + 1)}, T)
        if v < best[0]:
            best = (v, nu)
    return best


class MLP:
    """
    Quantum-inspired / classical control: same job, same information, matched
    parameter budget.  Input = component bits plus normalised steps-remaining;
    output = log g.  Trained on the same exact targets with the same forward
    KL, so any gap is about the model class and nothing else.
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
        X = np.concatenate([bits, np.full((bits.shape[0], 1), s / max(T, 1))], 1)
        z, _ = self._forward(X)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def fit(self, bits, targets, T, steps=400, lr=0.15):
        hist, st = [], {}
        for it in range(1, steps + 1):
            gW1 = np.zeros_like(self.W1); gb1 = np.zeros_like(self.b1)
            gW2 = np.zeros_like(self.W2); gb2 = np.zeros_like(self.b2)
            tot = 0.0
            for s in range(1, T + 1):
                X = np.concatenate([bits, np.full((bits.shape[0], 1), s / max(T, 1))], 1)
                z, H = self._forward(X)
                z = z - z.max(); e = np.exp(z); q = e / e.sum()
                t = targets[s]
                tot += float(-(t * np.log(np.maximum(q, 1e-300))).sum())
                d = (q - t)[:, None]                      # dKL/dz for softmax
                gW2 += H.T @ d; gb2 += d.sum(0)
                dh = (d @ self.W2.T) * (1 - H ** 2)
                gW1 += X.T @ dh; gb1 += dh.sum(0)
            for nm, p, g in (("W1", self.W1, gW1), ("b1", self.b1, gb1),
                             ("W2", self.W2, gW2), ("b2", self.b2, gb2)):
                m, v = st.setdefault(nm, (np.zeros_like(g), np.zeros_like(g)))
                m = 0.9 * m + 0.1 * (g / T); v = 0.999 * v + 0.001 * (g / T) ** 2
                st[nm] = (m, v)
                p -= lr * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
            hist.append(tot / T)
        return hist