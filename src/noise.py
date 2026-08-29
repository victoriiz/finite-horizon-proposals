from __future__ import annotations
import os, json
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

from qht import Chain, exact_variance, kernel_from_g, VQCProposal, MLP
from hybrid import DressedCircuit, _z_masks
from run_study import CFG, targets_from_h, g_from_probs, learning_error

T, n = CFG["T"], CFG["n"]
SHOTS = [100, 1000, 10_000, 100_000, None]   
DEPOL = [0.0, 0.01, 0.05]
SEEDS = [0, 1, 2]


# ------------------------------------------------------------------ noise
def apply_depolarising(p, lam):
    """Mix an output distribution with the uniform distribution at rate lam."""
    if lam <= 0:
        return p
    return (1.0 - lam) * p + lam / p.size


def shot_sample_distribution(p, shots, rng):
    """Empirical frequencies from `shots` measurements of a distribution."""
    if shots is None:
        return p
    p = np.maximum(p, 0)
    p = p / p.sum()
    return rng.multinomial(shots, p).astype(float) / shots


def shot_sample_expectations(z, shots, rng):
    """
    Finite-shot estimate of single-qubit expectations. For each qubit,
    P(outcome 1) = (1 - <Z>)/2, so the count of ones is Binomial(shots, p),
    and the estimator is <Z> = 1 - 2 * count / shots.
    """
    if shots is None:
        return z
    p1 = np.clip((1.0 - z) / 2.0, 0.0, 1.0)
    counts = rng.binomial(shots, p1)
    return 1.0 - 2.0 * counts / shots


# ------------------------------------------- noisy readout for each model
def born_probs_noisy(model, s, shots, lam, rng):
    p = model.probs(max(s, 1))
    p = apply_depolarising(p, lam)
    return shot_sample_distribution(p, shots, rng)


def dressed_probs_noisy(model, X, s, shots, lam, rng):
    """
    Re-run the dressed circuit's forward pass with a noisy measurement stage.
    The classical layers are exact; only the readout is perturbed, which is
    where hardware noise actually enters this architecture.
    """
    Xs = X[max(s, 1)]
    enc = model._encode(Xs)
    B = enc.shape[0]
    psi = np.zeros((B, 1 << model.n_q))
    psi[:, 0] = 1.0
    from hybrid import _ry_batch
    for q in range(model.n_q):
        psi = _ry_batch(psi, model.n_q, q, enc[:, q])
    psi = psi * model.cz[None, :]
    for l in range(model.L):
        for q in range(model.n_q):
            psi = _ry_batch(psi, model.n_q, q, np.full(B, model.theta[l, q]))
        psi = psi * model.cz[None, :]
    pmeas = psi ** 2
    pmeas = np.stack([apply_depolarising(row, lam) for row in pmeas])
    z = pmeas @ model.zm.T
    z = shot_sample_expectations(z, shots, rng)
    sc = (z @ model.W_post + model.b_post).ravel()
    sc = sc - sc.max()
    e = np.exp(sc)
    return e / e.sum()


# ------------------------------------------------------------------ score
def score(chain, probs_fn, T, grid=21):
    best = np.inf
    for c in np.logspace(-3, 4, grid):
        Q = {s: kernel_from_g(chain, g_from_probs(chain, probs_fn(s - 1), c),
                              eps=CFG["defensive_eps"])
             for s in range(1, T + 1)}
        v, _ = exact_variance(chain, Q, T)
        best = min(best, v)
    return best


def main():
    ch = Chain(n=n, caps=np.array(CFG["weighted_caps"]), C_min=CFG["weighted_C_min"],
               a0=CFG["a0"], b0=CFG["b0"], gamma=CFG["gamma"], eta=CFG["eta"])
    h = ch.exact_h(T)
    p = float(h[T, ch.x0])
    vn = p * (1 - p)
    tg = targets_from_h(ch, h, T)
    X = {s: np.concatenate([ch.bits.astype(float), np.full((ch.S, 1), s / T)], 1)
         for s in range(1, T + 1)}

    print("=" * 78)
    print("FINITE-SHOT AND DEPOLARISING EVALUATION")
    print(f"  weighted structure, n={n}, T={T}, p_T={p:.4e}, naive var={vn:.4e}")
    print("  models trained under exact supervision, noise applied at readout only")
    print("=" * 78)

    out = {"p_T": p, "var_naive": vn, "rows": []}

    print("\nThe classical regressor has no readout channel: its proposal is")
    print("deterministic and therefore exact at every shot budget. We include")
    print("it as a flat reference line.\n")
    mlp_v = []
    for sd in SEEDS:
        m = MLP(n, 16, seed=sd)
        m.fit(ch.bits.astype(float), tg, T, steps=CFG["mlp_steps"], lr=CFG["mlp_lr"])
        mlp_v.append(vn / score(ch, lambda s, m=m: m.probs(ch.bits.astype(float),
                                                           max(s, 1), T), T))
    mlp_med = float(np.median(mlp_v))
    print(f"  classical MLP, exact readout, median over {len(SEEDS)} seeds:"
          f" VRF {mlp_med:,.1f}\n")
    out["mlp_exact_vrf"] = mlp_med

    for lam in DEPOL:
        print(f"--- depolarising rate lambda = {lam} " + "-" * 40)
        print(f"{'shots':>10}{'hybrid VRF':>14}{'Born VRF':>12}"
              f"{'hybrid / MLP':>15}")
        for shots in SHOTS:
            hv, bv = [], []
            for sd in SEEDS:
                rng = np.random.default_rng(1000 + sd)
                hy = DressedCircuit(n + 1, n_q=CFG["hyb_qubits"], L=CFG["hyb_layers"],
                                    scale=CFG["hyb_scale"], seed=sd)
                hy.fit(X, tg, steps=CFG["hyb_steps"], lr=CFG["hyb_lr"])
                hv.append(vn / score(ch, lambda s, hy=hy, r=rng:
                                     dressed_probs_noisy(hy, X, s, shots, lam, r), T))
                bm = VQCProposal(n, T, L=CFG["vqc_layers"], encoding="static",
                                 seed=sd, sym=True)
                bm.fit(tg, steps=CFG["vqc_steps"], lr=CFG["vqc_lr"])
                bv.append(vn / score(ch, lambda s, bm=bm, r=rng:
                                     born_probs_noisy(bm, s, shots, lam, r), T))
            hm, bm_ = float(np.median(hv)), float(np.median(bv))
            lab = "exact" if shots is None else f"{shots:,}"
            print(f"{lab:>10}{hm:>14,.2f}{bm_:>12,.2f}{hm/mlp_med:>15.4f}")
            out["rows"].append(dict(lam=lam, shots=shots, hybrid=hv, born=bv,
                                    hybrid_med=hm, born_med=bm_))
        print()

    json.dump(out, open(os.path.join(RESULTS, "results_noise.json"), "w"),
              indent=2, default=float)
    print(f"wrote {RESULTS}/results_noise.json")


if __name__ == "__main__":
    main()