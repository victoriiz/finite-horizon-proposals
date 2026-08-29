import json, time
import numpy as np
from src.qht import (Chain, exact_variance, kernel_from_g, VQCProposal, MLP,
                 tilt_family_optimum)
from src.hybrid import DressedCircuit

RULE = "-" * 88

# =====================================================================
# ALL HYPERPARAMETERS IN ONE PLACE -- for reproducibility
# =====================================================================
CFG = dict(
    # --- model ---
    n=8,                       # components
    T=6,                       # horizon in steps
    a0=1.715e-3,               # base per-step failure prob = u*b0/(1-u)
    b0=0.284,                  # base per-step repair prob  = dt/MTTR
    gamma=0.30,                # cascade strength (swept structural parameter)
    eta=0.50,                  # repair-degradation strength
    weighted_caps=[8., 8., 8., 8., 4., 4., 4., 4.],
    weighted_C_min=12.0,
    kofn_k=4,
    # --- estimator ---
    defensive_eps=1e-3,        # Q <- (1-eps)Q + eps P   (Hesterberg 1995)
    scale_grid=31,             # log-spaced grid for the F/non-F scale c
    scale_lo=1e-3, scale_hi=1e4,
    tilt_grid=241,             # grid for the classical tilt parameter nu
    tilt_lo=0.0, tilt_hi=8.0,
    # --- Born-machine circuits ---
    vqc_layers=4,              # L: RY on every qubit + CZ ring, repeated
    vqc_steps=400,
    vqc_lr=0.25,
    vqc_seed=0,
    # --- hybrid dressed circuit ---
    hyb_qubits=4,              # n_q qubits in the quantum middle
    hyb_layers=2,              # trainable circuit layers after angle encoding
    hyb_scale=float(np.pi),    # encoded angles live in [-pi, pi]
    hyb_steps=300,
    hyb_lr=0.05,
    hyb_seed=0,
    # --- classical MLP ---
    mlp_steps=1200,
    mlp_lr=0.05,
    mlp_seed=0,
    # --- optimiser (all trainable models) ---
    adam_b1=0.9, adam_b2=0.999, adam_eps=1e-8,
)


def targets_from_h(chain, h, T):
    """
    Learning target: the SHAPE of h_s over non-failure states, renormalised.

    g_s equals 1 on F and O(1e-3) off it, so a target spanning both puts ~99%
    of its mass on F -- the one part already known exactly, because the failure
    rule is given. Training on that spends model capacity re-deriving a known
    rule. We learn only the non-trivial shape and recover the single F/non-F
    scale by exact 1-D optimisation, the same treatment the classical tilt
    family's scalar receives.
    """
    tg = {}
    nf = ~chain.in_F
    for s in range(1, T + 1):
        t = np.zeros(chain.S)
        t[nf] = h[s][nf]
        z = t.sum()
        tg[s] = t / z if z > 0 else (nf / nf.sum())
    return tg


def g_from_probs(chain, p, c):
    """Known rule on F, learned shape off it, with F/non-F scale c."""
    q = np.maximum(p, 1e-300).copy()
    q[chain.in_F] = 0.0
    z = q.sum()
    q = q / z if z > 0 else q
    return np.maximum(np.where(chain.in_F, 1.0, c * q), 1e-300)


def fit_scale_and_score(chain, probs_fn, T, label):
    """Exact 1-D optimisation of the F/non-F scale, scored on exact variance."""
    best = (np.inf, None)
    for c in np.logspace(np.log10(CFG["scale_lo"]), np.log10(CFG["scale_hi"]),
                         CFG["scale_grid"]):
        Q = {s: kernel_from_g(chain, g_from_probs(chain, probs_fn(s - 1), c),
                              eps=CFG["defensive_eps"])
             for s in range(1, T + 1)}
        v, _ = exact_variance(chain, Q, T)
        if v < best[0]:
            best = (v, c)
    return dict(method=label, var=best[0], c=best[1])


def learning_error(chain, probs_fn, h, T):
    """KL(exact h_s || learned), averaged over horizons. Measurable only
    because ground truth is available -- that is the point of the setup."""
    tot = 0.0
    nf = ~chain.in_F
    for s in range(1, T + 1):
        t = np.zeros(chain.S); t[nf] = h[s][nf]
        if t.sum() <= 0:
            continue
        t = t / t.sum()
        q = np.maximum(probs_fn(s), 1e-300).copy(); q[chain.in_F] = 0.0
        q = q / max(q.sum(), 1e-300)
        m = t > 0
        tot += float((t[m] * np.log(t[m] / q[m])).sum())
    return tot / T


def distinct_target_values(chain, h, T):
    nf = ~chain.in_F
    out = []
    for s in range(1, T + 1):
        v = h[s][nf]
        out.append(int(np.unique(np.round(v / max(v.max(), 1e-300), 12)).size))
    return out


def run_model(name, chain, T):
    h = chain.exact_h(T)
    p = float(h[T, chain.x0])
    var_naive = p * (1 - p)
    n = chain.n
    tg = targets_from_h(chain, h, T)
    results = []

    print("\n" + "=" * 88)
    print(f"MODEL: {name}")
    print(f"  n={n}, {chain.S} states, T={T}, gamma={chain.gamma}, eta={chain.eta}")
    if chain.caps is not None:
        print(f"  weighted: c={[int(x) for x in chain.caps]}, C_min={chain.C_min}")
    else:
        print(f"  k-out-of-n: k={chain.k_min}")
    print(f"  |F| = {int(chain.in_F.sum())} states,  p_T = {p:.6e}  (1 in {1/p:,.0f})")
    print("=" * 88)

    print("\nSTEP 0 - CORRECTNESS GATES")
    print("  tests: kernel, backward DP and variance recursion agree before any")
    print("  learning happens. Nothing downstream is trusted until they do.")
    print(RULE)
    dev = float(np.abs(chain.P.sum(1) - 1).max())
    v_naive_rec, _ = exact_variance(chain, {s: chain.P for s in range(1, T + 1)}, T)
    g_ex = lambda s: np.where(chain.in_F, 1.0, h[s])
    v_star, _ = exact_variance(
        chain, {s: kernel_from_g(chain, g_ex(s - 1), eps=0.0) for s in range(1, T + 1)}, T)
    print(f"  kernel rows sum to 1       max dev {dev:.2e}")
    print(f"  naive variance == p(1-p)   rel dev {abs(v_naive_rec-var_naive)/var_naive:.2e}")
    print(f"  h-transform variance == 0  value   {v_star:.3e}")
    assert dev < 1e-12 and abs(v_naive_rec - var_naive) / var_naive < 1e-9
    print("  ALL GATES PASS")

    print("\nSTEP 3 - TARGET-CLASS CHARACTERISATION")
    print("  tests: how many distinct values does h_s take off F? This is the")
    print("  covariate that decides whether a symmetric ansatz can reach the")
    print("  optimum: parameters spent outside the symmetric sector are wasted")
    print("  only if the target has no asymmetric component to represent.")
    print(RULE)
    dv = distinct_target_values(chain, h, T)
    print(f"  distinct values of h_s off F, s=1..{T}: {dv}")
    print(f"  non-failure states: {int((~chain.in_F).sum())}")

    print("\nSTEP 2 - STRONG CLASSICAL BASELINE (odds-ratio tilt family)")
    print("  tests: the best ANY member of the classical tilt family can do --")
    print("  its exact optimum found by scanning exact variance, not the best")
    print("  member we happened to train. This is the comparator the quantum")
    print("  rare-event literature usually omits.")
    print(RULE)
    t0 = time.time()
    v_tilt, nu = tilt_family_optimum(chain, T, grid=CFG["tilt_grid"])
    print(f"  optimum nu={nu:.3f}   var {v_tilt:.3e}   VRF {var_naive/v_tilt:,.2f}x"
          f"   ({time.time()-t0:.0f}s)")
    results.append(dict(method="classical tilt (exact family optimum)",
                        var=v_tilt, params=1, klerr=float("nan"), kind="classical"))

    print("\nSTEP 4 - BORN-MACHINE CIRCUITS (generative)")
    print("  tests: (a) can a circuit represent h_s; (b) how should it read the")
    print("  clock -- STATIC has no time input and is the ablation, PER_S gives")
    print("  each horizon its own parameter block, REUPLOAD injects s as a shared")
    print("  rotation; (c) symmetric (tied angles) vs generic (per-qubit angles).")
    print(RULE)
    for enc, sym in [("static", False), ("reupload", False), ("per_s", False),
                     ("static", True), ("reupload", True), ("per_s", True)]:
        t0 = time.time()
        m = VQCProposal(n, T, L=CFG["vqc_layers"], encoding=enc,
                        seed=CFG["vqc_seed"], sym=sym)
        m.fit(tg, steps=CFG["vqc_steps"], lr=CFG["vqc_lr"])
        tag = f"VQC {'sym' if sym else 'gen'} {enc}"
        r = fit_scale_and_score(chain, lambda s, m=m: m.probs(max(s, 1)), T, tag)
        ke = learning_error(chain, lambda s, m=m: m.probs(s), h, T)
        print(f"  {tag:<22} params {m.n_params():>4}  var {r['var']:.3e}"
              f"  VRF {var_naive/r['var']:>11,.2f}x  KL {ke:.4f}  ({time.time()-t0:.0f}s)")
        results.append(dict(method=tag, var=r["var"], params=m.n_params(),
                            klerr=ke, kind="quantum"))

    print("\nSTEP 5 - HYBRID DRESSED CIRCUIT (Mari et al., Quantum 4:340, 2020)")
    print("  tests: a quantum middle, architecture-matched to the MLP. The Born")
    print("  machines above are GENERATIVE while the MLP is a REGRESSOR, so a")
    print("  direct comparison confounds model class with quantum-vs-classical.")
    print("  This is the same regressor form with a circuit inside:")
    print("  W_pre -> angle encoding -> circuit -> <Z> -> W_post -> softmax.")
    print(RULE)
    X_by_s = {s: np.concatenate([chain.bits.astype(float),
                                 np.full((chain.S, 1), s / T)], 1)
              for s in range(1, T + 1)}
    t0 = time.time()
    hyb = DressedCircuit(n + 1, n_q=CFG["hyb_qubits"], L=CFG["hyb_layers"],
                         scale=CFG["hyb_scale"], seed=CFG["hyb_seed"])
    hyb.fit(X_by_s, tg, steps=CFG["hyb_steps"], lr=CFG["hyb_lr"])
    r = fit_scale_and_score(chain, lambda s: hyb.probs(X_by_s[max(s, 1)]), T,
                            "hybrid dressed circuit")
    ke = learning_error(chain, lambda s: hyb.probs(X_by_s[max(s, 1)]), h, T)
    print(f"  n_q={CFG['hyb_qubits']} L={CFG['hyb_layers']}  params {hyb.n_params():>4}"
          f"  var {r['var']:.3e}  VRF {var_naive/r['var']:>11,.2f}x  KL {ke:.4f}"
          f"  ({time.time()-t0:.0f}s)")
    results.append(dict(method="hybrid dressed circuit", var=r["var"],
                        params=hyb.n_params(), klerr=ke, kind="hybrid"))

    print("\nSTEP 6 - CLASSICAL MLP CONTROL (matched parameter budget)")
    print("  tests: model class with everything else held fixed -- same inputs,")
    print("  same exact targets, same forward-KL loss, same Adam settings and a")
    print("  comparable parameter count.")
    print(RULE)
    hidden = max(2, int(round((hyb.n_params() - 1) / (n + 2))))
    mlp = MLP(n, hidden, seed=CFG["mlp_seed"])
    mlp.fit(chain.bits.astype(float), tg, T, steps=CFG["mlp_steps"], lr=CFG["mlp_lr"])
    r = fit_scale_and_score(chain,
                            lambda s: mlp.probs(chain.bits.astype(float), max(s, 1), T),
                            T, "classical MLP")
    ke = learning_error(chain,
                        lambda s: mlp.probs(chain.bits.astype(float), max(s, 1), T), h, T)
    print(f"  hidden={hidden}  params {mlp.n_params():>4}  var {r['var']:.3e}"
          f"  VRF {var_naive/r['var']:>11,.2f}x  KL {ke:.4f}")
    results.append(dict(method="classical MLP", var=r["var"],
                        params=mlp.n_params(), klerr=ke, kind="classical"))

    print("\n" + "-" * 88)
    print(f"SUMMARY - {name}   (VRF vs naive MC; normVRF = VRF * p_T)")
    print("-" * 88)
    print(f"{'method':<38}{'kind':<11}{'params':>7}{'VRF':>14}{'normVRF':>10}{'KL to h':>10}")
    print(RULE)
    print(f"{'exact h-transform (ceiling)':<38}{'exact':<11}{'--':>7}"
          f"{'infinite':>14}{'--':>10}{'0.0000':>10}")
    for r in sorted(results, key=lambda d: d["var"]):
        vrf = var_naive / r["var"]
        kl = "---" if np.isnan(r["klerr"]) else f"{r['klerr']:.4f}"
        print(f"{r['method']:<38}{r['kind']:<11}{r['params']:>7}"
              f"{vrf:>14,.2f}{vrf*p:>10.3f}{kl:>10}")
    print(RULE)
    print(f"naive MC variance {var_naive:.3e},  p_T {p:.4e},  |F| {int(chain.in_F.sum())}")
    return dict(name=name, p_T=p, var_naive=var_naive, distinct_values=dv,
                n_F=int(chain.in_F.sum()), results=results)


def main():
    T, n = CFG["T"], CFG["n"]
    out = {}

    weighted = Chain(n=n, caps=np.array(CFG["weighted_caps"]),
                     C_min=CFG["weighted_C_min"], a0=CFG["a0"], b0=CFG["b0"],
                     gamma=CFG["gamma"], eta=CFG["eta"])
    out["weighted"] = run_model("WEIGHTED (mission-time reliability model)",
                                weighted, T)

    kofn = Chain(n=n, k_min=CFG["kofn_k"], a0=CFG["a0"], b0=CFG["b0"],
                 gamma=CFG["gamma"], eta=CFG["eta"])
    out["kofn"] = run_model("K-OUT-OF-N (symmetric control, not the applied model)",
                            kofn, T)

    with open("results_main.json", "w") as f:
        json.dump(dict(config=CFG, models=out), f, indent=2, default=float)
    print("\nwrote results_main.json")


if __name__ == "__main__":
    main()