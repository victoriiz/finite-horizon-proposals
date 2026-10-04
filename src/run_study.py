"""
run_study.py: main experiment with the corrected finite-horizon boundary.

A T-step proposal uses Q_s proportional to P * g_{s-1}.  g_0 is known
exactly, so learned models are trained only on h_1,...,h_{T-1}.
"""

import json
import os
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_ROOT, "results")
FIGURES = os.path.join(_ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

from qht import (
    Chain,
    MLP,
    VQCProposal,
    exact_variance,
    kernel_from_g,
    tilt_family_optimum,
)
from hybrid import DressedCircuit

RULE = "-" * 88


# =====================================================================
# ALL HYPERPARAMETERS IN ONE PLACE -- for reproducibility
# =====================================================================

CFG = dict(
    # --- model ---
    n=8,
    T=6,
    a0=1.715e-3,
    b0=0.284,
    gamma=0.30,
    eta=0.50,
    weighted_caps=[8.0, 8.0, 8.0, 8.0, 4.0, 4.0, 4.0, 4.0],
    weighted_C_min=12.0,
    kofn_k=4,
    # --- estimator ---
    defensive_eps=1e-3,
    scale_grid=31,
    scale_lo=1e-3,
    scale_hi=1e4,
    tilt_grid=241,
    tilt_lo=0.0,
    tilt_hi=8.0,
    # --- Born-machine circuits ---
    vqc_layers=4,
    vqc_steps=400,
    vqc_lr=0.25,
    vqc_seed=0,
    # --- hybrid dressed circuit ---
    hyb_qubits=4,
    hyb_layers=2,
    hyb_scale=float(np.pi),
    hyb_steps=300,
    hyb_lr=0.05,
    hyb_seed=0,
    # --- classical MLP ---
    mlp_steps=1200,
    mlp_lr=0.05,
    mlp_seed=0,
    # --- optimiser constants ---
    adam_b1=0.9,
    adam_b2=0.999,
    adam_eps=1e-8,
)


def targets_from_h(chain, h, T):
    """
    Exact learning targets: normalized h_s shapes on F^c for s=1,...,T-1.

    A T-step proposal uses Q_s based on g_{s-1}.  The boundary g_0 is known
    analytically (1 on F and 0 on F^c), so there is no learned target at s=0.
    h_T is used to evaluate p_T but is not needed to construct Q_1,...,Q_T.

    Values on F are not learned because g_s=1 there is already known exactly.
    """
    tg = {}
    nf = ~chain.in_F

    for s in range(1, T):
        t = np.zeros(chain.S)
        t[nf] = h[s][nf]
        z = t.sum()
        if z <= 0:
            raise ValueError(f"h_{s} has zero mass on F^c")
        tg[s] = t / z

    return tg


def g_from_probs(chain, p, c):
    """Restore full g: exact value 1 on F, learned normalized shape on F^c."""
    q = np.maximum(p, 1e-300).copy()
    q[chain.in_F] = 0.0
    z = q.sum()
    if z <= 0:
        raise ValueError("learned proposal has no mass on F^c")
    q /= z
    return np.where(chain.in_F, 1.0, c * q)


def fit_scale_and_score(chain, probs_fn, T, label):
    """
    Grid-select the shared F/non-F scale c using exact downstream variance.

    Q_1 uses the exactly known g_0. For s=2,...,T, Q_s uses the learned
    normalized shape of h_{s-1} with scale c.
    """
    g0 = np.where(chain.in_F, 1.0, 0.0)
    best = (np.inf, None)

    for c in np.logspace(
        np.log10(CFG["scale_lo"]),
        np.log10(CFG["scale_hi"]),
        CFG["scale_grid"],
    ):
        Q = {
            1: kernel_from_g(chain, g0, eps=CFG["defensive_eps"]),
        }
        for s in range(2, T + 1):
            g_hat = g_from_probs(chain, probs_fn(s - 1), c)
            Q[s] = kernel_from_g(chain, g_hat, eps=CFG["defensive_eps"])

        v, _ = exact_variance(chain, Q, T)
        if v < best[0]:
            best = (v, c)

    return dict(method=label, var=best[0], c=best[1])


def learning_error(chain, probs_fn, h, T):
    """
    Forward KL(exact normalized h_s || learned normalized shape), averaged
    over the learned horizons s=1,...,T-1.
    """
    tot = 0.0
    nf = ~chain.in_F
    horizons = range(1, T)

    for s in horizons:
        t = np.zeros(chain.S)
        t[nf] = h[s][nf]
        t /= t.sum()

        q = np.maximum(probs_fn(s), 1e-300).copy()
        q[chain.in_F] = 0.0
        q /= max(q.sum(), 1e-300)

        m = t > 0
        tot += float((t[m] * np.log(t[m] / q[m])).sum())

    return tot / (T - 1)


def distinct_target_values(chain, h, T):
    """Descriptive exact-target diagnostic, retained for all s=1,...,T."""
    nf = ~chain.in_F
    out = []
    for s in range(1, T + 1):
        v = h[s][nf]
        out.append(
            int(np.unique(np.round(v / max(v.max(), 1e-300), 12)).size)
        )
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
    print(
        f"  n={n}, {chain.S} states, T={T}, "
        f"gamma={chain.gamma}, eta={chain.eta}"
    )
    if chain.caps is not None:
        print(
            f"  weighted: c={[int(x) for x in chain.caps]}, "
            f"C_min={chain.C_min}"
        )
    else:
        print(f"  k-out-of-n: k={chain.k_min}")
    print(
        f"  |F| = {int(chain.in_F.sum())} states, "
        f"p_T = {p:.6e}  (1 in {1/p:,.0f})"
    )
    print("=" * 88)

    print("\nSTEP 0 - CORRECTNESS GATES")
    print("  tests: kernel, backward DP and variance recursion agree before any")
    print("  learning happens. Nothing downstream is trusted until they do.")
    print(RULE)

    dev = float(np.abs(chain.P.sum(1) - 1).max())
    v_naive_rec, _ = exact_variance(
        chain,
        {s: chain.P for s in range(1, T + 1)},
        T,
    )
    g_ex = lambda s: np.where(chain.in_F, 1.0, h[s])
    v_star, _ = exact_variance(
        chain,
        {
            s: kernel_from_g(chain, g_ex(s - 1), eps=0.0)
            for s in range(1, T + 1)
        },
        T,
    )

    print(f"  kernel rows sum to 1       max dev {dev:.2e}")
    print(
        "  naive variance == p(1-p)   rel dev "
        f"{abs(v_naive_rec-var_naive)/var_naive:.2e}"
    )
    print(f"  h-transform variance == 0  value   {v_star:.3e}")

    assert dev < 1e-12
    assert abs(v_naive_rec - var_naive) / var_naive < 1e-9
    assert set(tg) == set(range(1, T))

    # Explicit boundary check.
    g0 = np.where(chain.in_F, 1.0, 0.0)
    Q1 = kernel_from_g(chain, g0, eps=CFG["defensive_eps"])
    assert np.allclose(Q1.sum(axis=1), 1.0)

    # PER-s has exactly T-1 learned blocks.
    msym = VQCProposal(
        n,
        T,
        L=CFG["vqc_layers"],
        encoding="per_s",
        seed=0,
        sym=True,
    )
    mgen = VQCProposal(
        n,
        T,
        L=CFG["vqc_layers"],
        encoding="per_s",
        seed=0,
        sym=False,
    )
    assert msym.n_params() == (T - 1) * CFG["vqc_layers"]
    assert mgen.n_params() == (T - 1) * CFG["vqc_layers"] * n

    print("  learned horizons           s=1,...,T-1")
    print(
        "  PER-s parameter counts     "
        f"symmetric={msym.n_params()}, generic={mgen.n_params()}"
    )
    print("  ALL GATES PASS")

    print("\nSTEP 1 - TARGET-CLASS CHARACTERISATION")
    print("  diagnostic: number of distinct h_s values on F^c.")
    print("  We test whether this quantity predicts the benefit of parameter")
    print("  tying; the experiments do not assume that it does.")
    print(RULE)
    dv = distinct_target_values(chain, h, T)
    print(f"  distinct values of h_s off F, s=1..{T}: {dv}")
    print(f"  non-failure states: {int((~chain.in_F).sum())}")

    print("\nSTEP 2 - STRONG CLASSICAL BASELINE (odds-ratio tilt family)")
    print("  tests: best value on the fixed tilt grid; every candidate is scored")
    print("  using exact downstream variance.")
    print(RULE)
    t0 = time.time()
    v_tilt, nu = tilt_family_optimum(chain, T, grid=CFG["tilt_grid"])
    print(
        f"  best-grid nu={nu:.3f}   var {v_tilt:.3e}   "
        f"VRF {var_naive/v_tilt:,.2f}x   ({time.time()-t0:.0f}s)"
    )
    results.append(
        dict(
            method="classical tilt (best grid value)",
            var=v_tilt,
            params=1,
            klerr=float("nan"),
            kind="classical",
        )
    )

    print("\nSTEP 3 - BORN-MACHINE CIRCUITS (generative)")
    print("  tests horizon encoding and symmetry. Learned horizons are h_1,...,h_{T-1};")
    print("  the final-step boundary g_0 is inserted analytically at proposal scoring.")
    print(RULE)

    for enc, sym in [
        ("static", False),
        ("reupload", False),
        ("per_s", False),
        ("static", True),
        ("reupload", True),
        ("per_s", True),
    ]:
        t0 = time.time()
        m = VQCProposal(
            n,
            T,
            L=CFG["vqc_layers"],
            encoding=enc,
            seed=CFG["vqc_seed"],
            sym=sym,
        )
        m.fit(tg, steps=CFG["vqc_steps"], lr=CFG["vqc_lr"])
        tag = f"VQC {'sym' if sym else 'gen'} {enc}"
        f = lambda s, m=m: m.probs(s)
        r = fit_scale_and_score(chain, f, T, tag)
        ke = learning_error(chain, f, h, T)
        print(
            f"  {tag:<22} params {m.n_params():>4}  var {r['var']:.3e}"
            f"  VRF {var_naive/r['var']:>11,.2f}x  KL {ke:.4f}"
            f"  ({time.time()-t0:.0f}s)"
        )
        results.append(
            dict(
                method=tag,
                var=r["var"],
                params=m.n_params(),
                klerr=ke,
                kind="quantum",
            )
        )

    print("\nSTEP 4 - HYBRID DRESSED CIRCUIT")
    print("  statewise-regressive architecture with a quantum middle.")
    print(RULE)

    X_by_s = {
        s: np.concatenate(
            [chain.bits.astype(float), np.full((chain.S, 1), s / T)],
            1,
        )
        for s in range(1, T)
    }
    t0 = time.time()
    hyb = DressedCircuit(
        n + 1,
        n_q=CFG["hyb_qubits"],
        L=CFG["hyb_layers"],
        scale=CFG["hyb_scale"],
        seed=CFG["hyb_seed"],
    )
    hyb.fit(X_by_s, tg, steps=CFG["hyb_steps"], lr=CFG["hyb_lr"])
    f_hyb = lambda s: hyb.probs(X_by_s[s])
    r = fit_scale_and_score(chain, f_hyb, T, "hybrid dressed circuit")
    ke = learning_error(chain, f_hyb, h, T)
    print(
        f"  n_q={CFG['hyb_qubits']} L={CFG['hyb_layers']}  "
        f"params {hyb.n_params():>4}  var {r['var']:.3e}  "
        f"VRF {var_naive/r['var']:>11,.2f}x  KL {ke:.4f}"
        f"  ({time.time()-t0:.0f}s)"
    )
    results.append(
        dict(
            method="hybrid dressed circuit",
            var=r["var"],
            params=hyb.n_params(),
            klerr=ke,
            kind="hybrid",
        )
    )

    print("\nSTEP 5 - CLASSICAL MLP CONTROL")
    print("  same state/time inputs, exact targets, and forward-KL objective as")
    print("  the hybrid model, with approximately matched parameter count.")
    print("  Optimizer budgets are class-specific.")
    print(RULE)

    hidden = max(2, int(round((hyb.n_params() - 1) / (n + 2))))
    mlp = MLP(n, hidden, seed=CFG["mlp_seed"])
    mlp.fit(
        chain.bits.astype(float),
        tg,
        T,
        steps=CFG["mlp_steps"],
        lr=CFG["mlp_lr"],
    )
    f_mlp = lambda s: mlp.probs(chain.bits.astype(float), s, T)
    r = fit_scale_and_score(chain, f_mlp, T, "classical MLP")
    ke = learning_error(chain, f_mlp, h, T)
    print(
        f"  hidden={hidden}  params {mlp.n_params():>4}  var {r['var']:.3e}"
        f"  VRF {var_naive/r['var']:>11,.2f}x  KL {ke:.4f}"
    )
    results.append(
        dict(
            method="classical MLP",
            var=r["var"],
            params=mlp.n_params(),
            klerr=ke,
            kind="classical",
        )
    )

    print("\n" + "-" * 88)
    print(f"SUMMARY - {name}   (VRF vs naive MC; normVRF = VRF * p_T)")
    print("-" * 88)
    print(
        f"{'method':<38}{'kind':<11}{'params':>7}"
        f"{'VRF':>14}{'normVRF':>10}{'KL to h':>10}"
    )
    print(RULE)
    print(
        f"{'exact h-transform (ceiling)':<38}{'exact':<11}{'--':>7}"
        f"{'infinite':>14}{'--':>10}{'0.0000':>10}"
    )
    for rr in sorted(results, key=lambda d: d["var"]):
        vrf = var_naive / rr["var"]
        kl = "---" if np.isnan(rr["klerr"]) else f"{rr['klerr']:.4f}"
        print(
            f"{rr['method']:<38}{rr['kind']:<11}{rr['params']:>7}"
            f"{vrf:>14,.2f}{vrf*p:>10.3f}{kl:>10}"
        )
    print(RULE)
    print(
        f"naive MC variance {var_naive:.3e},  p_T {p:.4e},  "
        f"|F| {int(chain.in_F.sum())}"
    )

    return dict(
        name=name,
        p_T=p,
        var_naive=var_naive,
        distinct_values=dv,
        n_F=int(chain.in_F.sum()),
        results=results,
    )


def main():
    T, n = CFG["T"], CFG["n"]
    out = {}

    weighted = Chain(
        n=n,
        caps=np.array(CFG["weighted_caps"]),
        C_min=CFG["weighted_C_min"],
        a0=CFG["a0"],
        b0=CFG["b0"],
        gamma=CFG["gamma"],
        eta=CFG["eta"],
    )
    out["weighted"] = run_model(
        "WEIGHTED (mission-time reliability model)", weighted, T
    )

    kofn = Chain(
        n=n,
        k_min=CFG["kofn_k"],
        a0=CFG["a0"],
        b0=CFG["b0"],
        gamma=CFG["gamma"],
        eta=CFG["eta"],
    )
    out["kofn"] = run_model(
        "K-OUT-OF-N (symmetric control, not the applied model)", kofn, T
    )

    out_path = os.path.join(RESULTS, "results_main_boundaryfix.json")
    with open(out_path, "w") as f:
        json.dump(dict(config=CFG, models=out), f, indent=2, default=float)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
