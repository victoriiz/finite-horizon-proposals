"""Capacity sweep with corrected horizon indexing."""

import json
import os

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_ROOT, "results")
FIGURES = os.path.join(_ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

from qht import Chain, MLP
from hybrid import DressedCircuit
from run_study import CFG, fit_scale_and_score, learning_error, targets_from_h

T = CFG["T"]
n = CFG["n"]

models = {
    "WEIGHTED": Chain(
        n=n,
        caps=np.array(CFG["weighted_caps"]),
        C_min=CFG["weighted_C_min"],
        a0=CFG["a0"],
        b0=CFG["b0"],
        gamma=CFG["gamma"],
        eta=CFG["eta"],
    ),
    "KOFN": Chain(
        n=n,
        k_min=CFG["kofn_k"],
        a0=CFG["a0"],
        b0=CFG["b0"],
        gamma=CFG["gamma"],
        eta=CFG["eta"],
    ),
}

out = {}
for nm, ch in models.items():
    h = ch.exact_h(T)
    p = float(h[T, ch.x0])
    vn = p * (1 - p)
    tg = targets_from_h(ch, h, T)
    X = {
        s: np.concatenate(
            [ch.bits.astype(float), np.full((ch.S, 1), s / T)],
            1,
        )
        for s in range(1, T)
    }

    print(f"\n=== {nm}   p_T={p:.4e} ===", flush=True)
    print(f"{'model':<24}{'params':>7}{'VRF':>13}{'KL':>10}", flush=True)

    rows = []
    for hid in (4, 6, 8, 12, 16, 24):
        for sd in (0, 1):
            m = MLP(n, hid, seed=sd)
            m.fit(
                ch.bits.astype(float),
                tg,
                T,
                steps=CFG["mlp_steps"],
                lr=CFG["mlp_lr"],
            )
            f = lambda s, m=m: m.probs(ch.bits.astype(float), s, T)
            r = fit_scale_and_score(ch, f, T, "mlp")
            ke = learning_error(ch, f, h, T)
            rows.append(["MLP", hid, sd, m.n_params(), vn / r["var"], ke])

        best = max([r for r in rows if r[1] == hid], key=lambda r: r[4])
        print(
            f"{'MLP hidden='+str(hid):<24}{best[3]:>7}"
            f"{best[4]:>13,.2f}{best[5]:>10.4f}",
            flush=True,
        )

    hrows = []
    for nq, L in ((3, 2), (4, 2), (4, 3), (6, 2)):
        for sd in (0, 1):
            hy = DressedCircuit(
                n + 1,
                n_q=nq,
                L=L,
                scale=CFG["hyb_scale"],
                seed=sd,
            )
            hy.fit(X, tg, steps=CFG["hyb_steps"], lr=CFG["hyb_lr"])
            f = lambda s, hy=hy: hy.probs(X[s])
            r = fit_scale_and_score(ch, f, T, "hyb")
            ke = learning_error(ch, f, h, T)
            hrows.append(
                ["HYB", f"nq{nq}L{L}", sd, hy.n_params(), vn / r["var"], ke]
            )

        best = max(
            [r for r in hrows if r[1] == f"nq{nq}L{L}"],
            key=lambda r: r[4],
        )
        print(
            f"{'hybrid nq'+str(nq)+'L'+str(L):<24}{best[3]:>7}"
            f"{best[4]:>13,.2f}{best[5]:>10.4f}",
            flush=True,
        )

    bm = max(rows, key=lambda r: r[4])
    bh = max(hrows, key=lambda r: r[4])
    print(
        f"  BEST MLP    hidden={bm[1]} seed={bm[2]} params={bm[3]} "
        f"VRF={bm[4]:,.2f} KL={bm[5]:.4f}",
        flush=True,
    )
    print(
        f"  BEST HYBRID {bh[1]} seed={bh[2]} params={bh[3]} "
        f"VRF={bh[4]:,.2f} KL={bh[5]:.4f}",
        flush=True,
    )

    out[nm] = dict(mlp=rows, hyb=hrows, p_T=p, var_naive=vn)

out_path = os.path.join(RESULTS, "results_sweep_boundaryfix.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=float)
print(f"\nDONE -- wrote {out_path}", flush=True)
