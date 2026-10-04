"""
Eight-seed Born-machine grid used to re-check horizon-encoding and symmetry
claims after fixing the g_0 / PER-s horizon indexing.
"""

import json
import os

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

from qht import Chain, VQCProposal
from run_study import CFG, fit_scale_and_score, learning_error, targets_from_h

T = CFG["T"]
n = CFG["n"]
OUT = os.path.join(RESULTS, "results_born_boundaryfix.json")


def build(kind):
    if kind == "WEIGHTED":
        return Chain(
            n=n,
            caps=np.array(CFG["weighted_caps"]),
            C_min=CFG["weighted_C_min"],
            a0=CFG["a0"],
            b0=CFG["b0"],
            gamma=CFG["gamma"],
            eta=CFG["eta"],
        )
    return Chain(
        n=n,
        k_min=CFG["kofn_k"],
        a0=CFG["a0"],
        b0=CFG["b0"],
        gamma=CFG["gamma"],
        eta=CFG["eta"],
    )


configs = [
    ("static", False),
    ("reupload", False),
    ("per_s", False),
    ("static", True),
    ("reupload", True),
    ("per_s", True),
]

out = {}

for structure in ("WEIGHTED", "KOFN"):
    ch = build(structure)
    h = ch.exact_h(T)
    tg = targets_from_h(ch, h, T)
    p = float(h[T, ch.x0])
    var_naive = p * (1.0 - p)

    out[structure] = {"p_T": p, "configs": {}}
    print(f"\n=== {structure}  p_T={p:.6e} ===", flush=True)

    for enc, sym in configs:
        key = f"{enc}|{'sym' if sym else 'generic'}"
        vrfs = []
        kls = []
        params = None

        for seed in range(8):
            m = VQCProposal(
                n,
                T,
                L=CFG["vqc_layers"],
                encoding=enc,
                seed=seed,
                sym=sym,
            )
            m.fit(tg, steps=CFG["vqc_steps"], lr=CFG["vqc_lr"])
            f = lambda s, m=m: m.probs(s)
            score = fit_scale_and_score(ch, f, T, key)
            vrfs.append(var_naive / score["var"])
            kls.append(learning_error(ch, f, h, T))
            params = m.n_params()

            print(
                f"  {key:<22} seed={seed} params={params:>4} "
                f"VRF={vrfs[-1]:>11,.4g} KL={kls[-1]:.6f}",
                flush=True,
            )

        out[structure]["configs"][key] = {
            "params": params,
            "vrf": vrfs,
            "kl": kls,
            "median_vrf": float(np.median(vrfs)),
            "median_kl": float(np.median(kls)),
        }

        print(
            f"  -> {key:<19} median VRF={np.median(vrfs):.4g}, "
            f"median KL={np.median(kls):.6f}",
            flush=True,
        )

with open(OUT, "w") as f:
    json.dump(out, f, indent=2, default=float)

print(f"\nwrote {OUT}", flush=True)
