"""Does the study's conclusion survive the change of rate law?

Compares the four model classes under the two rate laws on weighted, T=6.
Born machines at one seed (the claim is only that they fail), MLP and hybrid
at three seeds (the claim is an ordering, which needs a median).
"""
import json, time, numpy as np
from qht import Chain, VQCProposal, MLP, exact_variance, tilt_family_optimum
from hybrid import DressedCircuit
from run_study import CFG, targets_from_h, fit_scale_and_score, learning_error

T = 6
CAPS = [8., 8., 8., 8., 4., 4., 4., 4.]
out = {}

for law in ["exponent", "cascade"]:
    ch = Chain(n=8, caps=CAPS, C_min=12.0, rate_law=law)
    h = ch.exact_h(T)
    p = float(h[T, ch.x0]); vn = p * (1 - p)
    tg = targets_from_h(ch, h, T)
    rec = {"p_T": p, "n_F": int(ch.in_F.sum())}

    dev = float(np.abs(ch.P.sum(1) - 1).max())
    vnr, _ = exact_variance(ch, {s: ch.P for s in range(1, T + 1)}, T)
    rec["gate_rowsum"] = dev
    rec["gate_naive_reldev"] = abs(vnr - vn) / vn

    v_tilt, nu = tilt_family_optimum(ch, T, grid=CFG["tilt_grid"])
    rec["tilt"] = {"vrf": vn / v_tilt, "nu": nu}

    X = {s: np.concatenate([ch.bits.astype(float), np.full((ch.S, 1), s / T)], 1)
         for s in range(1, T + 1)}

    for tag, seeds in [("hybrid_nq4", [0, 1, 2]), ("mlp_w16", [0, 1, 2])]:
        vals, kls = [], []
        for sd in seeds:
            t0 = time.time()
            if tag.startswith("hybrid"):
                m = DressedCircuit(9, n_q=4, L=2, scale=float(np.pi), seed=sd)
                m.fit(X, tg, steps=CFG["hyb_steps"], lr=CFG["hyb_lr"])
                f = lambda s, m=m: m.probs(X[max(s, 1)])
            else:
                m = MLP(8, 16, seed=sd)
                m.fit(ch.bits.astype(float), tg, T, steps=CFG["mlp_steps"], lr=CFG["mlp_lr"])
                f = lambda s, m=m: m.probs(ch.bits.astype(float), max(s, 1), T)
            r = fit_scale_and_score(ch, f, T, tag)
            vals.append(vn / r["var"]); kls.append(learning_error(ch, f, h, T))
            print(f"{law:9s} {tag:11s} seed {sd}  VRF {vals[-1]:10.2f}  "
                  f"KL {kls[-1]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        rec[tag] = {"vrf": vals, "median": float(np.median(vals)),
                    "kl_median": float(np.median(kls))}

    for enc, sym in [("static", True), ("static", False)]:
        t0 = time.time()
        m = VQCProposal(8, T, L=CFG["vqc_layers"], encoding=enc, seed=0, sym=sym)
        m.fit(tg, steps=CFG["vqc_steps"], lr=CFG["vqc_lr"])
        f = lambda s, m=m: m.probs(max(s, 1))
        r = fit_scale_and_score(ch, f, T, "vqc")
        tag = f"born_{'sym' if sym else 'gen'}_{enc}"
        rec[tag] = {"vrf": vn / r["var"], "kl": learning_error(ch, f, h, T),
                    "params": m.n_params()}
        print(f"{law:9s} {tag:17s}  VRF {rec[tag]['vrf']:8.2f}  "
              f"KL {rec[tag]['kl']:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    out[law] = rec
    print(json.dumps({law: rec}, indent=1), flush=True)

json.dump(out, open("/home/claude/qht/results/results_ratelaw.json", "w"), indent=1)
print("DONE")
