"""
finish.py: repeated-seed regressive-model grid + gamma-retuned rarity sweep.
"""

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

T, n = CFG["T"], CFG["n"]
OUT = os.path.join(RESULTS, "results_final_boundaryfix.json")

if os.path.exists(OUT):
    with open(OUT) as f:
        res = json.load(f)
else:
    res = {"seeds": {}, "rarity": {}}


def save():
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2, default=float)


def build(kind, gamma=CFG["gamma"]):
    if kind == "WEIGHTED":
        return Chain(
            n=n,
            caps=np.array(CFG["weighted_caps"]),
            C_min=CFG["weighted_C_min"],
            a0=CFG["a0"],
            b0=CFG["b0"],
            gamma=gamma,
            eta=CFG["eta"],
        )
    return Chain(
        n=n,
        k_min=CFG["kofn_k"],
        a0=CFG["a0"],
        b0=CFG["b0"],
        gamma=gamma,
        eta=CFG["eta"],
    )


def design(ch):
    return {
        s: np.concatenate(
            [ch.bits.astype(float), np.full((ch.S, 1), s / T)],
            1,
        )
        for s in range(1, T)
    }


def run_mlp(ch, tg, h, hid, sd):
    m = MLP(n, hid, seed=sd)
    m.fit(
        ch.bits.astype(float),
        tg,
        T,
        steps=CFG["mlp_steps"],
        lr=CFG["mlp_lr"],
    )
    f = lambda s: m.probs(ch.bits.astype(float), s, T)
    p = float(h[T, ch.x0])
    vn = p * (1 - p)
    score = fit_scale_and_score(ch, f, T, "m")
    return (
        vn / score["var"],
        learning_error(ch, f, h, T),
        m.n_params(),
    )


def run_hyb(ch, tg, h, X, nq, L, sd):
    hy = DressedCircuit(
        n + 1,
        n_q=nq,
        L=L,
        scale=CFG["hyb_scale"],
        seed=sd,
    )
    hy.fit(X, tg, steps=CFG["hyb_steps"], lr=CFG["hyb_lr"])
    f = lambda s: hy.probs(X[s])
    p = float(h[T, ch.x0])
    vn = p * (1 - p)
    score = fit_scale_and_score(ch, f, T, "h")
    return (
        vn / score["var"],
        learning_error(ch, f, h, T),
        hy.n_params(),
    )


# ---------- (A) repeated-seed regressive grid ----------
NSEED = 8
GRID = [
    ("WEIGHTED", "MLP", 6),
    ("WEIGHTED", "MLP", 16),
    ("WEIGHTED", "HYB", (4, 2)),
    ("WEIGHTED", "HYB", (6, 2)),
    ("KOFN", "MLP", 6),
    ("KOFN", "MLP", 16),
    ("KOFN", "HYB", (4, 2)),
    ("KOFN", "HYB", (6, 2)),
]

print("(A) SEED STABILITY -- 8 seeds per configuration", flush=True)
for kind, fam, spec in GRID:
    key = f"{kind}|{fam}|{spec}"
    if key in res["seeds"]:
        print(f"  [cached] {key}", flush=True)
        continue

    ch = build(kind)
    h = ch.exact_h(T)
    tg = targets_from_h(ch, h, T)
    X = design(ch)

    vrfs, kls, npar = [], [], None
    for sd in range(NSEED):
        if fam == "MLP":
            v, k, npar = run_mlp(ch, tg, h, spec, sd)
        else:
            v, k, npar = run_hyb(ch, tg, h, X, spec[0], spec[1], sd)
        vrfs.append(v)
        kls.append(k)

    res["seeds"][key] = dict(
        params=npar,
        vrf=vrfs,
        kl=kls,
        p_T=float(h[T, ch.x0]),
    )
    a = np.array(vrfs)
    print(
        f"  {kind:<9}{fam:<4}{str(spec):<8} params {npar:>4}  "
        f"median {np.median(a):>9,.1f}  "
        f"IQR [{np.percentile(a,25):,.1f}, {np.percentile(a,75):,.1f}]  "
        f"range {a.min():,.1f}..{a.max():,.1f}",
        flush=True,
    )
    save()


# ---------- (B) gamma-retuned rarity regime ----------
print("\n(B) RARITY REGIME -- gamma retuned per target p_T (WEIGHTED)", flush=True)


def solve_gamma(target, lo=1e-4, hi=3.0):
    f = lambda g: build("WEIGHTED", g).p_T(T)
    if not (f(lo) <= target <= f(hi)):
        return None
    for _ in range(45):
        mid = (lo + hi) / 2
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


for tgt in (1e-2, 1e-3, 1e-4, 1e-6, 1e-8):
    key = f"{tgt:.0e}"
    if key in res["rarity"]:
        print(f"  [cached] p_T={key}", flush=True)
        continue

    g = solve_gamma(tgt)
    if g is None:
        res["rarity"][key] = dict(gamma=None, note="unreachable")
        save()
        print(f"  p_T={key:>8}  unreachable", flush=True)
        continue

    ch = build("WEIGHTED", g)
    h = ch.exact_h(T)
    tg = targets_from_h(ch, h, T)
    X = design(ch)

    mv, hv = [], []
    for sd in range(3):
        mv.append(run_mlp(ch, tg, h, 16, sd)[0])
        hv.append(run_hyb(ch, tg, h, X, 4, 2, sd)[0])

    res["rarity"][key] = dict(
        gamma=g,
        p_T_actual=float(h[T, ch.x0]),
        mlp=mv,
        hyb=hv,
    )
    print(
        f"  p_T={key:>8}  gamma={g:.4f}  "
        f"MLP median {np.median(mv):>9,.1f}   "
        f"hybrid median {np.median(hv):>9,.1f}",
        flush=True,
    )
    save()

print(f"\nDONE -- wrote {OUT}", flush=True)
