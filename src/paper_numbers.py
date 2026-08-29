#!/usr/bin/env python3
"""paper_numbers.py: emit every figure quoted in the paper, read from the
result JSONs. 

Usage:  python3 paper_numbers.py
"""
import json, os, numpy as np
import qht

_R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
L = lambda f: json.load(open(os.path.join(_R, f)))
main, final, sweep, noise = L("results_main.json"), L("results_final.json"), \
                            L("results_sweep.json"), L("results_noise.json")
CFG = main["config"]
T, n = CFG["T"], CFG["n"]
RULE = "-" * 78


def chain(kind, gamma=CFG["gamma"]):
    if kind == "WEIGHTED":
        return qht.Chain(n=n, caps=np.array(CFG["weighted_caps"]),
                         C_min=CFG["weighted_C_min"], a0=CFG["a0"], b0=CFG["b0"],
                         gamma=gamma, eta=CFG["eta"])
    return qht.Chain(n=n, k_min=CFG["kofn_k"], a0=CFG["a0"], b0=CFG["b0"],
                     gamma=gamma, eta=CFG["eta"])


print("SECTION 3  PROBLEM SETUP"); print(RULE)
for kind, tag in (("WEIGHTED", "weighted"), ("KOFN", "kofn")):
    ch = chain(kind); h = ch.exact_h(T); nf = ~ch.in_F
    dv_sT = len(set(np.round(h[T][nf], 12).tolist()))
    dv_all = len(set(np.concatenate([np.round(h[s][nf], 12) for s in range(1, T + 1)]).tolist()))
    print(f"  {tag:<9} p_T = {ch.p_T(T):.4e}   |F| = {int(ch.in_F.sum())} of {ch.S}"
          f"   non-F = {int(nf.sum())}")
    print(f"  {'':<9} distinct h_s values off F: {dv_sT} at s=T, {dv_all} over s=1..T")
print(f"  correctness gates: rowsum {main['models']['weighted']['results'] and ''}")
for tag in ("weighted", "kofn"):
    m = main["models"][tag]
    print(f"  {tag:<9} gate distinct_values (from run) = {m['distinct_values']}")

print("\nTABLE 1  SINGLE CONFIGURATION, SINGLE SEED"); print(RULE)
for tag in ("weighted", "kofn"):
    m = main["models"][tag]
    print(f"  {tag.upper()},  p_T = {m['p_T']:.4e}")
    for r in sorted(m["results"], key=lambda d: d["var"]):
        kl = "" if r["klerr"] != r["klerr"] else f"{r['klerr']:.4f}"
        print(f"    {r['method']:<40}{r['kind']:<10}{r['params']:>5}"
              f"{m['var_naive']/r['var']:>12,.2f}{kl:>10}")

print("\nSECTION 5.2  CLOCK ENCODINGS (KL to h_s)"); print(RULE)
for tag in ("weighted", "kofn"):
    d = {r["method"]: r["klerr"] for r in main["models"][tag]["results"]}
    for sec in ("sym", "gen"):
        row = {e: d.get(f"VQC {sec} {e}") for e in ("static", "per_s", "reupload")}
        print(f"  {tag:<9} {sec:<4} " + "  ".join(
            f"{e}={v:.4f}" for e, v in row.items() if v is not None))

print("\nTABLE 2  EIGHT SEEDS PER CONFIGURATION"); print(RULE)
print(f"  {'structure':<10}{'config':<16}{'par':>5}{'median':>10}"
      f"{'IQR':>22}{'range':>22}{'normVRF':>9}")
for key, v in final["seeds"].items():
    kind, fam, spec = key.split("|")
    a = np.array(v["vrf"]); med = float(np.median(a))
    cfg = ("MLP width " + spec) if fam == "MLP" else ("hybrid n_q = " + spec[1])
    print(f"  {kind:<10}{cfg:<16}{v['params']:>5}{med:>10,.1f}"
          f"{f'[{np.percentile(a,25):,.1f}, {np.percentile(a,75):,.1f}]':>22}"
          f"{f'{a.min():,.1f} to {a.max():,.1f}':>22}{med*v['p_T']:>9.3f}")
    print(f"  {'':<10}{'':<16}spread max/min = {a.max()/max(a.min(),1e-12):,.1f}x")

print("\nTABLE 3  RARITY SWEEP (medians over 3 seeds)"); print(RULE)
print(f"  {'target':>8}{'gamma':>9}{'p_T':>12}{'MLP':>12}{'hybrid':>12}"
      f"{'ratio':>9}{'MLPnorm':>10}{'hybnorm':>10}")
ratios = []
for k, v in final["rarity"].items():
    if v.get("gamma") is None:
        print(f"  {k:>8}  unreachable"); continue
    mm, hh = float(np.median(v["mlp"])), float(np.median(v["hyb"]))
    p = v["p_T_actual"]; ratios.append((k, mm / hh))
    print(f"  {k:>8}{v['gamma']:>9.4f}{p:>12.3e}{mm:>12,.1f}{hh:>12,.1f}"
          f"{mm/hh:>9.1f}{mm*p:>10.4f}{hh*p:>10.5f}")
if ratios:
    print(f"  ratio path: " + " -> ".join(f"{r:.1f}" for _, r in ratios))
    print(f"  monotone in rarity? {all(b>=a for (_,a),(_,b) in zip(ratios,ratios[1:]))}")
    fst, lst = ratios[0][1], ratios[-1][1]
    print(f"  growth across the sweep: {fst:.1f}x -> {lst:.1f}x  ({lst/fst:.0f}x)")
    ks = [k for k, _ in ratios]
    nm = [float(np.median(final['rarity'][k]['mlp']))*final['rarity'][k]['p_T_actual'] for k in ks]
    nh = [float(np.median(final['rarity'][k]['hyb']))*final['rarity'][k]['p_T_actual'] for k in ks]
    print(f"  normVRF decay: MLP {nm[0]/nm[-1]:,.0f}x   hybrid {nh[0]/nh[-1]:,.0f}x")

print("\nTABLE 4  FINITE-SHOT READOUT"); print(RULE)
print(f"  classical MLP, exact readout, median: {noise['mlp_exact_vrf']:,.1f}")
print(f"  {'shots':>8}" + "".join(f"{f'lam={l}':>22}" for l in sorted({r['lam'] for r in noise['rows']})))
lams = sorted({r["lam"] for r in noise["rows"]})
shots = [r["shots"] for r in noise["rows"] if r["lam"] == lams[0]]
for sh in shots:
    lab = "exact" if sh is None else f"{sh:,}"
    cells = []
    for lam in lams:
        r = next(x for x in noise["rows"] if x["lam"] == lam and x["shots"] == sh)
        cells.append(f"{r['hybrid_med']:>10,.2f}{r['born_med']:>12,.2f}")
    print(f"  {lab:>8}" + "".join(cells))
hyb0 = [r["hybrid_med"] for r in noise["rows"] if r["lam"] == 0.0]
print(f"  hybrid at lambda=0 spans {min(hyb0):,.2f} to {max(hyb0):,.2f}")
born0 = [(r["shots"], r["born_med"]) for r in noise["rows"] if r["lam"] == 0.0]
print(f"  Born at lambda=0: " + ", ".join(
    f"{'exact' if s is None else format(s,',')}={v:,.2f}" for s, v in born0))
worst = max(lams)
r = next(x for x in noise["rows"] if x["lam"] == worst and x["shots"] == min(s for s in shots if s))
print(f"  hybrid at lambda={worst}, {min(s for s in shots if s):,} shots: {r['hybrid_med']:,.2f}")

print("\nAPPENDIX B  CAPACITY SWEEP (best of two seeds)"); print(RULE)
for kind in sweep:
    best_m = {}; best_h = {}
    for _, hid, sd, par, vrf, kl in sweep[kind]["mlp"]:
        best_m[par] = max(best_m.get(par, 0), vrf)
    for _, spec, sd, par, vrf, kl in sweep[kind]["hyb"]:
        best_h[par] = max(best_h.get(par, 0), vrf)
    print(f"  {kind}   MLP  " + "  ".join(f"{p}p:{v:,.1f}" for p, v in sorted(best_m.items())))
    print(f"  {'':<{len(kind)}}   HYB  " + "  ".join(f"{p}p:{v:,.1f}" for p, v in sorted(best_h.items())))
    if kind == "WEIGHTED":
        pm = min(best_m); ph = min(best_h)
        print(f"  smallest matched budgets: MLP {pm} params -> {best_m[pm]:,.1f},"
              f"  hybrid {ph} params -> {best_h[ph]:,.1f}")
