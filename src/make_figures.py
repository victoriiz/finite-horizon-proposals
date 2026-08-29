import json
import numpy as np
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_ROOT, "results")
FIGURES = os.path.join(_ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIGURES, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#111111", "#5b5b5b", "#dddddd"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "figure.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False,
})

fin = json.load(open(os.path.join(RESULTS,"results_final.json")))
swp = json.load(open(os.path.join(RESULTS,"results_sweep.json")))


# ---------------------------------------------------------------- Figure 1
def fig_rarity():
    ks = sorted(fin["rarity"], key=lambda k: float(k))
    p  = np.array([fin["rarity"][k]["p_T_actual"] for k in ks])
    m  = np.array([np.median(fin["rarity"][k]["mlp"]) for k in ks])
    h  = np.array([np.median(fin["rarity"][k]["hyb"]) for k in ks])

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))

    a = ax[0]
    a.loglog(p, m, "-o", color=BLUE,   lw=2, ms=5, label="classical MLP", zorder=3)
    a.loglog(p, h, "-s", color=ORANGE, lw=2, ms=5, label="hybrid circuit", zorder=3)
    # p is sorted ASCENDING, so index 0 is the rarest point and, after the axis
    # inversion, sits at the right edge. Label there, right-aligned.
    a.annotate("classical MLP", (p[0], m[0]), textcoords="offset points",
               xytext=(-6, 5), color=BLUE, fontsize=8, ha="right")
    a.annotate("hybrid", (p[0], h[0]), textcoords="offset points",
               xytext=(-6, -12), color=ORANGE, fontsize=8, ha="right")
    a.set_xlabel(r"failure probability $p_T$")
    a.set_ylabel("variance reduction factor")
    a.set_title("(a) variance reduction attained", fontsize=9, loc="left", pad=8)
    a.invert_xaxis()
    a.grid(True, which="major", color=GRID, lw=0.5, zorder=0)
    a.set_axisbelow(True)
    a.legend(frameon=False, fontsize=8, loc="upper left")
    a.set_ylim(5, 5e6)

    b = ax[1]
    b.loglog(p, m / h, "-o", color=INK, lw=2, ms=5, zorder=3)
    for xi, yi in zip(p, m / h):
        b.annotate(f"{yi:.0f}×", (xi, yi), textcoords="offset points",
                   xytext=(0, 7), ha="center", fontsize=7.5, color=INK)
    b.set_xlabel(r"failure probability $p_T$")
    b.set_ylabel("classical / hybrid")
    b.set_title("(b) the classical margin widens with rarity", fontsize=9,
                loc="left", pad=8)
    b.invert_xaxis()
    b.set_ylim(1.2, 900)
    b.grid(True, which="major", color=GRID, lw=0.5, zorder=0)
    b.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES,"fig_rarity.pdf"), bbox_inches="tight")
    print(f"wrote {FIGURES}/fig_rarity.pdf")


# ---------------------------------------------------------------- Figure 2
def fig_seeds():
    order = [("WEIGHTED", "MLP", "6"), ("WEIGHTED", "MLP", "16"),
             ("WEIGHTED", "HYB", "(4, 2)"), ("WEIGHTED", "HYB", "(6, 2)"),
             ("KOFN", "MLP", "6"), ("KOFN", "MLP", "16"),
             ("KOFN", "HYB", "(4, 2)"), ("KOFN", "HYB", "(6, 2)")]
    labels = ["MLP\nh=6", "MLP\nh=16", "hybrid\n$n_q$=4", "hybrid\n$n_q$=6"] * 2
    data, cols = [], []
    for kind, fam, spec in order:
        data.append(fin["seeds"][f"{kind}|{fam}|{spec}"]["vrf"])
        cols.append(BLUE if fam == "MLP" else ORANGE)

    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    pos = [1, 2, 3, 4, 5.8, 6.8, 7.8, 8.8]
    bp = ax.boxplot(data, positions=pos, widths=0.55, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.6), whis=(0, 100),
                    flierprops=dict(marker="", ls="none"),
                    capprops=dict(color=MUTED, lw=0.7),
                    whiskerprops=dict(color=MUTED, lw=0.7))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.30)
        patch.set_edgecolor(c); patch.set_linewidth(1.1)
    rng = np.random.default_rng(0)
    for x, d, c in zip(pos, data, cols):
        ax.scatter(x + rng.uniform(-0.13, 0.13, len(d)), d, s=11, color=c,
                   edgecolor="white", linewidth=0.4, zorder=4)

    ax.set_yscale("log")
    ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("variance reduction factor")
    ax.axhline(1.0, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.annotate("naive Monte Carlo", (9.35, 1.0), fontsize=7.5, color=MUTED,
                va="center", annotation_clip=False)
    ax.text(2.5, ax.get_ylim()[1] * 0.55, "WEIGHTED\n(mission-time model)",
            ha="center", fontsize=8, color=MUTED)
    ax.text(7.3, ax.get_ylim()[1] * 0.55, "K-OUT-OF-N\n(symmetric control)",
            ha="center", fontsize=8, color=MUTED)
    ax.axvline(4.9, color=GRID, lw=0.8)
    h1 = plt.Line2D([], [], color=BLUE, lw=6, alpha=0.5, label="classical MLP")
    h2 = plt.Line2D([], [], color=ORANGE, lw=6, alpha=0.5, label="hybrid circuit")
    ax.legend(handles=[h1, h2], frameon=False, fontsize=8, loc="lower left", ncol=2)
    ax.grid(True, axis="y", which="major", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES,"fig_seeds.pdf"), bbox_inches="tight")
    print(f"wrote {FIGURES}/fig_seeds.pdf")


# ---------------------------------------------------------------- Figure 3
def fig_capacity():
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for a, key, ttl in ((ax[0], "WEIGHTED", "(a) WEIGHTED (mission-time model)"),
                        (ax[1], "KOFN", "(b) K-OUT-OF-N (control)")):
        d = swp[key]
        for fam, col, mk, lab in (("mlp", BLUE, "o", "classical MLP"),
                                  ("hyb", ORANGE, "s", "hybrid circuit")):
            rows = d[fam]
            groups = {}
            for r in rows:
                groups.setdefault(r[3], []).append(r[4])
            xs = sorted(groups)
            ys = [max(groups[x]) for x in xs]
            # MARKERS ONLY, no connecting line: with 2 seeds per point the
            # apparent trend is not distinguishable from seed noise (Fig. 2).
            a.plot(xs, ys, ls="none", marker=mk, color=col, ms=6,
                   markeredgecolor="white", markeredgewidth=0.5,
                   label=lab, zorder=3)
        a.set_yscale("log")
        a.set_xlabel("trainable parameters")
        a.set_title(ttl, fontsize=9, loc="left", pad=8)
        a.grid(True, which="major", color=GRID, lw=0.5, zorder=0)
        a.set_axisbelow(True)
    ax[0].set_ylabel("variance reduction factor\n(best of 2 seeds)")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES,"fig_capacity.pdf"), bbox_inches="tight")
    print(f"wrote {FIGURES}/fig_capacity.pdf")


if __name__ == "__main__":
    fig_rarity(); fig_seeds(); fig_capacity()