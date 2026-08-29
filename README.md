# Learning the time-inhomogeneous h-transform

Variational quantum, hybrid classical-quantum, and classical proposals for
finite-horizon rare-event estimation, scored against **exact** ground truth.
h_s, p_T, the classical family optimum, and the
variance of every proposal are computed in closed form.

## Headline

The classical network wins, and its margin **grows with rarity** (2.2x at
p_T=1e-2, 185x at p_T=1e-8, which is the regime quantum rare-event methods are
pitched at). Run-to-run variance spans three orders of magnitude, so single-run
comparisons here carry no information.

## Files

    qht.py           chain, exact h_s, exact variance recursion,
                     Born-machine circuits, MLP, odds-ratio tilt family
    hybrid.py        dressed quantum circuit (Mari et al., Quantum 4:340, 2020):
                     batched simulator + parameter-shift gradients
    run_study.py     main experiment; CFG at the top holds EVERY hyperparameter
    sweep.py         capacity sweep (MLP width, hybrid size)
    finish.py        8-seed grid + rarity sweep; resumable, saves incrementally
    report.tex/pdf   full internal report
    results_*.json   all raw numbers

## Run

    python run_study.py     # main table, both models
    python sweep.py         # capacity sweep
    python finish.py        # seeds + rarity (resumable: re-run to continue)

Requires numpy only.


