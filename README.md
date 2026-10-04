# Learning the time-inhomogeneous h-transform

Variational quantum, hybrid classical-quantum, and classical proposals for
finite-horizon rare-event estimation, scored against **exact** ground truth.
h_s, p_T, the classical family optimum, and the
variance of every proposal are computed in closed form.

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


