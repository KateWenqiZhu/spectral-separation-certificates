# Spectral separation certificates for tridiagonal matrices

Numerical experiments and a rigorous, structure-exploiting certification
pipeline accompanying the paper

> **Convexity Certificates for Spectral Separation of Tridiagonal Matrices**

which resolves problem **KE-02** of the *Open Problems in Numerical Linear
Algebra* catalogue (deterministic "Minami-type" perturbation; Amsel et al.,
Problem 3.2). Given a Hermitian tridiagonal matrix `T` with `‖T‖₂ ≤ 1` and a
budget `0 < δ < 1/2`, the method constructs a diagonal perturbation
`D = t·diag(1,…,n)` with `‖D‖₂ ≤ δ` that provably separates the whole spectrum:

```
gap(T + D) > 2^(-36) · δ² / n^12
```

in `O(n · polylog(n/δ))` exact arithmetic operations. The construction rests on
the centered **slope variation** of the convex spectral function
`F(t) = Σ_j j·λ_j(T + tX)`: a window with `S_h(t) < 1/2` certifies a
quantitative gap, and `O(log n)` interval refinements locate one.

This repository contains everything needed to reproduce the numerical sections:
the algorithm, three spectral-function oracles (double precision, dense
arbitrary precision, and a structure-exploiting rigorous `O(n²)` tridiagonal
oracle), the experiment drivers, the raw data, and the figure/table generators.

## What is here

| Path | Contents |
|------|----------|
| `ke02num/core.py` | Matrix families, `F(t)`, `S_h(t)`, Algorithm 1 (interval selection), the double-precision and dense-`arb` backends |
| `ke02num/hybrid.py` | Structure-exploiting **rigorous `O(n²)` tridiagonal oracle**: LAPACK `dstemr` eigenpairs + two-sided **Sturm bracketing on the exact matrix** + Newton/bisection refinement + a direct eigenvalue-enclosure gap certificate |
| `run_sweep.py` | Double-precision sweep over 6 matrix families (certificate sharpness, early stopping, baselines) |
| `run_arb.py` | Dense arbitrary-precision rigorous runs of all `K` refinements |
| `run_adaptive.py` | Adaptive-precision pipeline: double-precision window selection + rigorous `S_h(t) < 1/2` stopping + direct gap certificate |
| `run_direct_s.py` | Direct-`S` vs endpoint-potential stopping comparison |
| `run_fallback.py` | Rigorous full-`K` fallback (guaranteed endpoint comparisons meeting the theorem's 1/64 tolerance) |
| `make_figures.py` | Regenerates `figures/` and `tables/` from `data/` |
| `data/` | All raw results (`*.jsonl`, `arb_*.json`, calibration) |
| `figures/`, `tables/` | Generated figures (PDF/PNG) and LaTeX tables |
| `docs/` | `RESULTS_INDEX.md` (file-by-file index), `AUDIT_round2.md` (rigor audit) |

## Matrix families

`random`, `nearly reducible`, `two weakly coupled blocks`, `Wilkinson Wₙ⁺`,
`tridiagonal Toeplitz` (Chebyshev Jacobi), and `diagonal with repeated
eigenvalues`, each scaled to `‖T‖₂ ≤ 1`, for `n ∈ {10,…,2000}`.

## Two kinds of certificate

For every selected matrix the pipeline reports both:

- **slope certificate** `h²/(2²⁰n⁴)` — the worst-case analytic lower bound from
  the theory (applies to *all* inputs);
- **direct certificate** `min_j (L_{j+1} − H_j)` from rigorous eigenvalue
  enclosures — the instance's actual certified gap, typically `~10²⁸` larger.

## Rigor

Every eigenvalue bracket is certified by **Sturm counts evaluated in `arb` ball
arithmetic on the exact matrix** (exact `d_i + t·i`, exact `β_i²`). The
double-precision residual only *seeds* the search; if it were ever too small the
Sturm check fails and widens, so the floating-point bound carries no rigor — it
only affects speed. Validated against `mpmath` and independent 400-bit Sturm
recounts (zero violations across all families, including exact multiplicities,
zero off-diagonal blocks, and the `3×10⁻⁵¹` Wilkinson pair). See
`docs/AUDIT_round2.md`.

## Reproduce

```bash
pip install -r requirements.txt

python run_sweep.py                 # double-precision sweep  -> data/sweep.jsonl
python run_arb.py 10 20 50 100      # dense rigorous full-K   -> data/arb_*.json
python run_adaptive.py 50 100 200 500 1000 2000   # adaptive rigorous pipeline
python run_direct_s.py A            # experiment A (direct-S vs 1/8 stopping)
python make_figures.py              # figures/ and tables/ from data/
```

`python-flint` (the `arb`/`acb` interval-arithmetic bindings) is required for
all rigorous runs. `run_arb.py` on large `n` is slow by design (dense
high-precision); the hybrid `O(n²)` oracle in `run_adaptive.py` certifies
`n = 1000` in tens of seconds.

## License

MIT (see `LICENSE`). If you use this, please cite the paper (`CITATION.cff`).
