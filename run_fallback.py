"""
Point 1: rigorous full-K fallback.

If the direct-S early-stopping test never certifies S_h(t) < 1/2, the algorithm
must fall back on the worst-case theorem of Section 4.  Its proof needs each
endpoint-slope difference to be computed with error <= 1/64 (so that the inexact
contraction d_{j+1} <= 1/2 d_j + 1/32 holds) and the final d_K < 1/8 to be
certified.  In double precision this fails beyond n ~ 100 (the 1/(2 theta h)
amplification), which is exactly why we need a guaranteed version.

Here every endpoint-slope difference and every candidate comparison is a RIGOROUS
enclosure from the hybrid oracle, with adaptive precision: the eigenvalue
brackets are narrowed (Sturm bisection) until the enclosure of d is tight enough
to (a) decide the candidate comparison, or recognise it as a near-tie within
1/64, and (b) certify d_K < 1/8 at the end.  A near-tie is harmless: the
contraction lemma guarantees at least one candidate has potential <= 1/2 parent,
so picking either when their enclosures overlap keeps d_{j+1} <= 1/2 d_j + 1/32.

This fallback is never triggered on any benchmark instance (direct-S always
certifies first); it is run here on demand to show it is sound and implementable.
"""
import sys, time
import numpy as np
from flint import arb, ctx
from ke02num.core import make_matrix, K_theta
from ke02num.hybrid import HybridBackend

DELTA = 0.48


def d_enclosure(hb, t, h, theta, rho_bits):
    """Rigorous (mid, rad) of the endpoint-slope difference of [t-h, t+h],
    secants over length 2*theta*h, using eigenvalue brackets narrowed to rho_bits."""
    old = hb.refined_mode
    hb.refined_mode = True
    # temporarily set refinement target
    a, b = arb(t) - arb(h), arb(t) + arb(h)
    L = 2 * arb(theta) * arb(h)
    # F at the four abscissae with refinement to rho_bits
    def Fr(x):
        hb.refine(float(x.mid()) if hasattr(x, "mid") else float(x), rho_bits=rho_bits)
        return hb.F(x)
    Fa, Fa2, Fb2, Fb = hb.F(a), hb.F(a + L), hb.F(b - L), hb.F(b)
    hb.refined_mode = old
    val = ((Fb - Fb2) - (Fa2 - Fa)) / L
    return float(val.mid()), float(val.rad())


def endpoint_diff_adaptive(hb, t, h, theta, want_tol):
    """Narrow precision until the endpoint-slope-difference enclosure radius < want_tol
    (or precision budget exhausted). Returns (mid, rad, rho_bits_used)."""
    for rho_bits in (60, 90, 140, 220, 320, 500):
        # refine the four points to rho_bits
        hb.refined_mode = True
        a, b = arb(t) - arb(h), arb(t) + arb(h)
        Lsec = 2 * arb(theta) * arb(h)
        for x in (a, a + Lsec, b - Lsec, b):
            hb.refine(float(x.mid()), rho_bits=rho_bits)
        Fa, Fa2, Fb2, Fb = hb.F(a), hb.F(a + Lsec), hb.F(b - Lsec), hb.F(b)
        hb.refined_mode = False
        val = ((Fb - Fb2) - (Fa2 - Fa)) / Lsec
        if float(val.rad()) < want_tol:
            return float(val.mid()), float(val.rad()), rho_bits
    return float(val.mid()), float(val.rad()), rho_bits


def rigorous_full_K(fam, n, seed):
    d, b = make_matrix(fam, n, seed)
    K, theta = K_theta(n)
    hb = HybridBackend(d, b, DELTA)
    t = DELTA / (2 * n); h = DELTA / (2 * n)
    max_cmp_unc = 0.0
    n_ties = 0
    t0 = time.time()
    for j in range(K):
        h = (0.5 - theta) * h
        # two candidates, each of radius 2h' with the parent convention [t-2h,t] / [t,t+2h]
        hl = h
        dl_m, dl_r, _ = endpoint_diff_adaptive(hb, t - hl, hl, theta, want_tol=1 / 128)
        dr_m, dr_r, _ = endpoint_diff_adaptive(hb, t + hl, hl, theta, want_tol=1 / 128)
        max_cmp_unc = max(max_cmp_unc, dl_r, dr_r)
        # rigorous comparison: if enclosures are separated, pick the smaller;
        # otherwise it is a near-tie (|dl - dr| <= dl_r + dr_r <= 1/64), pick left.
        if dl_m + dl_r < dr_m - dr_r:
            t = t - hl
        elif dr_m + dr_r < dl_m - dl_r:
            t = t + hl
        else:
            n_ties += 1
            t = t - hl  # near-tie: either candidate keeps the contraction bound
    h = h / 2.0
    # certify d_K < 1/8 at the output, with enough precision
    dK_m, dK_r, rho = endpoint_diff_adaptive(hb, t, h * 2, theta, want_tol=1 / 64)
    # S_{h}(t) at the output radius (the actual certificate)
    hb.refined_mode = True
    for x in (arb(t) - arb(h), arb(t), arb(t) + arb(h)):
        hb.refine(float(x.mid()), rho_bits=max(rho, 90))
    Sval = (hb.F(arb(t) + arb(h)) + hb.F(arb(t) - arb(h)) - 2 * hb.F(arb(t))) / arb(h)
    hb.refined_mode = False
    S_m, S_r = float(Sval.mid()), float(Sval.rad())
    cert = h ** 2 / (2 ** 20 * n ** 4)
    gap_direct = hb.gap_direct_lower_refined(t)
    return dict(fam=fam, n=n, K=K, t=t, h=h, dK_m=dK_m, dK_r=dK_r,
                dK_upper=dK_m + dK_r, S_m=S_m, S_r=S_r, S_upper=S_m + S_r,
                cert=cert, gap_direct=gap_direct, max_cmp_unc=max_cmp_unc,
                n_ties=n_ties, wall=time.time() - t0)


if __name__ == "__main__":
    for (fam, n) in [("random", 50), ("wilkinson", 50), ("random", 100)]:
        r = rigorous_full_K(fam, n, 1)
        ok_cmp = r["max_cmp_unc"] <= 1 / 64
        ok_dK = r["dK_upper"] < 1 / 8
        ok_S = r["S_upper"] < 0.5
        print(f"{fam:10s} n={n:4d} K={r['K']}: full-K rigorous: "
              f"d_K <= {r['dK_upper']:.4f} (<1/8? {ok_dK}), "
              f"max endpoint-cmp uncertainty = {r['max_cmp_unc']:.2e} (<=1/64? {ok_cmp}), "
              f"ties={r['n_ties']}, S_h(t) <= {r['S_upper']:.4f} (<1/2? {ok_S}), "
              f"cert={r['cert']:.2e}, gap_direct={r['gap_direct']:.3e}, wall={r['wall']:.1f}s",
              flush=True)
