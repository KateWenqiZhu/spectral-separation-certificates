"""
Adaptive-precision, rigorously certified early stopping (Experiments A-D, round 2).

Stage 1  double precision (dstemr) selects the windows (t_j, h_j) exactly as
         Algorithm 1 does (model-based decisions; the path itself needs no rigor).
Stage 2  cheap rigorous test at (t_j, h_j): S_{h_j}(t_j) enclosed with the hybrid
         O(n^2) oracle (double eigenpairs + rigorous residuals + Sturm counts);
         enclosure width ~ n u.  Certified if upper bound < 1/2.
Stage 3  if undecidable (lower bound < 1/2 <= upper bound), refine the three
         spectral-function values at t_j, t_j +- h_j by Newton + Sturm
         verification to width 2^-89 and re-test.
Output   first certified j, the certificate h_j^2/(2^20 n^4), the direct gap
         bound from the eigenvalue enclosures, and wall time per stage.

usage: python3 run_adaptive.py 100 200 500 1000 [--families ...] [--seeds 1 2]
"""
import json, sys, time, pathlib, argparse
import numpy as np
from ke02num.core import (FAMILIES, make_matrix, FPBackend, run_algorithm, state_at, K_theta, slope_variation)
from ke02num.hybrid import HybridBackend

DELTA = 0.48
OUT = pathlib.Path(__file__).parent / "data"


def cert(h, n): return h ** 2 / (2.0 ** 20 * n ** 4)


def adaptive(fam, n, seed, max_refined=12):
    d, b = make_matrix(fam, n, seed)
    K, theta = K_theta(n)
    # ---- stage 1: selection path in double precision (full K, no rigor needed)
    fp = FPBackend(d, b, DELTA)
    t0 = time.time()
    res = run_algorithm(fp, fam, n, seed, DELTA, stop_when_undecidable=False, record_S=False,
                        record_S_full=True, record_gap_direct=False)
    wall1 = time.time() - t0
    path = [(r.j, r.t, r.h, r.S_full, r.S_full_err, r.d_hat, r.d_err) for r in res.iters]
    out = dict(family=fam, n=n, seed=seed, K=K, wall_stage1=wall1, fp_evals=fp.evals,
               fp_stop_j_S=res.stop_j_S, fp_stop_j_cert8=res.stop_j_cert_eighth, fp_undecidable_j=res.undecidable_j,
               gap_before=res.gap_before, full_K_cert=cert(path[-1][2] / 2, n))
    # ---- stages 2/3
    hb = HybridBackend(d, b, DELTA)
    t0 = time.time()
    n_ref = 0; log = []
    certified = None
    for (j, t, h, Sfp, Sfp_err, dfp, dfp_err) in path:
        hb.refined_mode = False
        S2, e2 = slope_variation(hb, t, h)
        entry = dict(j=j, h=h, S_fp=Sfp, S2=S2, e2=e2)
        if S2 + e2 < 0.5:
            certified = ("stage2", j, t, h, S2, e2); log.append(entry); break
        if S2 - e2 < 0.5 and n_ref < max_refined:        # undecidable -> refine
            hb.refined_mode = True
            t1 = time.time()
            S3, e3 = slope_variation(hb, t, h)
            n_ref += 1
            entry.update(S3=S3, e3=e3, wall_refine=time.time() - t1)
            if S3 + e3 < 0.5:
                certified = ("stage3", j, t, h, S3, e3); log.append(entry); break
        log.append(entry)
    wall23 = time.time() - t0
    out.update(wall_stage23=wall23, wall_eig=hb.wall, wall_refine=hb.wall_refine, hyb_evals=hb.evals,
               n_refined_tests=n_ref, n_sturm_counts=hb.n_counts, max_cluster=hb.max_cluster, log=log)
    if certified:
        stage, j, t, h, S, e = certified
        hb.refined_mode = False
        gd_cheap = hb.gap_direct_lower(t)
        # refined direct gap bound (if refinement available at t)
        mids, w = hb.refine(t)
        gl = np.array([float(x.mid()) for x in mids])
        gd_ref = float(np.min(np.diff(gl) - (w[1:] + w[:-1])))
        out.update(certified_stage=stage, j_star=j, t_star=t, h_star=h, S_star=S, S_err=e,
                   slope_cert=cert(h, n), gap_direct_cheap=gd_cheap, gap_direct_refined=gd_ref,
                   gap_computed=fp.gap(t),
                   gain_vs_c8=(cert(h, n) / cert(state_at(res, res.stop_j_cert_eighth).h / 2, n)) if res.stop_j_cert_eighth is not None else None,
                   gain_vs_fullK=cert(h, n) / out["full_K_cert"])
    else:
        out.update(certified_stage=None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", nargs="+", type=int)
    ap.add_argument("--families", nargs="*", default=FAMILIES)
    ap.add_argument("--seeds", nargs="*", type=int, default=[1])
    a = ap.parse_args()
    with open(OUT / "adaptive.jsonl", "a") as f:
        for n in a.sizes:
            for fam in a.families:
                for seed in a.seeds:
                    r = adaptive(fam, n, seed)
                    f.write(json.dumps(r) + "\n"); f.flush()
                    if r["certified_stage"]:
                        print(f"{fam:17s} n={n:5d} K={r['K']:2d} j*={r['j_star']:2d} ({r['certified_stage']}) "
                              f"S={r['S_star']:.3f}+-{r['S_err']:.1e} cert={r['slope_cert']:.2e} "
                              f"gap_direct={r['gap_direct_refined']:.3e} (cheap {r['gap_direct_cheap']:.3e}) "
                              f"gain/c8={r['gain_vs_c8'] if r['gain_vs_c8'] is None else round(r['gain_vs_c8'])} gain/K={r['gain_vs_fullK']:.1e} "
                              f"wall: fp {r['wall_stage1']:.1f}s, rigorous {r['wall_stage23']:.1f}s (refine {r['wall_refine']:.1f}s, {r['n_refined_tests']} tests) "
                              f"j_c8(fp)={r['fp_stop_j_cert8']} und={r['fp_undecidable_j']}", flush=True)
                    else:
                        print(f"{fam:17s} n={n:5d} NOT certified; wall {r['wall_stage23']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
