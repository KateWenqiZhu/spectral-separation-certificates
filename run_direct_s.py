"""
Experiments A-D of the second round:
  A. direct S_{h_j}(t_j) < 1/2 stopping vs endpoint-potential d_j < 1/8 stopping
     (double precision, model-based error bounds) over all families, n up to 1000;
  B/C. rigorous runs with the hybrid O(n^2) tridiagonal oracle (double eigenpairs +
     rigorous residuals + arb Sturm counts), n = 100 ... 1000, stopping at the
     first rigorously certified S < 1/2;
  D. two certificates at the selected parameter: slope certificate h^2/(2^20 n^4)
     versus the direct rigorous eigenvalue-enclosure gap bound.
Output: data/directS_fp.jsonl, data/directS_hyb.jsonl
"""
import json, time, pathlib, sys
import numpy as np
from ke02num.core import (FAMILIES, DETERMINISTIC, make_matrix, FPBackend, run_algorithm, state_at, K_theta)
from ke02num.hybrid import HybridBackend

DELTA = 0.48
OUT = pathlib.Path(__file__).parent / "data"


def cert(h, n): return h ** 2 / (2.0 ** 20 * n ** 4)


def summarize(res, be):
    n = res.n
    rec = dict(family=res.family, n=n, seed=res.seed, K=res.K, backend=be.name,
               stop_j_S=res.stop_j_S, stop_j_S_half=res.stop_j_S_half, stop_j_cert8=res.stop_j_cert_eighth,
               undecidable_j=res.undecidable_j, gap_before=res.gap_before, evals=be.evals, wall=res.wall)
    it = res.iters
    rec["iters"] = [dict(j=x.j, h=x.h, t=x.t, d=x.d_hat, d_err=x.d_err, S=x.S_full, S_err=x.S_full_err,
                         gap_direct=x.gap_direct) for x in it]
    if res.stop_j_S is not None:
        st = state_at(res, res.stop_j_S)
        rec["S_t"] = st.t; rec["S_h"] = st.h; rec["S_val"] = st.S_full; rec["S_err"] = st.S_full_err
        rec["S_cert"] = cert(st.h, n)
        g = be.gap(st.t)
        rec["S_gap"] = float(g.mid()) if hasattr(g, "mid") else float(g)
        rec["S_gap_direct"] = st.gap_direct
    if res.stop_j_cert_eighth is not None:
        st = state_at(res, res.stop_j_cert_eighth)
        rec["c8_t"] = st.t; rec["c8_h"] = st.h; rec["c8_cert"] = cert(st.h / 2, n)
        g = be.gap(st.t)
        rec["c8_gap"] = float(g.mid()) if hasattr(g, "mid") else float(g)
        rec["c8_gap_direct"] = st.gap_direct
    return rec


def part_A():
    cases = []
    for fam in FAMILIES:
        for n in [10, 20, 50, 100, 200, 500, 1000]:
            seeds = [1] if fam in DETERMINISTIC else (list(range(1, 31)) if n <= 200 else list(range(1, 11)))
            for s in seeds:
                cases.append((fam, n, s))
    t0 = time.time()
    with open(OUT / "directS_fp.jsonl", "w") as f:
        for k, (fam, n, seed) in enumerate(cases):
            d, b = make_matrix(fam, n, seed)
            be = FPBackend(d, b, DELTA, C_fp=1.0)
            res = run_algorithm(be, fam, n, seed, DELTA, stop_when_undecidable=True, record_S=False,
                                record_S_full=True, record_gap_direct=True)
            f.write(json.dumps(summarize(res, be)) + "\n")
            if (k + 1) % 50 == 0:
                print(f"A: {k+1}/{len(cases)} [{time.time()-t0:.0f}s]", flush=True)


def part_BCD(sizes, families=FAMILIES, seed=1):
    with open(OUT / "directS_hyb.jsonl", "a") as f:
        for n in sizes:
            for fam in families:
                d, b = make_matrix(fam, n, seed)
                hb = HybridBackend(d, b, DELTA)
                t0 = time.time()
                res = run_algorithm(hb, fam, n, seed, DELTA, stop_when_undecidable=False, record_S=False,
                                    record_S_full=True, record_gap_direct=True, stop_on="S")
                rec = summarize(res, hb)
                rec["wall_total"] = time.time() - t0
                rec["wall_eig"] = hb.wall
                rec["n_sturm_counts"] = hb.n_counts
                rec["max_cluster"] = hb.max_cluster
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"BCD: {fam:17s} n={n:5d} K={res.K} j_S={res.stop_j_S} evals={hb.evals} "
                      f"wall={rec['wall_total']:.1f}s S={rec.get('S_val', float('nan')):.3g}+-{rec.get('S_err', float('nan')):.1e} "
                      f"cert={rec.get('S_cert', float('nan')):.2e} gap={rec.get('S_gap', float('nan')):.2e} "
                      f"gap_direct={rec.get('S_gap_direct', float('nan')):.2e} j_c8={res.stop_j_cert_eighth}", flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "A":
        part_A()
    else:
        sizes = [int(x) for x in sys.argv[2:]]
        part_BCD(sizes)
