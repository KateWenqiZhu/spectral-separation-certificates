"""
Experiment 1 (certificate sharpness), Experiment 2 (early stopping) and
Experiment 3 (baselines) in double precision, over six matrix families,
n in {10,20,50,100,200} and 30 seeds for the random families.

Output: data/sweep.jsonl  (one JSON record per (family, n, seed))
        data/fp_calibration.json
"""
import json, sys, time, pathlib
import numpy as np
from ke02num.core import (FAMILIES, DETERMINISTIC, make_matrix, FPBackend, ArbBackend,
                          run_algorithm, baselines, state_at, cert_gap, K_theta)

DELTA = 0.48
NS = [10, 20, 50, 100, 200]
SEEDS = 30
OUT = pathlib.Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)


def calibrate(n_points=12):
    """max |F_fp - F_arb| / eps_F over families and n (eps_F with C_fp = 1)."""
    rng = np.random.default_rng(2026)
    worst = {}
    for n in [10, 20, 50, 100]:
        for fam in FAMILIES:
            d, b = make_matrix(fam, n, 1)
            fp = FPBackend(d, b, DELTA, C_fp=1.0)
            ar = ArbBackend(d, b, DELTA, prec=128)
            w = 0.0
            for t in rng.uniform(0, DELTA / n, n_points):
                err = abs(fp.F(t) - float(ar.F(t).mid()))
                w = max(w, err / fp.eps_F())
            worst[f"{fam}:{n}"] = w
    return worst


def main():
    t_start = time.time()
    cal = calibrate()
    json.dump({"C_fp": 1.0, "max_ratio_by_case": cal, "max_ratio": max(cal.values())},
              open(OUT / "fp_calibration.json", "w"), indent=1)
    print("calibration max ratio", max(cal.values()), flush=True)

    with open(OUT / "sweep.jsonl", "w") as f:
        for fam in FAMILIES:
            seeds = [1] if fam in DETERMINISTIC else list(range(1, SEEDS + 1))
            for n in NS:
                for seed in seeds:
                    d, b = make_matrix(fam, n, seed)
                    be = FPBackend(d, b, DELTA, C_fp=1.0)
                    res = run_algorithm(be, fam, n, seed, DELTA)
                    rec = res.to_dict()
                    # gaps / certificates at the three stopping points
                    for key, j in [("heur", res.stop_j_heuristic), ("cert8", res.stop_j_cert_eighth),
                                   ("cert2", res.stop_j_cert_half)]:
                        if j is None:
                            rec[f"{key}_j"] = None
                            continue
                        st = state_at(res, j)
                        rec[f"{key}_j"] = j
                        rec[f"{key}_t"] = st.t
                        rec[f"{key}_h"] = st.h
                        rec[f"{key}_gap"] = be.gap(st.t)
                        rec[f"{key}_cert"] = cert_gap(st.h, n)
                    # practical output: cert8 if available, else last decidable iterate
                    if res.stop_j_cert_eighth is not None:
                        jp = res.stop_j_cert_eighth; rec["practical_certified"] = True
                    else:
                        dec = [r.j for r in res.iters if r.d_err < 0.125]
                        jp = dec[-1] if dec else 0; rec["practical_certified"] = False
                    st = state_at(res, jp)
                    rec["practical_j"] = jp; rec["practical_t"] = st.t; rec["practical_h"] = st.h
                    rec["practical_gap"] = be.gap(st.t); rec["practical_cert"] = cert_gap(st.h, n)
                    be2 = FPBackend(d, b, DELTA, C_fp=1.0)
                    t0 = time.time()
                    run_algorithm(be2, fam, n, seed, DELTA, max_iter=jp, record_S=False,
                                  stop_when_undecidable=False)
                    rec["practical_evals"] = be2.evals  # distinct F evaluations up to iteration jp
                    rec["practical_wall"] = time.time() - t0
                    rec["gap_resolution"] = be.gap_resolution()
                    rng = np.random.default_rng(10_000 + seed)
                    rec["baselines"] = baselines(d, b, DELTA, rng, n_grid=1001)
                    f.write(json.dumps(rec) + "\n")
                print(f"{fam:17s} n={n:4d} done  [{time.time()-t_start:6.1f}s]", flush=True)


if __name__ == "__main__":
    main()
