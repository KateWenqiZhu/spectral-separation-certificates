"""
Rigorous (arb ball arithmetic) runs of the full prescribed Algorithm 1
(all K refinements) on one instance per family, for the sizes given on the
command line.  Also validates the double-precision run on the same matrix:
identical selection path up to the fp-undecidable iteration, and
|d_hat_fp - d_arb| <= d_err_fp.

usage: python3 run_arb.py 10 20 50 100        (sizes)
       python3 run_arb.py 200 --families random wilkinson
output: data/arb_<n>_<family>.json
"""
import json, sys, time, pathlib, argparse
import numpy as np
from ke02num.core import (FAMILIES, make_matrix, FPBackend, ArbBackend, run_algorithm, state_at)

DELTA = 0.48
OUT = pathlib.Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", nargs="+", type=int)
    ap.add_argument("--families", nargs="*", default=FAMILIES)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--prec", type=int, default=256)
    a = ap.parse_args()
    for n in a.sizes:
        for fam in a.families:
            path = OUT / f"arb_{n}_{fam}.json"
            if path.exists():
                print("skip", path, flush=True)
                continue
            d, b = make_matrix(fam, n, a.seed)
            t0 = time.time()
            ar = ArbBackend(d, b, DELTA, prec=a.prec)
            res = run_algorithm(ar, fam, n, a.seed, DELTA, stop_when_undecidable=False)
            fp = FPBackend(d, b, DELTA, C_fp=1.0)
            resfp = run_algorithm(fp, fam, n, a.seed, DELTA, stop_when_undecidable=True)
            # validation of the fp run against the rigorous one
            # (i) first iteration at which fp and arb select different candidates, and
            #     whether that happened at a near-tie (|d_L - d_R| within the fp error bound);
            # (ii) |d_hat_fp - d_arb| <= d_err_fp along the common path.
            first_mismatch = None
            mismatch_is_tie = None
            model_ok = True
            worst_ratio = 0.0
            for rf in resfp.iters:
                ra = state_at(res, rf.j)
                if ra.t != rf.t:
                    first_mismatch = rf.j
                    # arb candidate potentials are stored on the *arb* record of iteration j
                    mismatch_is_tie = abs(ra.d_left - ra.d_right) <= 2 * rf.d_err
                    break
                err = abs(rf.d_hat - ra.d_hat)
                worst_ratio = max(worst_ratio, err / rf.d_err if rf.d_err > 0 else 0)
                if err > rf.d_err + ra.d_err:
                    model_ok = False
            same_path = first_mismatch is None
            rec = res.to_dict()
            rec["R"] = res.gap_final / res.cert_final
            rec["gap_final_rel_rad"] = res.gap_final_err / max(res.gap_final, 1e-300)
            rec["arb_wall_eig"] = ar.wall
            rec["fp_validation"] = dict(same_path=same_path, first_mismatch_j=first_mismatch, mismatch_is_tie=mismatch_is_tie, model_ok=model_ok,
                                        worst_err_over_bound=worst_ratio,
                                        fp_undecidable_j=resfp.undecidable_j,
                                        fp_cert8_j=resfp.stop_j_cert_eighth,
                                        fp_last_j=resfp.iters[-1].j)
            rec["fp_iters"] = [vars(x) for x in resfp.iters]
            rec["fp_gap_final"] = resfp.gap_final
            rec["fp_final_t"] = resfp.final_t
            rec["fp_last_j"] = resfp.iters[-1].j
            json.dump(rec, open(path, "w"), indent=1)
            print(f"{fam:17s} n={n:4d}  K={res.K} evals={res.evals} wall={time.time()-t0:7.1f}s "
                  f"gap={res.gap_final:.3e}(+-{res.gap_final_err:.1e}) cert={res.cert_final:.3e} "
                  f"log10R={np.log10(rec['R']):.1f} d_K={res.iters[-1].d_hat:.3e}+-{res.iters[-1].d_err:.1e} "
                  f"fp:mismatch_j={first_mismatch} tie={mismatch_is_tie} model_ok={model_ok} worst={worst_ratio:.2f}", flush=True)


if __name__ == "__main__":
    main()
