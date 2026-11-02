"""
Core routines for the KE-02 numerical experiments.

Conventions follow the manuscript "Convexity Certificates for Spectral
Separation of Tridiagonal Matrices":

    X = diag(1,...,n),   H(t) = T + tX,   0 <= t <= delta/n,
    F(t) = sum_j j * lambda_j(H(t))       (eigenvalues in nondecreasing order),
    S_h(t) = (F(t+h) + F(t-h) - 2F(t)) / h  (centered slope variation),
    Algorithm 1: K = ceil(log2(16 n^3)), theta = 1/(4K),
                 t0 = h0 = delta/(2n); each refinement h <- (1/2 - theta) h,
                 candidates [t-2h, t] and [t, t+2h], endpoint secant slopes
                 over a length 2*theta*h, keep the candidate with the smaller
                 endpoint-slope difference; return (t, h/2).
    Certificate (Prop. 3.1):  S_h(t) < 1/2  =>  gap(H(t)) > h^2 / (2^20 n^4).

Two evaluation backends for F are provided:
    * "fp"  : IEEE double precision (LAPACK tridiagonal eigensolver), with a
              documented a-priori error model used for *fp-certified* early
              stopping;
    * "arb" : rigorous ball arithmetic through python-flint (arb/acb), giving
              genuine enclosures of F, of the endpoint-slope differences and of
              the final gap.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal, eigh_tridiagonal

U_DOUBLE = 2.0 ** -53  # unit roundoff


# --------------------------------------------------------------------------
# Matrix families (real symmetric tridiagonal, ||T||_2 <= 1)
# --------------------------------------------------------------------------

def _normalize(d: np.ndarray, b: np.ndarray, target: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    lam = eigvalsh_tridiagonal(d, b)
    s = max(abs(lam[0]), abs(lam[-1]))
    if s == 0:
        return d, b
    return d * (target / s), b * (target / s)


def make_matrix(family: str, n: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (diag d, offdiag beta) of a real symmetric tridiagonal T with ||T||_2 <= 1."""
    rng = np.random.default_rng(seed)
    if family == "diag_repeated":
        # beta = 0, diagonal drawn from n/2 equispaced values in [-1/2,1/2]:
        # exactly repeated eigenvalues at t=0 and many exact crossings
        # d_i + i t = d_j + j t inside the window 0 < t <= delta/n
        vals = np.linspace(-0.5, 0.5, max(2, n // 2))
        d = rng.choice(vals, size=n)
        b = np.zeros(n - 1)
        return d, b
    if family == "toeplitz":
        # Jacobi matrix of Chebyshev polynomials: eigenvalues cos(k pi/(n+1))
        d = np.zeros(n)
        b = 0.5 * np.ones(n - 1)
        return d, b
    if family == "random":
        d = rng.uniform(-1, 1, n)
        b = rng.uniform(-1, 1, n - 1)
        return _normalize(d, b)
    if family == "nearly_reducible":
        d = rng.uniform(-1, 1, n)
        b = rng.uniform(-1, 1, n - 1)
        m = max(2, n // 20)
        idx = rng.choice(n - 1, size=m, replace=False)
        b[idx] = 1e-9 * np.sign(b[idx] + 1e-300)
        return _normalize(d, b)
    if family == "wilkinson":
        # W_n^+ : diag |i-(n+1)/2|, offdiag 1, scaled to unit norm
        i = np.arange(1, n + 1)
        d = np.abs(i - (n + 1) / 2.0)
        b = np.ones(n - 1)
        return _normalize(d, b)
    if family == "two_blocks":
        # two identical random tridiagonal blocks joined by a tiny coupling
        m = n // 2
        d1 = rng.uniform(-1, 1, m)
        b1 = rng.uniform(-1, 1, m - 1)
        d = np.concatenate([d1, d1 if 2 * m == n else np.concatenate([d1, [rng.uniform(-1, 1)]])])
        b = np.concatenate([b1, [1e-7], b1 if 2 * m == n else np.concatenate([b1, [rng.uniform(-1, 1)]])])
        return _normalize(d, b)
    raise ValueError(family)


FAMILIES = ["diag_repeated", "toeplitz", "random", "nearly_reducible", "wilkinson", "two_blocks"]
DETERMINISTIC = {"toeplitz", "wilkinson"}
FAMILY_LABEL = {
    "diag_repeated": "diagonal, repeated eigenvalues",
    "toeplitz": "tridiagonal Toeplitz (Chebyshev Jacobi)",
    "random": "random tridiagonal",
    "nearly_reducible": "nearly reducible",
    "wilkinson": "Wilkinson $W_n^+$",
    "two_blocks": "two weakly coupled blocks",
}


# --------------------------------------------------------------------------
# Double-precision backend
# --------------------------------------------------------------------------

class FPBackend:
    """F(t) in double precision with an a-priori absolute error model.

    Error model.  LAPACK's symmetric tridiagonal eigensolvers return the exact
    eigenvalues of T + E with ||E||_2 <= p(n) u ||T||_2 for a modest p(n).
    We use   eps_F = C_fp * n * u * ||H||_2 * n(n+1)/2,
    with ||H||_2 <= 1 + delta, and calibrate C_fp against arb evaluations
    (see calibrate_fp_error).  This is an *empirical* error model, so stopping
    decisions based on it are called "fp-certified", as opposed to the
    rigorous arb enclosures.
    """

    name = "fp"

    def __init__(self, d: np.ndarray, b: np.ndarray, delta: float, C_fp: float = 1.0):
        self.d = np.asarray(d, float)
        self.b = np.asarray(b, float)
        self.n = len(d)
        self.delta = delta
        self.X = np.arange(1, self.n + 1, dtype=float)
        self.w = self.X.copy()
        self.evals = 0
        self.cache: Dict[float, float] = {}
        self.C_fp = C_fp

    def F(self, t: float) -> float:
        t = float(t)
        if t in self.cache:
            return self.cache[t]
        lam = eigvalsh_tridiagonal(self.d + t * self.X, self.b, lapack_driver="stemr")
        val = float(np.dot(self.w, lam))
        self.cache[t] = val
        self.evals += 1
        return val

    def eps_F(self) -> float:
        n = self.n
        return self.C_fp * n * U_DOUBLE * (1.0 + self.delta) * n * (n + 1) / 2.0

    def gap(self, t: float) -> float:
        lam = eigvalsh_tridiagonal(self.d + t * self.X, self.b, lapack_driver="stemr")
        return float(np.min(np.diff(lam)))

    def gap_resolution(self) -> float:
        """Absolute eigenvalue accuracy of the double-precision solver (model)."""
        return self.C_fp * self.n * U_DOUBLE * (1.0 + self.delta)


# --------------------------------------------------------------------------
# Rigorous backend (python-flint / arb)
# --------------------------------------------------------------------------

class ArbBackend:
    """F(t) as a rigorous arb ball via certified eigenvalue enclosures."""

    name = "arb"

    def __init__(self, d: np.ndarray, b: np.ndarray, delta: float, prec: int = 256,
                 algorithm: str = "vdhoeven_mourrain"):
        import flint
        from flint import arb, acb_mat, ctx
        self.flint = flint
        self.prec = prec
        ctx.prec = prec
        self.n = len(d)
        self.delta = delta
        self.algorithm = algorithm
        # exact conversion of the double entries to arb
        self.d = [arb(float(x)) for x in d]
        self.b = [arb(float(x)) for x in b]
        self.evals = 0
        self.cache: Dict[str, object] = {}
        self.wall = 0.0

    def _matrix(self, t):
        from flint import acb_mat, arb
        n = self.n
        rows = [[arb(0)] * n for _ in range(n)]
        for i in range(n):
            rows[i][i] = self.d[i] + t * (i + 1)
            if i + 1 < n:
                rows[i][i + 1] = self.b[i]
                rows[i + 1][i] = self.b[i]
        return acb_mat(rows)

    def eigs(self, t):
        """Sorted rigorous enclosures (list of arb) of the eigenvalues of H(t)."""
        from flint import ctx, arb
        ctx.prec = self.prec
        if not hasattr(t, "mid"):
            t = arb(float(t))          # exact: never let t*(i+1) round in double
        key = str(t)
        if key in self.cache:
            return self.cache[key]
        t0 = time.time()
        if all(float(x) == 0.0 for x in self.b):
            # diagonal matrix: eigenvalues d_i + i t exactly (no isolation needed)
            re = [self.d[i] + t * (i + 1) for i in range(self.n)]
        else:
            # an unreduced symmetric tridiagonal matrix has simple eigenvalues,
            # so isolation succeeds at sufficiently high precision
            # (nearly) coincident eigenvalues -- e.g. the top pair of W_50^+ differs by
            # ~3e-51 -- may need more than the working precision to be isolated
            ev = None
            for mult in (1, 2, 4, 8):
                ctx.prec = self.prec * mult
                try:
                    ev = self._matrix(t).eig(algorithm=self.algorithm)
                    break
                except ValueError:
                    continue
            ctx.prec = self.prec
            if ev is None:
                raise ValueError("eigenvalue isolation failed up to prec %d" % (8 * self.prec))
            re = [e.real for e in ev]
        # real symmetric: take real parts; sort by midpoint; enlarge by max radius
        # sort by exact (arb) midpoints; float() would merge nearby eigenvalues
        import functools
        mids = [x.mid() for x in re]
        order = sorted(range(self.n), key=functools.cmp_to_key(
            lambda i, j: -1 if mids[i] < mids[j] else (1 if mids[i] > mids[j] else 0)))
        re_sorted = [re[i] for i in order]
        rmax = max(float(x.rad()) for x in re_sorted)
        # order statistics are 1-Lipschitz in sup norm -> uniform enlargement
        from flint import arb
        out = [arb(x.mid(), rmax) for x in re_sorted]
        self.cache[key] = out
        self.evals += 1
        self.wall += time.time() - t0
        return out

    def F(self, t):
        ev = self.eigs(t)
        from flint import arb
        s = arb(0)
        for j, lam in enumerate(ev, start=1):
            s += j * lam
        return s

    def gap(self, t):
        if not hasattr(t, "mid"):
            t = self.flint.arb(float(t))
        ev = self.eigs(t)
        g = None
        for a, c in zip(ev[:-1], ev[1:]):
            dlt = c - a
            g = dlt if g is None else (dlt if float(dlt.mid()) < float(g.mid()) else g)
        return g  # arb ball


# --------------------------------------------------------------------------
# Algorithm 1 with full bookkeeping
# --------------------------------------------------------------------------

@dataclass
class IterRecord:
    j: int
    t: float
    h: float            # radius of the selected interval I_j
    d_hat: float        # computed endpoint-slope difference of I_j
    d_err: float        # error bound on d_hat (fp model or arb radius)
    d_left: float = float("nan")
    d_right: float = float("nan")
    S_half: float = float("nan")   # S_{h/2}(t) computed (midpoint)
    S_half_err: float = float("nan")
    S_full: float = float("nan")   # S_{h}(t): direct certificate test at the full window radius
    S_full_err: float = float("nan")
    gap_direct: float = float("nan")   # rigorous/model lower bound on gap(H(t_j)) from eigenvalue enclosures


@dataclass
class RunResult:
    family: str
    n: int
    seed: int
    delta: float
    backend: str
    K: int
    theta: float
    iters: List[IterRecord] = field(default_factory=list)
    evals: int = 0
    wall: float = 0.0
    stop_j_heuristic: Optional[int] = None    # first j with d_hat < 1/8 (uncertified)
    stop_j_cert_eighth: Optional[int] = None  # first j with d_hat + err < 1/8
    stop_j_cert_half: Optional[int] = None    # first j with d_hat + err < 1/2
    undecidable_j: Optional[int] = None       # first j with err >= 1/8
    stop_j_S: Optional[int] = None            # first j with S_{h_j}(t_j) + err < 1/2 (direct certificate)
    stop_j_S_half: Optional[int] = None       # first j with S_{h_j/2}(t_j) + err < 1/2
    stopped_by: str = "K"
    final_t: float = float("nan")
    final_h_cert: float = float("nan")        # certification radius returned (h/2)
    gap_before: float = float("nan")
    gap_final: float = float("nan")
    gap_final_err: float = float("nan")
    cert_final: float = float("nan")          # h_cert^2 / (2^20 n^4)
    cert_theory: float = float("nan")         # 2^-36 delta^2 / n^12

    def to_dict(self):
        d = asdict(self)
        return d


def K_theta(n: int) -> Tuple[int, float]:
    K = int(math.ceil(math.log2(16 * n ** 3)))
    return K, 1.0 / (4 * K)


def _as_float_and_err(x) -> Tuple[float, float]:
    """Return (midpoint, radius) for an arb ball or (x, 0) for a float."""
    if hasattr(x, "mid"):
        return float(x.mid()), float(x.rad())
    return float(x), 0.0


def endpoint_slope_difference(backend, t: float, h: float, theta: float):
    """d(I) = s_R(I) - s_L(I) for I=[t-h,t+h], each secant over a length 2*theta*h.

    Returns (d_hat, d_err).  For the fp backend d_err = 4 eps_F / (2 theta h).
    For the arb backend d_err is the radius of the rigorous enclosure.
    """
    if backend.name == "fp":
        a, b = t - h, t + h
        L = 2.0 * theta * h
        Fa, Fa2, Fb2, Fb = backend.F(a), backend.F(a + L), backend.F(b - L), backend.F(b)
        d = ((Fb - Fb2) - (Fa2 - Fa)) / L
        return float(d), 4.0 * backend.eps_F() / L
    # rigorous: form the abscissae in ball arithmetic so that rounding of the
    # evaluation points is accounted for (t, h are exact doubles)
    arb = backend.flint.arb
    T, Hh = arb(t), arb(h)
    L = 2 * arb(theta) * Hh
    a, b = T - Hh, T + Hh
    Fa, Fa2, Fb2, Fb = backend.F(a), backend.F(a + L), backend.F(b - L), backend.F(b)
    val = ((Fb - Fb2) - (Fa2 - Fa)) / L
    return _as_float_and_err(val)


def slope_variation(backend, t: float, h: float):
    if backend.name == "fp":
        Fp, Fm, F0 = backend.F(t + h), backend.F(t - h), backend.F(t)
        return float((Fp + Fm - 2 * F0) / h), 4.0 * backend.eps_F() / h
    arb = backend.flint.arb
    T, Hh = arb(t), arb(h)
    Fp, Fm, F0 = backend.F(T + Hh), backend.F(T - Hh), backend.F(T)
    s, se = _as_float_and_err((Fp + Fm - 2 * F0) / Hh)
    if s + se < 0:   # impossible for a convex F: enclosure bug
        raise RuntimeError(f"negative slope variation enclosure {s} +- {se}")
    return s, se


def run_algorithm(backend, family: str, n: int, seed: int, delta: float,
                  max_iter: Optional[int] = None, stop_when_undecidable: bool = True,
                  record_S: bool = True, time_limit: Optional[float] = None,
                  record_S_full: bool = False, record_gap_direct: bool = False,
                  stop_on: Optional[str] = None) -> RunResult:
    """Run Algorithm 1 on backend; record the potential and error at every iteration.

    If max_iter is None, the prescribed K refinements are performed (subject to
    stop_when_undecidable for the fp backend: we stop once the error bound on the
    potential exceeds 1/8, since nothing can be certified beyond that point).
    """
    K, theta = K_theta(n)
    res = RunResult(family=family, n=n, seed=seed, delta=delta, backend=backend.name, K=K, theta=theta)
    res.cert_theory = 2.0 ** -36 * delta ** 2 / n ** 12
    t = delta / (2.0 * n)
    h = delta / (2.0 * n)
    wall0 = time.time()
    g0, g0e = _as_float_and_err(backend.gap(0.0))
    res.gap_before = g0

    def record(j, t, h, dl=float("nan"), dr=float("nan")):
        d_hat, d_err = endpoint_slope_difference(backend, t, h, theta)
        rec = IterRecord(j=j, t=t, h=h, d_hat=d_hat, d_err=d_err, d_left=dl, d_right=dr)
        if record_S:
            s, se = slope_variation(backend, t, h / 2.0)
            rec.S_half, rec.S_half_err = s, se
            if res.stop_j_S_half is None and s + se < 0.5:
                res.stop_j_S_half = j
        if record_S_full:
            s, se = slope_variation(backend, t, h)
            rec.S_full, rec.S_full_err = s, se
            if res.stop_j_S is None and s + se < 0.5:
                res.stop_j_S = j
        if record_gap_direct and hasattr(backend, "gap_direct_lower"):
            rec.gap_direct = backend.gap_direct_lower(t)
        elif record_gap_direct and backend.name == "fp":
            rec.gap_direct = backend.gap(t) - 2 * backend.gap_resolution()
        res.iters.append(rec)
        if res.stop_j_heuristic is None and d_hat < 0.125:
            res.stop_j_heuristic = j
        if res.stop_j_cert_eighth is None and d_hat + d_err < 0.125:
            res.stop_j_cert_eighth = j
        if res.stop_j_cert_half is None and d_hat + d_err < 0.5:
            res.stop_j_cert_half = j
        if res.undecidable_j is None and d_err >= 0.125:
            res.undecidable_j = j
        return rec

    record(0, t, h)
    n_iter = K if max_iter is None else min(K, max_iter)
    for j in range(1, n_iter + 1):
        if stop_when_undecidable and backend.name == "fp" and res.undecidable_j is not None:
            break
        if stop_on == "S" and res.stop_j_S is not None:
            res.stopped_by = "S"; break
        if stop_on == "cert8" and res.stop_j_cert_eighth is not None:
            res.stopped_by = "cert8"; break
        if time_limit is not None and time.time() - wall0 > time_limit:
            break
        hh = (0.5 - theta) * h
        dl, _ = endpoint_slope_difference(backend, t - hh, hh, theta)
        dr, _ = endpoint_slope_difference(backend, t + hh, hh, theta)
        t = t - hh if dl <= dr else t + hh
        h = hh
        record(j, t, h, dl, dr)
    res.final_t = t
    res.final_h_cert = h / 2.0
    g, ge = _as_float_and_err(backend.gap(t))
    res.gap_final, res.gap_final_err = g, ge
    res.cert_final = (h / 2.0) ** 2 / (2.0 ** 20 * n ** 4)
    res.evals = backend.evals
    res.wall = time.time() - wall0
    return res


def state_at(res: RunResult, j: int) -> IterRecord:
    for r in res.iters:
        if r.j == j:
            return r
    raise KeyError(j)


def cert_gap(h_radius: float, n: int) -> float:
    """Certified gap from a selected interval of radius h: (h/2)^2 / (2^20 n^4)."""
    return (h_radius / 2.0) ** 2 / (2.0 ** 20 * n ** 4)


# --------------------------------------------------------------------------
# Baselines (Experiment 3)
# --------------------------------------------------------------------------

def gap_of(d, b, D):
    lam = eigvalsh_tridiagonal(np.asarray(d) + D, np.asarray(b), lapack_driver="stemr")
    return float(np.min(np.diff(lam)))


def baselines(d, b, delta, rng: np.random.Generator, n_grid: int = 2001) -> Dict[str, float]:
    n = len(d)
    X = np.arange(1, n + 1, dtype=float)
    out = {}
    out["none"] = gap_of(d, b, np.zeros(n))
    out["ramp_max"] = gap_of(d, b, (delta / n) * X)
    tr = rng.uniform(0, delta / n)
    out["random_t"] = gap_of(d, b, tr * X)
    out["random_diag"] = gap_of(d, b, rng.uniform(-delta, delta, n))
    ts = np.linspace(0, delta / n, n_grid)
    gs = np.array([gap_of(d, b, tt * X) for tt in ts])
    out["oracle_grid"] = float(gs.max())
    out["oracle_grid_t"] = float(ts[int(gs.argmax())])
    out["grid_median"] = float(np.median(gs))
    return out


def gap_profile(d, b, delta, n_grid: int = 2001):
    n = len(d)
    X = np.arange(1, n + 1, dtype=float)
    ts = np.linspace(0, delta / n, n_grid)
    gs = np.array([gap_of(d, b, tt * X) for tt in ts])
    return ts, gs
