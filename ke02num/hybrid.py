"""
Hybrid rigorous tridiagonal spectral oracle (structure-exploiting).

For H = T + tX (real symmetric tridiagonal, exact entries as dyadic numbers):

  1. double-precision eigenpairs (lambda_hat_j, u_hat_j) from LAPACK dstemr
     (O(n^2));
  2. a rigorous a-posteriori residual bound r_j >= || (H - lambda_hat_j I) u_hat_j ||_2 / ||u_hat_j||_2,
     computed in double precision with an explicit IEEE rounding-error bound,
     against the *exact* H (matrix rounding and rounding of the abscissa t are
     charged to the bound).  For Hermitian H this gives
     dist(lambda_hat_j, spec(H)) <= r_j   (Krylov-Weinstein / Bauer-Fike);
  3. the intervals [lambda_hat_j - r_j, lambda_hat_j + r_j] are merged into
     clusters; a cluster made of m intervals is verified to contain exactly m
     eigenvalues by two rigorous Sturm counts in arb ball arithmetic (O(n) each);
     disjoint singleton intervals need no count (pigeonhole: n disjoint
     intervals each containing >= 1 of the n eigenvalues).
  4. Since sorted true eigenvalues and sorted approximations both lie inside
     the same cluster intervals, |lambda_(j) - lambda_hat_(j)| <= width of the
     cluster containing j, which gives rigorous enclosures of F(t), of the
     endpoint-slope differences, of S_h(t), and a rigorous lower bound on gap(H).

Cost: O(n^2) floating point + O(n * #clusters) ball arithmetic, versus O(n^3)
high-precision dense eigensolution for the dense arb backend.  Enclosure width
is limited by double-precision eigenpair accuracy (~ n u ||H||), which is
sufficient for the direct S_h(t) < 1/2 test at the early iterations where it
first succeeds, but not for the endpoint-slope potential at late iterations.
"""
from __future__ import annotations
import math, time
from typing import Dict, List, Tuple
import numpy as np
from scipy.linalg import eigh_tridiagonal

U = 2.0 ** -53


class Ball:
    """Minimal midpoint-radius interval, enough for F-combinations."""
    __slots__ = ("m", "r")

    def __init__(self, m, r=0.0):
        self.m = float(m); self.r = float(r)

    def mid(self): return self.m
    def rad(self): return self.r

    def _c(self, o):
        if isinstance(o, Ball):
            return o
        if hasattr(o, "mid"):      # arb ball
            return Ball(float(o.mid()), float(o.rad()))
        return Ball(float(o), 0.0)

    def __add__(self, o):
        o = self._c(o); m = self.m + o.m
        return Ball(m, self.r + o.r + abs(m) * 2 * U)

    __radd__ = __add__

    def __sub__(self, o):
        o = self._c(o); m = self.m - o.m
        return Ball(m, self.r + o.r + abs(m) * 2 * U)

    def __rsub__(self, o):
        return self._c(o).__sub__(self)

    def __mul__(self, o):
        o = self._c(o); m = self.m * o.m
        return Ball(m, abs(self.m) * o.r + abs(o.m) * self.r + self.r * o.r + abs(m) * 2 * U)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = self._c(o)
        if o.r >= abs(o.m):
            raise ZeroDivisionError("division by a ball containing zero")
        m = self.m / o.m
        lo = abs(o.m) - o.r
        return Ball(m, (self.r + abs(m) * o.r) / lo + abs(m) * 2 * U)

    def __repr__(self): return f"[{self.m:.6e} +- {self.r:.1e}]"


class HybridBackend:
    name = "hyb"

    def __init__(self, d, b, delta, verify_prec: int = 256):
        import flint
        self.flint = flint
        self.d = np.asarray(d, float); self.b = np.asarray(b, float)
        self.n = len(d); self.delta = delta
        self.X = np.arange(1, self.n + 1, dtype=float)
        self.w = self.X.copy()
        self.prec = verify_prec
        self.cache: Dict[str, object] = {}
        self.evals = 0
        self.wall = 0.0
        self.n_counts = 0
        self.max_cluster = 1
        self.refined: Dict[str, object] = {}
        self.refined_mode = False
        self.wall_refine = 0.0
        # exact arb copies of the data (beta_i^2 is exact at >= 106 bits)
        flint.ctx.prec = self.prec
        arb = flint.arb
        self._d_arb = [arb(float(x)) for x in d]
        self._b2_arb = [arb(float(x)) ** 2 for x in b]  # beta_i^2 exactly (at prec)

    # ---------------------------------------------------------------- Sturm
    def sturm_count(self, sigma) -> int:
        """Rigorous number of eigenvalues of H(t) strictly below sigma (arb ball).

        H is passed through self._cur (exact arb diagonal of H(t)).  Raises if a
        pivot ball contains zero (caller perturbs sigma and retries).
        """
        from flint import ctx, arb
        ctx.prec = self.prec
        dg = self._cur
        q = dg[0] - sigma
        cnt = 0
        self.n_counts += 1
        for i in range(self.n):
            if i > 0:
                q = (dg[i] - sigma) - self._b2_arb[i - 1] / q
            if q.contains(0):          # sign undecidable
                raise ValueError("pivot contains zero")
            if q < 0:
                cnt += 1
        return cnt

    def _count_robust(self, x: float, direction: int) -> Tuple[int, float]:
        """Sturm count at x, moving x outward (direction=-1 left, +1 right) if undecidable."""
        from flint import arb
        step = 2 ** -40 * max(1.0, abs(x))
        for k in range(60):
            xx = x + direction * step * (2 ** k - 1) if k else x
            try:
                return self.sturm_count(arb(xx)), xx
            except (ValueError, ZeroDivisionError):
                continue
        raise RuntimeError("Sturm count undecidable")

    # ---------------------------------------------------------------- eigs
    def eigs(self, t) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lambda_hat sorted, widths) with rigorous |lambda_(j) - lambda_hat_j| <= widths[j]."""
        from flint import arb, ctx
        key = str(t)
        if key in self.cache:
            lam, w, self._cur, self._L, self._H = self.cache[key]
            return lam, w
        t0 = time.time()
        ctx.prec = self.prec
        if hasattr(t, "mid"):
            tb = t; tf = float(t.mid()); t_err = float(t.rad()) + abs(tf - float(t.mid()))
        else:
            tb = arb(float(t)); tf = float(t); t_err = 0.0
        # exact diagonal of H(t) in arb (for Sturm counts)
        self._cur = [self._d_arb[i] + tb * (i + 1) for i in range(self.n)]
        n = self.n
        dh = self.d + tf * self.X                       # rounded diagonal
        lam, V = eigh_tridiagonal(dh, self.b, lapack_driver="stemr")
        # residual of each computed pair against the *rounded* matrix, in double
        # R = dh*V + b*(shifted V) - lam*V
        R = dh[:, None] * V - lam[None, :] * V
        R[:-1] += self.b[:, None] * V[1:]
        R[1:] += self.b[:, None] * V[:-1]
        rnorm = np.linalg.norm(R, axis=0)
        vnorm = np.linalg.norm(V, axis=0)
        # rigorous rounding bound for the residual computation (componentwise,
        # at most 5 flops per entry): gamma_5 * (|dh||V| + |b||V_shift| + |lam||V|)
        A = np.abs(dh)[:, None] * np.abs(V) + np.abs(lam)[None, :] * np.abs(V)
        A[:-1] += np.abs(self.b)[:, None] * np.abs(V[1:])
        A[1:] += np.abs(self.b)[:, None] * np.abs(V[:-1])
        g5 = 5 * U / (1 - 5 * U)
        err_round = g5 * np.linalg.norm(A, axis=0)
        # exact H differs from the rounded dh on the diagonal: |d_i + t(i+1) - dh_i|
        # <= u|dh_i| (one rounding of the sum; t*(i+1) is exact for small i, else
        # one more rounding) + |t_err|(i+1)
        diag_err = (2 * U) * np.abs(dh) + U * np.abs(tf) * self.X + t_err * self.X
        err_matrix = np.linalg.norm(diag_err[:, None] * np.abs(V), axis=0)
        # norm computations themselves: relative error <= n u, generously 4 n u
        slack = 1.0 + 4 * n * U
        r = (rnorm * slack + err_round + err_matrix) / (vnorm * (1 - 4 * n * U))
        # sort
        order = np.argsort(lam)
        lam = lam[order]; r = r[order]
        # Rigorous enclosure by two-sided Sturm bracketing on the EXACT matrix.
        # For each sorted index j we certify a bracket [L_j, H_j] containing the
        # j-th eigenvalue by  sturm_count(L_j) == j  and  sturm_count(H_j) == j+1,
        # both evaluated in arb on the exact diagonal self._cur and exact beta^2.
        # The double-precision residual r only SEEDS the bracket; if it were ever
        # too small the count would not match and the bracket is widened, so the
        # float bound carries no rigor -- a wrong r makes the method slower, never
        # unsound.  Eigenvalues that cannot be separated (true/near multiples)
        # share a cluster bracket certified by its two endpoint counts.
        L, H = self._bracket_sturm(lam, r)
        widths = H - L
        self._L, self._H = L, H
        self.cache[key] = (lam, widths, self._cur, L, H)
        self.evals += 1
        self.wall += time.time() - t0
        return lam, widths

    def _bracket_sturm(self, lam, r):
        """Rigorous [L_j, H_j] per sorted eigenvalue; counts on the exact matrix."""
        from flint import arb
        n = self.n
        # outer bounds: ||H||_2 <= 1 + delta < 2, so spectrum in [-2, 2]
        c0, x0 = self._count_robust(-2.0 - 1e-9, -1)
        cn, xn = self._count_robust(2.0 + 1e-9, +1)
        if c0 != 0 or cn != n:
            raise RuntimeError(f"outer Sturm bounds wrong: count(low)={c0}, count(high)={cn}, n={n}")

        def find(target, lamref, rad, outward):
            """A point p with sturm_count(p) == target near eigenvalue `lamref`.

            outward=-1 builds the lower bracket (p < lamref, count target == j);
            outward=+1 the upper bracket (p > lamref, count target == j+1).
            `p` stays strictly on the `outward` side of lamref; widens on an
            undecidable pivot or when too few eigenvalues are enclosed, pulls back
            toward lamref when too many.  Returns None if it cannot isolate.
            """
            rr = rad
            p = lamref + outward * rr
            for _ in range(70):
                try:
                    c = self.sturm_count(arb(p))
                except (ValueError, ZeroDivisionError):
                    rr *= 2.0; p = lamref + outward * rr; continue
                if c == target:
                    return p
                need_wider = (c < target) if outward > 0 else (c > target)
                if need_wider:
                    rr *= 2.0; p = lamref + outward * rr
                else:                       # overshot: pull back toward lamref
                    p = 0.5 * (p + lamref)
            return None

        L = np.full(n, np.nan); H = np.full(n, np.nan)
        for j in range(n):
            L[j] = find(j, lam[j], float(r[j]), -1)
            H[j] = find(j + 1, lam[j], float(r[j]), +1)
        # handle eigenvalues that could not be bracketed individually: cluster them.
        # A nan L[j] means j cannot be separated from j-1; nan H[j] from j+1.
        max_cl = 1
        j = 0
        while j < n:
            k = j
            # extend the cluster while an interior separation is missing
            while k + 1 < n and (np.isnan(H[k]) or np.isnan(L[k + 1]) or H[k] > L[k + 1] + 0):
                k += 1
            # cluster j..k : lower bound = a point with count j, upper = count k+1
            lo = L[j] if not np.isnan(L[j]) else self._count_robust(lam[j] - 8 * float(r[j]), -1)[1]
            hi = H[k] if not np.isnan(H[k]) else self._count_robust(lam[k] + 8 * float(r[k]), +1)[1]
            # verify the cluster really holds exactly (k-j+1) eigenvalues
            cl = self.sturm_count(arb(lo)); cu = self.sturm_count(arb(hi))
            if cl != j or cu != k + 1:
                # widen until the endpoint counts are exactly right
                clo, lo = self._count_robust(lo, -1) if cl != j else (cl, lo)
                chi, hi = self._count_robust(hi, +1) if cu != k + 1 else (cu, hi)
                cl = self.sturm_count(arb(lo)); cu = self.sturm_count(arb(hi))
                if cl != j or cu != k + 1:
                    raise RuntimeError(f"cluster [{j},{k}] count check failed: {cl},{cu} vs {j},{k+1}")
            L[j:k + 1] = lo; H[j:k + 1] = hi
            max_cl = max(max_cl, k - j + 1)
            j = k + 1
        self.max_cluster = max(self.max_cluster, max_cl)
        return L, H

    # ------------------------------------------------------------ refinement
    def _newton(self, x, steps: int = 3):
        """Newton iterations on det(H - xI) via the pivot recurrence, in arb."""
        from flint import arb
        dg = self._cur
        for _ in range(steps):
            q = dg[0] - x; qd = -1
            ssum = qd / q
            for i in range(1, self.n):
                b2 = self._b2_arb[i - 1]
                qd = -1 + b2 * qd / (q * q)
                q = (dg[i] - x) - b2 / q
                ssum += qd / q
            # strip the radius: Newton need not be rigorous, the Sturm verification is
            x = arb((x - 1 / ssum).mid())
        return x

    def refine(self, t, rho_bits: int = 90, steps: int = 3):
        """Narrow each eigenvalue bracket to width ~2^-rho_bits, rigorously.

        Every narrowing step is a Sturm count on the exact matrix, so the
        result is rigorous regardless of how the split point is chosen; a Newton
        step only *proposes* a split point to accelerate convergence, and a plain
        bisection midpoint is the guaranteed fallback.  True/near multiples that
        share a cluster bracket (L_j == L_{j+1}) cannot be narrowed and keep the
        cluster bracket -- that is the honest outcome, not a failure.
        Returns (mids as list of arb, widths np.ndarray).
        """
        from flint import arb, ctx
        key = str(t)
        if key in self.refined:
            return self.refined[key]
        lam, w = self.eigs(t)            # sets self._cur, self._L, self._H
        t0 = time.time()
        ctx.prec = self.prec
        rho = float(arb(2) ** (-rho_bits))
        L = [arb(float(x)) for x in self._L]
        H = [arb(float(x)) for x in self._H]
        widths = w.copy()
        for j in range(self.n):
            # skip eigenvalues that share a cluster bracket (unresolvable multiples)
            shared = (j + 1 < self.n and self._L[j + 1] == self._L[j]) or \
                     (j > 0 and self._L[j - 1] == self._L[j])
            if shared or widths[j] <= rho:
                continue
            lo, hi = L[j], H[j]
            # Newton gives a good first split; bisection guarantees progress.
            try:
                xn = self._newton(arb(float(lam[j])), steps)
            except (ValueError, ZeroDivisionError):
                xn = None
            for _ in range(80):
                wj = float((hi - lo).mid()) + float((hi - lo).rad())
                if wj <= rho:
                    break
                # split point: Newton estimate if strictly inside, else midpoint
                m = xn if (xn is not None and float((xn - lo).mid()) > 0
                           and float((hi - xn).mid()) > 0) else 0.5 * (lo + hi)
                m = arb(m.mid()) if hasattr(m, "mid") else arb(m)
                xn = None  # Newton used once; bisection from here on
                try:
                    c = self.sturm_count(m)
                except (ValueError, ZeroDivisionError):
                    # undecidable exactly at m: nudge the split toward lo
                    m = arb((0.75 * lo + 0.25 * hi).mid())
                    try:
                        c = self.sturm_count(m)
                    except (ValueError, ZeroDivisionError):
                        break
                if c <= j:
                    lo = m
                else:
                    hi = m
            L[j], H[j] = lo, hi
            widths[j] = float((hi - lo).mid()) + float((hi - lo).rad())
        mids = [0.5 * (L[j] + H[j]) for j in range(self.n)]
        self._Lr = [float(x.mid()) for x in L]
        self._Hr = [float(x.mid()) for x in H]
        self._Lr_arb, self._Hr_arb = L, H
        out = (mids, widths)
        self.refined[key] = out
        self.wall_refine += time.time() - t0
        return out

    def F(self, t):
        if self.refined_mode:
            from flint import arb
            mids, w = self.refine(t)
            s = arb(0)
            for j, x in enumerate(mids, start=1):
                s += j * x
            return arb(s.mid(), float(s.rad()) + float(np.dot(self.w, w)))
        self.eigs(t)
        # F(t) = sum_j j * lambda_j, lambda_j in [L_j, H_j]  ->  rigorous ball
        mids = 0.5 * (self._L + self._H)
        half = 0.5 * (self._H - self._L)
        m = float(np.dot(self.w, mids))
        rad = float(np.dot(self.w, half)) + abs(m) * 4 * self.n * U
        return Ball(m, rad)

    def gap(self, t) -> Ball:
        """Ball whose lower end is a rigorous lower bound on gap(H(t))."""
        self.eigs(t)
        # lambda_{j+1} - lambda_j >= L_{j+1} - H_j
        lower = self._L[1:] - self._H[:-1]
        mid = 0.5 * (self._L[1:] + self._H[1:]) - 0.5 * (self._L[:-1] + self._H[:-1])
        k = int(np.argmin(lower))
        return Ball(float(mid[k]), float(mid[k] - lower[k]))

    def gap_direct_lower(self, t) -> float:
        self.eigs(t)
        return float(np.min(self._L[1:] - self._H[:-1]))

    def gap_direct_lower_refined(self, t) -> float:
        self.refine(t)
        L = np.array(self._Lr); H = np.array(self._Hr)
        return float(np.min(L[1:] - H[:-1]))

    def enclosure_width(self, t) -> float:
        lam, w = self.eigs(t)
        return float(w.max())
