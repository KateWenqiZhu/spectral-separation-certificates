"""Figures and tables for Section 5 from data/sweep.jsonl and data/arb_*.json."""
import json, glob, pathlib, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ke02num.core import FAMILIES, FAMILY_LABEL, DETERMINISTIC, make_matrix, gap_profile, K_theta, FPBackend, run_algorithm, state_at

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
TAB = HERE / "tables"; TAB.mkdir(exist_ok=True)
DELTA = 0.48

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9.5, "mathtext.fontset": "dejavuserif",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
                     "pdf.fonttype": 42, "legend.frameon": False})
BLUE, ORANGE, AQUA, VIOLET, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#8a8884", "#2b2b2b"
FAM_ORDER = ["random", "nearly_reducible", "two_blocks", "wilkinson", "toeplitz", "diag_repeated"]
SHORT = {"random": "random", "nearly_reducible": "nearly reducible", "two_blocks": "two weakly coupled blocks",
         "wilkinson": r"Wilkinson $W_n^+$", "toeplitz": "tridiagonal Toeplitz", "diag_repeated": "diagonal, repeated eigenvalues"}

rows = [json.loads(l) for l in open(DATA / "sweep.jsonl")]
recs = []
for r in rows:
    bl = r["baselines"]
    recs.append(dict(family=r["family"], n=r["n"], seed=r["seed"], K=r["K"],
                     d0=r["iters"][0]["d_hat"], cert8=r.get("cert8_j"), cert2=r.get("cert2_j"), heur=r.get("heur_j"),
                     undec=r["undecidable_j"], certified=r["practical_certified"], gap0=r["gap_before"],
                     gap=r["practical_gap"], cert=r["practical_cert"], evals=r["practical_evals"], wall=r["practical_wall"],
                     pj=r["practical_j"], none=bl["none"], ramp=bl["ramp_max"], rt=bl["random_t"], rd=bl["random_diag"],
                     orc=bl["oracle_grid"], gmed=bl["grid_median"], res=r["gap_resolution"]))
df = pd.DataFrame(recs)
df["R"] = df.gap / df.cert
arb = {}
for p in sorted(glob.glob(str(DATA / "arb_*.json"))):
    a = json.load(open(p)); arb[(a["family"], a["n"])] = a

NS = sorted(df.n.unique())


def theory(n): return 2.0 ** -36 * DELTA ** 2 / n ** 12


# ---------------------------------------------------------------- Figure A
fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4), sharex=True, sharey=True)
for ax, fam in zip(axes.ravel(), FAM_ORDER):
    sub = df[df.family == fam]
    g = sub.groupby("n")
    med = g.gap.median(); lo = g.gap.min(); hi = g.gap.max()
    cmed = g.cert.median(); clo = g.cert.min(); chi = g.cert.max()
    ns = np.array(med.index, float)
    cf = g.certified.mean().values
    okc = cf >= 0.5
    ax.fill_between(ns, lo.values, hi.values, color=BLUE, alpha=.18, lw=0)
    ax.plot(ns, med.values, "-", color=BLUE, lw=1.6)
    ax.plot(ns[okc], med.values[okc], "o", color=BLUE, ms=5, label=r"$\mathrm{gap}(T+t_*X)$, fp-certified early stop")
    ax.fill_between(ns, clo.values, chi.values, color=ORANGE, alpha=.18, lw=0)
    ax.plot(ns, cmed.values, "-", color=ORANGE, lw=1.6)
    ax.plot(ns[okc], cmed.values[okc], "s", color=ORANGE, ms=4.5, label=r"certificate $h_*^2/(2^{20}n^4)$, fp-certified early stop")
    if (~okc).any():
        ax.plot(ns[~okc], med.values[~okc], "o", color=BLUE, ms=5, mfc="white", label="last fp-decidable iterate (not certified in double precision)")
        ax.plot(ns[~okc], cmed.values[~okc], "s", color=ORANGE, ms=4.5, mfc="white")
    an = [n for n in NS if (fam, n) in arb]
    if an:
        ax.plot(an, [arb[(fam, n)]["gap_final"] for n in an], "*", color=BLUE, ms=11, mfc="white", mew=1.4,
                label=r"$\mathrm{gap}(T+tX)$, rigorous, all $K$ refinements")
        ax.plot(an, [arb[(fam, n)]["cert_final"] for n in an], "*", color=ORANGE, ms=11, mfc="white", mew=1.4,
                label=r"certificate $h^2/(2^{20}n^4)$, rigorous, all $K$")
    g0 = g.gap0.median()
    ok = g0.values > 0
    ax.plot(ns[ok], g0.values[ok], "x", color=GREY, ms=6, mew=1.2, label=r"$\mathrm{gap}(T)$ (unperturbed)")
    nn = np.logspace(1, np.log10(200), 50)
    ax.plot(nn, theory(nn), "--", color=INK, lw=1.0, label=r"worst-case bound $2^{-36}\delta^2 n^{-12}$")
    ax.set(xscale="log", yscale="log", title=SHORT[fam], ylim=(1e-37, 3))
    ax.set_xticks([10, 20, 50, 100, 200]); ax.set_xticklabels(["10", "20", "50", "100", "200"])
    ax.set_yticks([1, 1e-6, 1e-12, 1e-18, 1e-24, 1e-30, 1e-36])
    ax.grid(alpha=.15)
for ax in axes[1]: ax.set_xlabel("$n$")
for ax in axes[:, 0]: ax.set_ylabel("eigenvalue gap")
hl = {}
for a_ in axes.ravel():
    for h_, l_ in zip(*a_.get_legend_handles_labels()):
        hl.setdefault(l_, h_)
fig.legend(list(hl.values()), list(hl.keys()), loc="lower center", ncol=3, fontsize=8.4, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(FIG / "ke02_gap_vs_certificate.pdf", bbox_inches="tight"); fig.savefig(FIG / "ke02_gap_vs_certificate.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure B
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.2, 4.1), gridspec_kw={"width_ratios": [1.15, 1]})
# left: potential trajectory, random n=50 seed 1
fam, n = "random", 50
r = next(x for x in rows if x["family"] == fam and x["n"] == n and x["seed"] == 1)
it = r["iters"]; js = [x["j"] for x in it]; d = np.array([x["d_hat"] for x in it]); e = np.array([x["d_err"] for x in it])
K = r["K"]
if (fam, n) in arb:
    ita = arb[(fam, n)]["iters"]
    ax.plot([x["j"] for x in ita], [max(x["d_hat"], 1e-9) for x in ita], "-", color=BLUE, lw=1.5, label=r"rigorous $d_j$ (ball arithmetic)")
    ax.plot([x["j"] for x in ita], [max(x["S_half"], 1e-9) for x in ita], ":", color=AQUA, lw=1.5, label=r"rigorous $\mathcal{S}_{h_j/2}(t_j)$")
dec = e < 0.125
ax.errorbar(np.array(js)[dec], d[dec], yerr=np.minimum(e[dec], d[dec] * 0.999), fmt="o", color=ORANGE, ms=4.5, capsize=2, lw=1,
            label=r"double precision $\hat d_j\pm\eta_j$")
if (~dec).any():
    ax.plot(np.array(js)[~dec], np.clip(d[~dec], 1.6e-6, None), "o", color=ORANGE, ms=4.5, mfc="white", label="double precision, undecidable ($\\eta_j\\geq1/8$)")
jj = np.arange(0, K + 1)
ax.plot(jj, n ** 3 / 2.0 ** jj, "--", color=GREY, lw=1, label=r"a priori bound $n^3 2^{-j}$")
ax.axhline(0.5, color=INK, lw=.8, ls="-."); ax.text(K + .3, 0.5, "1/2", va="center", fontsize=8.5)
ax.axhline(0.125, color=INK, lw=.8, ls="-."); ax.text(K + .3, 0.125, "1/8", va="center", fontsize=8.5)
if r["undecidable_j"] is not None:
    ax.axvspan(r["undecidable_j"] - .5, K + .5, color=GREY, alpha=.12, lw=0)
    ax.text(r["undecidable_j"] + .2, 2e-4, "fp error\n$\\eta_j\\geq 1/8$", fontsize=8, color=GREY)
ax.set(yscale="log", xlabel="refinement $j$", ylabel=r"endpoint-slope difference $d_j$",
       title=rf"(a) random tridiagonal, $n={n}$, $K={K}$", ylim=(1e-6, 3e5), xlim=(-.5, K + 2))
ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.9)); ax.grid(alpha=.15)
# right: stopping iteration vs n
cols = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", VIOLET]
mk = ["o", "s", "D", "^", "v", "P"]
for c, m, fam in zip(cols, mk, FAM_ORDER):
    sub = df[df.family == fam]
    g = sub.groupby("n").cert8
    med = g.median(); lo = g.min(); hi = g.max()
    ns = np.array(med.index, float)
    ok = ~med.isna().values
    bx.errorbar(ns[ok] * (1 + 0.04 * (FAM_ORDER.index(fam) - 2.5)), med.values[ok],
                yerr=[med.values[ok] - lo.values[ok], hi.values[ok] - med.values[ok]], fmt=m + "-", color=c, ms=5, lw=1.2, capsize=2, label=SHORT[fam])
nn = np.array(NS, float)
bx.plot(nn, [K_theta(int(x))[0] for x in nn], "k-", lw=1.6, label=r"prescribed $K=\lceil\log_2(16n^3)\rceil$")
und = df.groupby("n").undec.median()
bx.plot(und.index.values, und.values, "--", color=GREY, lw=1.2, label=r"first fp-undecidable $j$")
bx.set(xscale="log", xlabel="$n$", ylabel=r"first certified $j_*$  ($\hat d_j+\eta_j<1/8$)", title="(b) early stopping versus the prescribed $K$")
bx.set_xticks([10, 20, 50, 100, 200]); bx.set_xticklabels(["10", "20", "50", "100", "200"]); bx.set_ylim(-1, 29)
bx.legend(fontsize=7.6, ncol=2, loc="upper left"); bx.grid(alpha=.15)
fig.tight_layout(w_pad=2)
fig.savefig(FIG / "ke02_early_stopping.pdf", bbox_inches="tight"); fig.savefig(FIG / "ke02_early_stopping.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure C
NB = 100
methods = [("none", "$D=0$"), ("ramp", r"ramp $D=(\delta/n)X$"), ("rt", r"random $t\in[0,\delta/n]$, $D=tX$"),
           ("rd", r"random diagonal $|D_{ii}|\leq\delta$"), ("gap", r"Algorithm 1, $D=t_*X$"), ("orc", r"best $t$ on a grid (oracle)")]
fig, axes = plt.subplots(2, 3, figsize=(11.2, 5.6), sharex=True, sharey=True)
FLOOR = -6.0
for ax, fam in zip(axes.ravel(), FAM_ORDER):
    sub = df[(df.family == fam) & (df.n == NB)]
    for k, (col, lab) in enumerate(methods):
        raw = np.log10(np.maximum(sub[col].values, 1e-300))
        y = len(methods) - 1 - k
        color = BLUE if col == "gap" else (INK if col == "orc" else GREY)
        below = raw < FLOOR
        v = raw[~below]
        if below.any():
            ax.scatter([FLOOR - 0.35], [y], s=70, marker="<", color=color, facecolors="white", lw=1.4, zorder=4)
            ax.text(FLOOR - 0.25, y + 0.3, rf"{below.sum()}/{len(raw)} below $10^{{-6}}$" if len(raw) > 1 else r"$< 10^{-6}$",
                    fontsize=6.8, color=color, va="bottom")
        if len(v) > 1:
            ax.boxplot([v], positions=[y], orientation="horizontal", widths=.55, showfliers=False,
                       medianprops=dict(color=color, lw=2), boxprops=dict(color=color), whiskerprops=dict(color=color), capprops=dict(color=color))
            ax.scatter(v, y + np.random.default_rng(0).uniform(-.18, .18, len(v)), s=9, color=color, alpha=.55, lw=0)
        elif len(v) == 1:
            ax.scatter(v, [y], s=60, color=color, marker="D")
    ax.set_yticks(range(len(methods))); ax.set_yticklabels([lab for _, lab in methods][::-1], fontsize=8.2)
    ax.set_title(SHORT[fam]); ax.grid(alpha=.15, axis="x")
    ax.set_xlim(FLOOR - 0.6, -0.4)
    ax.set_xticks([-6, -5, -4, -3, -2, -1]); ax.set_xticklabels(["$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])
for ax in axes[1]: ax.set_xlabel(rf"$\mathrm{{gap}}(T+D)$,  $n={NB}$,  $\|D\|_2 \leq \delta={DELTA}$")
fig.tight_layout(w_pad=1.5)
fig.savefig(FIG / "ke02_baselines.pdf", bbox_inches="tight"); fig.savefig(FIG / "ke02_baselines.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure D (gap profile)
fam, n, seed = "two_blocks", 50, 1
d, b = make_matrix(fam, n, seed)
ts, gs = gap_profile(d, b, DELTA, n_grid=8001)
be = FPBackend(d, b, DELTA)
res = run_algorithm(be, fam, n, seed, DELTA)
jstar = res.stop_j_cert_eighth
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.2, 3.7), gridspec_kw={"width_ratios": [1.45, 1]})
ax.plot(ts, np.maximum(gs, 1e-16), color=INK, lw=0.9, label=r"$\mathrm{gap}(T+tX)$")
cmap = [ORANGE, AQUA, VIOLET, BLUE]
shown = [0, 2, 4, jstar]
for c, j in zip(cmap, shown):
    st = state_at(res, j)
    ax.axvspan(st.t - st.h, st.t + st.h, color=c, alpha=.16, lw=0)
    ax.plot([st.t - st.h, st.t + st.h], [10 ** (-0.25 - 0.3 * shown.index(j))] * 2, color=c, lw=2.4, solid_capstyle="butt",
            label=rf"$I_{{{j}}}$: $d_{{{j}}}\approx{st.d_hat:.3g}$")
st = state_at(res, jstar)
gstar = be.gap(st.t)
ax.plot([st.t], [gstar], "o", color=BLUE, ms=7, zorder=5, label=rf"$t_*$ ($j_*={jstar}$), gap $\approx{gstar:.1e}$")
ax.set(yscale="log", xlabel="$t$", ylabel="minimum eigenvalue gap", xlim=(0, DELTA / n), ylim=(3e-7, 2),
       title=rf"(a) gap profile on $[0,\delta/n]$, two weakly coupled blocks, $n={n}$")
ax.legend(fontsize=7.8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22)); ax.grid(alpha=.15)
# zoom
s4 = state_at(res, 4)
lo_, hi_ = s4.t - s4.h, s4.t + s4.h
tz = np.linspace(lo_, hi_, 4001); gz = np.array([be.gap(tt) for tt in tz])
bx.plot(tz, gz, color=INK, lw=0.9)
for c, j in zip([VIOLET, "#eda100", BLUE], [4, 6, jstar]):
    sj = state_at(res, j)
    bx.axvspan(sj.t - sj.h, sj.t + sj.h, color=c, alpha=.16, lw=0)
    bx.plot([sj.t - sj.h, sj.t + sj.h], [10 ** (-2.05 - 0.18 * [4, 6, jstar].index(j))] * 2, color=c, lw=2.4, solid_capstyle="butt",
            label=rf"$I_{{{j}}}$: $d_{{{j}}}\approx{sj.d_hat:.3g}$")
bx.plot([st.t], [gstar], "o", color=BLUE, ms=7, zorder=5)
bx.set(yscale="log", xlabel="$t$", xlim=(lo_, hi_), ylim=(3e-6, 2e-2), title=r"(b) zoom into $I_4$: refinements avoid the collision dips")
bx.legend(fontsize=7.8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22)); bx.grid(alpha=.15)
fig.tight_layout(w_pad=1.5)
fig.savefig(FIG / "ke02_gap_profile.pdf", bbox_inches="tight"); fig.savefig(FIG / "ke02_gap_profile.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Tables


def fmt_e(x):
    if x is None or (isinstance(x, float) and (math.isnan(x))): return "--"
    if x == 0: return "$0$"
    m, ex = f"{x:.1e}".split("e")
    return rf"${m}\times10^{{{int(ex)}}}$"


def rng_int(s):
    s = s.dropna()
    if len(s) == 0: return "--"
    lo, hi, md = int(s.min()), int(s.max()), int(round(s.median()))
    return f"{md}" if lo == hi else f"{md} [{lo},{hi}]"


lines = []
for fam in FAM_ORDER:
    sub = df[df.family == fam]
    first = True
    for n in NS:
        s = sub[sub.n == n]
        nc = int(s.certified.sum()); tot = len(s)
        lines.append(" & ".join([
            (r"\multirow{%d}{*}{%s}" % (len(NS), SHORT[fam]) if first else ""), str(n), str(int(s.K.iloc[0])),
            fmt_e(s.d0.median()), rng_int(s.cert8), rng_int(s.undec),
            f"{nc}/{tot}", str(int(s.evals.median())), fmt_e(s.gap0.median()), fmt_e(s.gap.median()),
            fmt_e(s.cert.median()), f"{np.log10(s.R).median():.1f}"]) + r" \\")
        first = False
    lines.append(r"\midrule")
lines = lines[:-1]
open(TAB / "table_sweep.tex", "w").write("\n".join(lines) + "\n\\bottomrule\n")

lines = []
for fam in FAM_ORDER:
    for n in NS:
        if (fam, n) not in arb: continue
        a = arb[(fam, n)]
        last = a["iters"][-1]
        v = a["fp_validation"]
        mism = "never" if v["same_path"] else (f"$j={v['first_mismatch_j']}$ (tie)" if v["mismatch_is_tie"] else f"$j={v['first_mismatch_j']}$")
        lines.append(" & ".join([SHORT[fam], str(n), str(a["K"]), fmt_e(a["gap_before"]), fmt_e(last["d_hat"]), fmt_e(last["d_err"]),
                                 fmt_e(a["gap_final"]), fmt_e(a["gap_final_err"]), fmt_e(a["cert_final"]), f"{np.log10(a['R']):.1f}",
                                 fmt_e(a.get("fp_gap_final")), mism]) + r" \\")
open(TAB / "table_arb.tex", "w").write("\n".join(lines) + "\n\\bottomrule\n")

lines = []
for fam in FAM_ORDER:
    s = df[(df.family == fam) & (df.n == NB)]
    ge_rd = (s.gap >= s.rd).mean(); ge_rt = (s.gap >= s.rt).mean()
    lines.append(" & ".join([SHORT[fam], fmt_e(s.none.median()), fmt_e(s.ramp.median()), fmt_e(s.rt.median()), fmt_e(s.rd.median()),
                             fmt_e(s.gap.median()), fmt_e(s.orc.median()), f"{(s.gap / s.orc).median():.2f}",
                             f"{100 * ge_rt:.0f}\\%", f"{100 * ge_rd:.0f}\\%"]) + r" \\")
open(TAB / "table_baselines.tex", "w").write("\n".join(lines) + "\n\\bottomrule\n")

# summary numbers for the text
summary = {}
summary["d0_over_n3"] = {fam: float((df[df.family == fam].d0 / df[df.family == fam].n ** 3).median()) for fam in FAM_ORDER}
summary["calibration"] = json.load(open(DATA / "fp_calibration.json"))["max_ratio"]
# contraction ratio of d_j over decidable iterations with d_j > 1
rat = []
for r in rows:
    it = r["iters"]
    for a_, b_ in zip(it[:-1], it[1:]):
        if a_["d_hat"] > 1 and b_["d_err"] < 1e-2: rat.append(b_["d_hat"] / a_["d_hat"])
summary["contraction_ratio_median"] = float(np.median(rat)); summary["contraction_ratio_q90"] = float(np.quantile(rat, .9))
summary["logR_fp_range"] = [float(np.log10(df.R).min()), float(np.log10(df.R).max())]
summary["frac_ours_ge_rd_all"] = float((df.gap >= df.rd).mean()); summary["frac_ours_ge_rt_all"] = float((df.gap >= df.rt).mean())
summary["ours_over_orc_median_random_families"] = float((df[df.family.isin(["random", "nearly_reducible", "two_blocks", "diag_repeated"])].gap /
                                                          df[df.family.isin(["random", "nearly_reducible", "two_blocks", "diag_repeated"])].orc).median())
def fit_exp(fam, col):
    s = df[df.family == fam].groupby("n")[col].median()
    s = s[s > 0]
    p = np.polyfit(np.log(s.index.values.astype(float)), np.log(s.values), 1)
    return float(p[0])
summary["gap_exponent_fit"] = {fam: fit_exp(fam, "gap") for fam in FAM_ORDER}
summary["oracle_exponent_fit"] = {fam: fit_exp(fam, "orc") for fam in FAM_ORDER}
summary["cert2_saving_median"] = float((df.cert8 - df.cert2).median())
summary["certified_fraction_by_n"] = {int(n): float(df[df.n == n].certified.mean()) for n in NS}
summary["arb_logR"] = {f"{k[0]}:{k[1]}": float(np.log10(a["R"])) for k, a in arb.items()}
summary["arb_wall"] = {f"{k[0]}:{k[1]}": a["wall"] for k, a in arb.items()}
summary["fp_wall_median_by_n"] = {int(n): float(df[df.n == n].wall.median()) for n in NS}
json.dump(summary, open(TAB / "summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
