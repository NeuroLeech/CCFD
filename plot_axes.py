"""Plot the 13 independent measure axes against the non-input parameters.

One representative per redundancy group (|rho| > 0.8), chosen as the most
reliable member of its group. The question the figures answer is narrow: which
of the fluid/drive parameters actually move each axis, and does the answer differ
between the two input conditions.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import FIELDS as F, VIDEOS as OUT

# one representative per redundancy group; parenthesised members follow it
AXES = [
    ("active_frac_mean", "extent / reach / turnover  (12 measures)"),
    ("allpair_r",        "pattern repertoire  (+energy_in_driven_cv)"),
    ("anisotropy_mean",  "anisotropy  (+cv, tau, length_scale_cv)"),
    ("drive_tau",        "drive timescale  (+top5_energy_tau)"),
    ("partratio_cv",     "concentration variability  (+active_frac_cv)"),
    ("pattern_tau",      "pattern decorrelation"),
    ("corr_length",      "correlation length"),
    ("field_tau",        "field decorrelation"),
    ("length_scale_mean", "gradient length scale"),
    ("length_scale_tau", "length-scale decorrelation"),
    ("n_clusters_cv",    "cluster-count variability"),
    ("speed_mean",       "normal-flow speed"),
    ("tau_ratio",        "field tau / drive tau"),
]
PARAMS = [("Ld", True), ("sponge_strength", False), ("sponge_width", False),
          ("tau", True), ("silent", False)]


def rank_r(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or np.std(a[m]) < 1e-30 or np.std(b[m]) < 1e-30:
        return np.nan
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    z = np.load(os.path.join(F, "measures.npz"), allow_pickle=True)
    M, keys = z["M"], list(z["keys"])
    man = json.load(open(os.path.join(F, "manifest.json")))
    isB = np.array([r["cond"] == "B_V1" for r in man])
    P = {p: np.array([r[p] for r in man]) for p, _ in PARAMS}
    rel = {k: v for k, v in zip(keys, np.zeros(len(keys)))}

    # ---------------------------------------------------------------- heatmap
    R = np.full((len(AXES), len(PARAMS)), np.nan)
    for i, (k, _) in enumerate(AXES):
        v = M[:, keys.index(k)]
        for j, (p, _lg) in enumerate(PARAMS):
            R[i, j] = rank_r(v, P[p])

    fig, ax = plt.subplots(figsize=(7.2, 8.0))
    im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(PARAMS)))
    ax.set_xticklabels([p for p, _ in PARAMS], rotation=35, ha="right")
    ax.set_yticks(range(len(AXES)))
    ax.set_yticklabels([f"{k}" for k, _ in AXES], fontsize=9)
    for i in range(len(AXES)):
        for j in range(len(PARAMS)):
            if np.isfinite(R[i, j]):
                ax.text(j, i, f"{R[i,j]:+.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if abs(R[i, j]) > 0.6 else "black")
    ax.set_title("Spearman rho: measure axis vs non-input parameter\n"
                 "n = 20 runs (10 parameter sets x 2 input conditions)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.6, label="rho")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "axes_vs_params_heatmap.png"), dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- scatters
    fig, axs = plt.subplots(len(AXES), len(PARAMS),
                            figsize=(13.5, 2.05*len(AXES)))
    for i, (k, lbl) in enumerate(AXES):
        v = M[:, keys.index(k)]
        for j, (p, lg) in enumerate(PARAMS):
            a = axs[i, j]
            x = P[p]
            a.scatter(x[~isB], v[~isB], s=34, c="#1f77b4", label="A 10r+10v",
                      edgecolors="none")
            a.scatter(x[isB], v[isB], s=34, c="#d62728", marker="s",
                      label="B V1", edgecolors="none")
            if lg:
                a.set_xscale("log")
            r = R[i, j]
            a.set_title(f"rho {r:+.2f}", fontsize=8,
                        color="k" if abs(r) < 0.5 else "#b30000")
            a.tick_params(labelsize=7)
            if j == 0:
                a.set_ylabel(k, fontsize=8)
            if i == len(AXES)-1:
                a.set_xlabel(p, fontsize=8)
            if i == 0 and j == 0:
                a.legend(fontsize=7, loc="best", framealpha=0.7)
    fig.suptitle("13 independent measure axes vs the non-input parameters "
                 "(input regions fixed)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(os.path.join(OUT, "axes_vs_params_scatter.png"), dpi=110)
    plt.close(fig)

    # strongest relationships, printed
    print("  strongest measure-parameter relationships (|rho| >= 0.5):")
    flat = [(abs(R[i, j]), R[i, j], AXES[i][0], PARAMS[j][0])
            for i in range(len(AXES)) for j in range(len(PARAMS))
            if np.isfinite(R[i, j]) and abs(R[i, j]) >= 0.5]
    for _, r, k, p in sorted(flat, reverse=True):
        print(f"    {k:20s} vs {p:16s} rho {r:+.2f}")
    print(f"\n  wrote axes_vs_params_heatmap.png and axes_vs_params_scatter.png")


if __name__ == "__main__":
    main()
