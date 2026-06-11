"""Step 6 — Regenerate the paper figures.

Reads the CSVs produced by 05_conformal.py from outputs/reported_results/ when
available; otherwise falls back to the reported values shipped in that directory.
Creates the figure directory automatically. Produces Figures 1-5 and Appendix
Figure A.6 of the paper (PPG-only InceptionTime).
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_config

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "axes.linewidth": 0.6})

CLASSES = ["Critical", "Other", "Normal", "Pacing/Block", "AF/AFLT", "Tachy/Brady"]
N_TEST = [268, 1562, 99225, 12756, 58769, 56174]
TARGET = 0.90
C_G, C_C, C_A = "#3B6FB6", "#C0504D", "#E08214"

# Reported values (paper, PPG-only InceptionTime) used as fallback / labels.
GLOBAL_COV = [0.075, 0.329, 0.884, 0.886, 0.912, 0.937]
CC_COV     = [0.825, 0.901, 0.876, 0.933, 0.874, 0.929]
GLOBAL_SZ  = [1.96, 3.06, 2.64, 2.94, 2.57, 1.98]
CC_SZ      = [3.08, 4.28, 3.86, 4.24, 3.60, 2.67]


def _load_per_class(fd):
    import pandas as pd
    csv = fd.parent / "outputs" / "reported_results" / "per_class_inceptiontime.csv"
    if not csv.exists():
        csv = fd.parent / "reported_results" / "per_class_inceptiontime.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv, comment="#")
    df = df[df.cls != "OVERALL"].set_index("cls")
    g = [float(df.loc[c, "lac_global_cov"]) for c in CLASSES]
    cc = [float(df.loc[c, "lac_cc_cov"]) for c in CLASSES]
    gs = [float(df.loc[c, "lac_global_size"]) for c in CLASSES]
    cs = [float(df.loc[c, "lac_cc_size"]) for c in CLASSES]
    return g, cc, gs, cs


def fig1(fd):
    loaded = _load_per_class(fd)
    gcov, cccov = (loaded[0], loaded[1]) if loaded else (GLOBAL_COV, CC_COV)
    fig, ax = plt.subplots(figsize=(7.0, 3.4)); x = np.arange(len(CLASSES)); w = 0.38
    ax.bar(x - w/2, gcov, w, label="Global CP", color=C_G, edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, cccov, w, label="Class-Conditional CP", color=C_C, edgecolor="white", linewidth=0.5)
    ax.axhline(TARGET, color="black", ls="--", lw=1.0, label=f"Target ({TARGET:.0%})")
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, fontsize=8.5)
    ax.set_ylabel("Empirical Coverage"); ax.set_ylim(0, 1.08)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.95)
    for xi, n in zip(x, N_TEST):
        ax.text(xi, -0.13, f"n={n:,}", ha="center", va="top", fontsize=6.2,
                transform=ax.get_xaxis_transform(), color="#555")
    plt.tight_layout(); plt.savefig(fd / "fig1_coverage_by_class.pdf", bbox_inches="tight"); plt.close()


def fig2(fd):
    import pandas as pd
    crit = [0.075, 0.231, 0.163]; overall = [0.900, 0.960, 0.883]
    try:
        rr = fd.parent / "outputs" / "reported_results"
        if not rr.exists():
            rr = fd.parent / "reported_results"
        it = pd.read_csv(rr / "per_class_inceptiontime.csv", comment="#")
        it = it[it.cls == "OVERALL"].iloc[0]
        itc = pd.read_csv(rr / "per_class_inceptiontime.csv", comment="#")
        itc = itc[itc.cls == "Critical"].iloc[0]
        xg = pd.read_csv(rr / "per_class_xgboost.csv", comment="#")
        xgc = xg[xg.cls == "Critical"].iloc[0]; xgo = xg[xg.cls == "OVERALL"].iloc[0]
        overall = [float(it.lac_global_cov), float(it.aps_global_cov), float(xgo.lac_global_cov)]
        crit = [float(itc.lac_global_cov), float(itc.aps_global_cov), float(xgc.lac_global_cov)]
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    cats = ["InceptionTime\nLAC", "InceptionTime\nAPS", "XGBoost\nLAC"]
    x = np.arange(len(cats)); w = 0.38
    ax.bar(x - w/2, overall, w, label="Overall coverage", color="#bdbdbd", edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, crit, w, label="Critical-rhythm coverage", color=C_C, edgecolor="white", linewidth=0.5)
    ax.axhline(TARGET, color="black", ls="--", lw=1.0, label=f"Target ({TARGET:.0%})")
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("Empirical Coverage"); ax.set_ylim(0, 1.08)
    ax.legend(fontsize=7.5, loc="center right", framealpha=0.95)
    plt.tight_layout(); plt.savefig(fd / "fig5_model_comparison.pdf", bbox_inches="tight"); plt.close()


def fig3(fd):
    import pandas as pd
    point = [0.075, 0.075, 0.825, 0.825]
    lo = [0.045, 0.007, 0.780, 0.578]; hi = [0.108, 0.198, 0.869, 0.985]
    try:
        rr = fd.parent / "outputs" / "reported_results"
        if not rr.exists():
            rr = fd.parent / "reported_results"
        d = pd.read_csv(rr / "critical_ci_inceptiontime.csv", comment="#").set_index("strategy")
        g, c = d.loc["global"], d.loc["class_conditional"]
        point = [float(g.estimate), float(g.estimate), float(c.estimate), float(c.estimate)]
        lo = [float(g.seg_lo), float(g.cluster_lo), float(c.seg_lo), float(c.cluster_lo)]
        hi = [float(g.seg_hi), float(g.cluster_hi), float(c.seg_hi), float(c.cluster_hi)]
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    labels = ["Global\n(segment)", "Global\n(cluster)", "Class-Cond\n(segment)", "Class-Cond\n(cluster)"]
    cols = [C_G, C_G, C_C, C_C]; y = np.arange(4)
    for i in range(4):
        ax.errorbar(point[i], y[i], xerr=[[point[i]-lo[i]], [hi[i]-point[i]]], fmt="o",
                    color=cols[i], capsize=5, markersize=7, elinewidth=1.5,
                    markeredgecolor="black", markeredgewidth=0.4)
    ax.axvline(TARGET, color="black", ls="--", lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("Critical-Rhythm Coverage"); ax.set_xlim(0, 1.05)
    plt.tight_layout(); plt.savefig(fd / "fig6_critical_ci.pdf", bbox_inches="tight"); plt.close()


def fig4(fd):
    import pandas as pd
    csv = fd.parent / "outputs" / "reported_results" / "per_patient_inceptiontime.csv"
    if not csv.exists():
        # try the run-output location produced by 05_conformal.py
        csv = fd.parent / "reported_results" / "per_patient_inceptiontime.csv"
    if not csv.exists():
        raise FileNotFoundError(
            "per_patient_inceptiontime.csv is required to reproduce Figure 4. "
            "Run src/05_conformal.py first (it writes this file to "
            "outputs/reported_results/), or use the shipped copy in that directory.")
    df = pd.read_csv(csv, comment="#")
    pp = df["coverage"].values
    n_below_all = int((pp < TARGET).sum())
    big = df[df["n_seg"] >= 30] if "n_seg" in df.columns else df
    n_below_big = int((big["coverage"] < TARGET).sum())
    fig, ax = plt.subplots(figsize=(5.2, 3.4)); bins = np.linspace(0.3, 1.0, 22)
    ax.hist(pp, bins=bins, alpha=0.7, color=C_G, label="Global CP", edgecolor="white", linewidth=0.4)
    ax.axvline(TARGET, color="red", ls="--", lw=1.2, label=f"Target ({TARGET:.0%})")
    ax.set_xlabel("Per-Patient Coverage"); ax.set_ylabel("Number of Patients")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.text(0.02, 0.55,
            f"{n_below_all}/{len(pp)} below target (all)\n"
            f"{n_below_big}/{len(big)} below target (n$\\geq$30 seg)",
            transform=ax.transAxes, fontsize=7, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#cccccc", linewidth=0.5))
    plt.tight_layout(); plt.savefig(fd / "fig4_patient_coverage.pdf", bbox_inches="tight"); plt.close()


def fig5(fd):
    import pandas as pd
    sc = [0.894, 0.923, 0.928]; ss = [2.41, 2.70, 3.05]
    try:
        rr = fd.parent / "outputs" / "reported_results"
        if not rr.exists():
            rr = fd.parent / "reported_results"
        d = pd.read_csv(rr / "sqi_stratum_inceptiontime.csv", comment="#").set_index("stratum")
        sc = [float(d.loc[k, "coverage"]) for k in ["high", "mixed", "low"]]
        ss = [float(d.loc[k, "mean_size"]) for k in ["high", "mixed", "low"]]
    except Exception:
        pass
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    strata = ["High\n(SQI=1)", "Mixed", "Low\n(SQI=0)"]
    cols = ["#2C7BB6", "#FDAE61", "#1A9641"]
    b = ax1.bar(strata, sc, color=cols, edgecolor="white", linewidth=0.6)
    ax1.axhline(TARGET, color="black", ls="--", lw=1.0, label=f"Target ({TARGET:.0%})")
    ax1.set_ylabel("Empirical Coverage"); ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=7.5, loc="lower right"); ax1.set_title("(a) Coverage by signal quality", fontsize=9)
    for bb, c in zip(b, sc): ax1.text(bb.get_x()+bb.get_width()/2, c+0.02, f"{c:.3f}", ha="center", fontsize=7.5)
    b2 = ax2.bar(strata, ss, color=cols, edgecolor="white", linewidth=0.6)
    ax2.set_ylabel("Average Prediction Set Size"); ax2.set_ylim(0, 3.6)
    ax2.set_title("(b) Set size by signal quality", fontsize=9)
    for bb, s in zip(b2, ss): ax2.text(bb.get_x()+bb.get_width()/2, s+0.06, f"{s:.2f}", ha="center", fontsize=7.5)
    plt.tight_layout(); plt.savefig(fd / "fig3_sqi_stratum.pdf", bbox_inches="tight"); plt.close()


def figA6(fd):
    gcov, cccov, gsz, csz = GLOBAL_COV, CC_COV, GLOBAL_SZ, CC_SZ
    loaded = _load_per_class(fd)
    if loaded:
        gcov, cccov, gsz, csz = loaded
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    short = {"Critical": "Crit", "Other": "Other", "Normal": "Norm",
             "Pacing/Block": "Pace", "AF/AFLT": "AF", "Tachy/Brady": "Tachy"}
    for i, cl in enumerate(CLASSES):
        emph = cl in ("Critical", "Other")
        ax.scatter(gsz[i], gcov[i], s=70 if emph else 50, marker="o", color=C_G,
                   edgecolor="black", linewidth=0.8 if emph else 0.4, zorder=4, alpha=0.9)
        ax.scatter(csz[i], cccov[i], s=70 if emph else 50, marker="s", color=C_C,
                   edgecolor="black", linewidth=0.8 if emph else 0.4, zorder=4, alpha=0.9)
        fw = "bold" if emph else "normal"
        ax.annotate(short[cl], (gsz[i], gcov[i]),
                    xytext=(gsz[i]+0.05, gcov[i]+0.01), fontsize=6.5, color=C_G, fontweight=fw)
        ax.annotate(short[cl], (csz[i], cccov[i]),
                    xytext=(csz[i]+0.05, cccov[i]+0.01), fontsize=6.5, color=C_C, fontweight=fw)
    ax.axhline(TARGET, color="black", ls="--", lw=0.9)
    ax.set_xlabel("Average Prediction Set Size"); ax.set_ylabel("Empirical Coverage")
    ax.set_ylim(0, 1.06); ax.set_xlim(1.5, 4.6)
    from matplotlib.lines import Line2D
    el = [Line2D([0], [0], marker="o", color="w", markerfacecolor=C_G, markersize=8,
                 label="Global CP", markeredgecolor="black", markeredgewidth=0.4),
          Line2D([0], [0], marker="s", color="w", markerfacecolor=C_C, markersize=8,
                 label="Class-Cond. CP", markeredgecolor="black", markeredgewidth=0.4)]
    ax.legend(handles=el, fontsize=7.5, loc="lower right")
    plt.tight_layout(); plt.savefig(fd / "fig2_coverage_efficiency.pdf", bbox_inches="tight"); plt.close()


def main(cfg):
    fd = Path(cfg.get("figure_dir", "./figs")); fd.mkdir(parents=True, exist_ok=True)
    fig1(fd); fig2(fd); fig3(fd); fig4(fd); fig5(fd); figA6(fd)
    print(f"Figures written to {fd}/ (fig1-fig5, figA6)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    main(load_config(ap.parse_args().config))
