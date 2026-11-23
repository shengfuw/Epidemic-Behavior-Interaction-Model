"""EBIM figures, drawn from CSVs written by run_experiments.py and the parameter sweeps in main.ipynb.

Usage: python make_figures.py <figure> [<figure> ...]
Figures: timeseries, assortativity, xcorr, topology, heatmaps, table2, all.
Each figure is saved as PDF and 300 dpi PNG in ../figures/.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

import model  # noqa: F401  (sets the shared rcParams used by all EBIM figures)
from network_metrics import lagged_xcorr, nearest_peak

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIG = HERE.parent / "figures"

LABEL = {"complex": "Complex", "simple": "Simple", "simple_prevalence": "Simple +\nprevalence"}
FORMULAS = ["complex", "simple", "simple_prevalence"]
METRICS = [("r_deg", "Degree assortativity", "C2", "-"), ("C", "Transitivity", "C3", "--"),
           ("r_adopt", "Adoption assortativity", "C4", ":")]
SHORT = {"r_deg": "Degree\nassortativity", "C": "Global\ntransitivity", "r_adopt": "Adoption\nassortativity"}


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", FIG / f"{name}.pdf", "and .png")


def panel_label(ax, s, x=-0.08, y=1.04):
    ax.text(x, y, f"({s})", transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="right")


# ------------------------------------------------------------------ Network structure over time

def fig_timeseries():
    d = pd.read_csv(RES / "timeseries_seed111.csv").sort_values("t")
    peaks, _ = find_peaks(d["A"].to_numpy(), prominence=0.2)
    troughs, _ = find_peaks(-d["I"].to_numpy(), prominence=0.2)
    fig, axs = plt.subplots(4, 1, figsize=(8, 9.5), sharex=True, gridspec_kw={"height_ratios": [1.3, 1, 1, 1]})
    axs[0].plot(d["t"], d["I"], color="C0", lw=1.6, label=r"$I(t)$: infected")
    axs[0].plot(d["t"], d["A"], color="C1", lw=1.6, ls="--", label=r"$A(t)$: adopting")
    axs[0].set_ylabel("Proportion of agents")
    axs[0].set_ylim(-0.02, 1.02)
    for ax, (col, name, color, ls) in zip(axs[1:], METRICS):
        ax.plot(d["t"], d[col], color=color, lw=1.4, ls=ls)
        ax.set_ylabel(SHORT[col])
    for ax, s in zip(axs, "abcd"):
        for p in troughs:
            ax.axvline(d["t"].iloc[p], color="C0", lw=0.8, ls=(0, (4, 2, 1, 2)), alpha=0.6, zorder=0)
        for p in peaks:
            ax.axvline(d["t"].iloc[p], color="C1", lw=0.8, ls=(0, (1, 2)), alpha=0.8, zorder=0)
        ax.grid(alpha=0.25)
        panel_label(ax, s, x=-0.15, y=0.98)
    axs[0].plot([], [], color="C0", lw=0.8, ls=(0, (4, 2, 1, 2)), label="Infection trough")
    axs[0].plot([], [], color="C1", lw=0.8, ls=(0, (1, 2)), label="Adoption peak")
    axs[0].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=10, handlelength=2.2, columnspacing=1.2)
    axs[-1].set_xlabel(r"Time step $t$")
    axs[-1].set_xlim(d["t"].min(), d["t"].max())
    fig.align_ylabels(axs)
    save(fig, "structure_timeseries")
    print(f"adoption peaks marked: {len(peaks)}; infection troughs marked: {len(troughs)}; t range {d['t'].min()}-{d['t'].max()}")
    print(d[["I", "A", "r_deg", "C", "r_adopt"]].describe().round(4).to_string())


# ------------------------------------------------------------------ Adoption assortativity by behavior model

def fig_assortativity(prefix="replicates"):
    s = pd.read_csv(RES / f"{prefix}_summary.csv")
    levels = sorted(s["beta_a"].unique())
    fig, axs = plt.subplots(1, len(levels), figsize=(4.6 * len(levels), 4.6), sharey=True, squeeze=False)
    rng = np.random.default_rng(1)
    for ax, ba, lab in zip(axs[0], levels, "ab"):
        sub = s[s["beta_a"] == ba]
        data = [sub.loc[sub["formula"] == f, "r_adopt_mean"].dropna().to_numpy() for f in FORMULAS]
        vp = ax.violinplot(data, positions=range(3), widths=0.8, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor("0.85")
            body.set_edgecolor("0.5")
            body.set_alpha(1)
        ax.boxplot(data, positions=range(3), widths=0.22, showfliers=False, medianprops=dict(color="black", lw=1.5))
        for i, f in enumerate(FORMULAS):
            g = sub[sub["formula"] == f]
            x = i + rng.uniform(-0.12, 0.12, len(g))
            ext = g["extinct"].to_numpy()
            ax.scatter(x[~ext], g["r_adopt_mean"][~ext], s=14, color="C0", zorder=3, label="Disease persists" if i == 0 else None)
            ax.scatter(x[ext], g["r_adopt_mean"][ext], s=16, facecolor="white", edgecolor="C3", marker="^", zorder=3, label="Disease died out" if i == 0 else None)
        ax.axhline(0, color="0.4", lw=0.8, ls="--")
        ax.set_xticks(range(3), [LABEL[f] for f in FORMULAS])
        ax.set_title(rf"$\beta_a = {ba:g}$ ($\beta_n = 0.8$)", fontsize=13)
        ax.grid(axis="y", alpha=0.25)
        panel_label(ax, lab, x=-0.1)
    axs[0][0].set_ylabel("Adoption assortativity $r_A$\n(mean over last 1,000 steps)")
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=11)
    save(fig, "adoption_assortativity")
    # numbers for the text
    rows = []
    for ba in levels:
        for f in FORMULAS:
            g = s[(s["beta_a"] == ba) & (s["formula"] == f)]
            surv = g[~g["extinct"]]
            rows.append(dict(beta_a=ba, formula=f, n=len(g), n_defined=int(g["r_adopt_mean"].notna().sum()), n_extinct=int(g["extinct"].sum()),
                             median_rA=g["r_adopt_mean"].median(), mean_rA=g["r_adopt_mean"].mean(),
                             median_rA_survivors=surv["r_adopt_mean"].median() if len(surv) else np.nan,
                             mean_A=g["A_mean"].mean(), undefined_share=g["r_adopt_undef"].mean()))
    tab = pd.DataFrame(rows)
    for ba in levels:  # median difference complex - other, bootstrap 95% CI over runs
        c = s[(s["beta_a"] == ba) & (s["formula"] == "complex")]["r_adopt_mean"].dropna().to_numpy()
        for f in FORMULAS[1:]:
            o = s[(s["beta_a"] == ba) & (s["formula"] == f)]["r_adopt_mean"].dropna().to_numpy()
            print(f"beta_a={ba}: runs with defined r_A: complex {len(c)}, {f} {len(o)}")
            boots = [np.median(rng.choice(c, len(c))) - np.median(rng.choice(o, len(o))) for _ in range(4000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"beta_a={ba}: median r_A complex - {f} = {np.median(c) - np.median(o):.3f} (95% CI {lo:.3f}, {hi:.3f})")
    tab.to_csv(RES / f"{prefix}_assortativity_table.csv", index=False)
    print(tab.round(4).to_string())


# ------------------------------------------------------------------ Lagged cross-correlation

def run_ccfs(steps, max_lag):
    out = {m: [] for m in ["r_deg", "C", "r_adopt", "A"]}
    for _, g in steps.groupby("seed"):
        g = g.sort_values("t")
        for m in out:
            out[m].append(lagged_xcorr(g[m].to_numpy(), g["I"].to_numpy(), max_lag)[1])
    return {m: np.array(v) for m, v in out.items()}


def xcorr_data(prefix="replicates", beta_a=0.08, max_lag=300):
    """Run-averaged CCFs (with bootstrap bands) and nearest-zero peaks for complex-contagion runs."""
    s = pd.read_csv(RES / f"{prefix}_summary.csv")
    keep = s[(s["formula"] == "complex") & (s["beta_a"] == beta_a) & (~s["I_zero_in_window"])]
    steps = pd.read_csv(RES / f"{prefix}_steps.csv.gz")
    steps = steps[(steps["formula"] == "complex") & (steps["beta_a"] == beta_a) & steps["seed"].isin(keep["seed"])]
    n_all = ((s["formula"] == "complex") & (s["beta_a"] == beta_a)).sum()
    bound = int(min(max_lag, np.nanmedian(1 / keep["I_freq"])))
    print(f"complex, beta_a={beta_a}: {len(keep)} of {n_all} runs used (others hit I = 0 in the window); peak search |tau| <= {bound}")
    lags = np.arange(-max_lag, max_lag + 1)
    ccfs = run_ccfs(steps, max_lag)
    rng = np.random.default_rng(2)
    out, rows = {}, []
    for m in ["r_deg", "C", "r_adopt", "A"]:
        X = ccfs[m]
        mean = np.nanmean(X, axis=0)
        boots = np.array([np.nanmean(X[rng.integers(0, len(X), len(X))], axis=0) for _ in range(2000)])
        lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
        tau, val = nearest_peak(lags, mean, bound)
        run_taus = np.array([nearest_peak(lags, x, bound)[0] for x in X], dtype=float)
        out[m] = dict(mean=mean, lo=lo, hi=hi, tau=tau, val=val)
        rows.append(dict(beta_a=beta_a, metric=m, peak_tau=tau, peak_r=val, r_at_0=mean[max_lag], runs=len(X), runs_total=n_all,
                         search_bound=bound, run_tau_median=np.nanmedian(run_taus), run_tau_q25=np.nanpercentile(run_taus, 25),
                         run_tau_q75=np.nanpercentile(run_taus, 75), runs_no_peak=int(np.isnan(run_taus).sum())))
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / f"{prefix}_xcorr_peaks_beta{beta_a}.csv", index=False)
    print(tab.round(3).to_string())
    return lags, bound, out


def fig_xcorr(prefix="replicates", levels=(0.08, 0.16), max_lag=300):
    data = {ba: xcorr_data(prefix, ba, max_lag) for ba in levels}
    for xlim, name in [(100, "lagged_xcorr"), (max_lag, "lagged_xcorr_full")]:
        fig, axs = plt.subplots(1, len(levels), figsize=(6.2 * len(levels), 4.8), sharey=True, squeeze=False)
        for ax, ba, lab in zip(axs[0], levels, "ab"):
            lags, bound, out = data[ba]
            ax.axhline(0, color="0.4", lw=0.8)
            ax.axvline(0, color="0.4", lw=0.8)
            ax.axvspan(-max_lag, -bound, color="0.95", zorder=0)
            ax.axvspan(bound, max_lag, color="0.95", zorder=0)
            for m, mname, color, ls in METRICS:
                o = out[m]
                ax.fill_between(lags, o["lo"], o["hi"], color=color, alpha=0.18, lw=0)
                ax.plot(lags, o["mean"], color=color, ls=ls, lw=1.8, label=f"{mname} vs $I$")
                if np.isfinite(o["tau"]):
                    ax.plot(o["tau"], o["val"], "o", color=color, ms=6, mec="black", mew=0.6, zorder=4)
                    ax.annotate(rf"$\tau = {o['tau']}$", (o["tau"], o["val"]), xytext=(10, -4), textcoords="offset points",
                                fontsize=11, color=color, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            ax.set_xlim(-xlim, xlim)
            ax.set_ylim(-1, 1)
            ax.set_title(rf"$\beta_a = {ba:g}$ ($\beta_n = 0.8$)", fontsize=13)
            ax.set_xlabel(r"Lag $\tau$ (time steps); $\tau > 0$: structure lags infection")
            ax.grid(alpha=0.25)
            panel_label(ax, lab)
        axs[0][0].set_ylabel(r"Correlation $r(X_{t+\tau}, I_t)$")
        axs[0][0].legend(frameon=False, loc="lower left", fontsize=11)
        save(fig, name)


# ------------------------------------------------------------------ Network types

def fig_topology():
    s = pd.read_csv(RES / "topology_summary.csv")
    steps = pd.read_csv(RES / "topology_steps.csv.gz")
    nets, ks = ["ER", "WS", "BA"], [4, 6]
    full = {"ER": "Erdős–Rényi (ER)", "WS": "Watts–Strogatz (WS)", "BA": "Barabási–Albert (BA)"}
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for r, k in enumerate(ks):
        for c, net in enumerate(nets):
            ax = axs[r, c]
            g = steps[steps["tag"] == f"{net}_k{k}"].sort_values("t")
            info = s[(s["tag"] == f"{net}_k{k}") & (s["seed"] == 111)].iloc[0]
            ax.plot(g["t"], g["I"], color="C0", lw=1.4, label=r"$I$: Infected")
            ax.plot(g["t"], g["A"], color="C1", lw=1.4, ls="--", label=r"$A$: Adopting")
            ax.set_title(f"{full[net]}, " + rf"$\langle k \rangle = {info['k_init']:.2f}$", fontsize=13)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.25)
            panel_label(ax, "abcdef"[r * 3 + c])
            if c == 0:
                ax.set_ylabel("Proportion of agents")
            if r == 1:
                ax.set_xlabel(r"Time step $t$")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.955), ncol=2, fontsize=12)
    save(fig, "network_type_comparison")
    rep = s[s["seed"] != 111]
    print(rep.groupby(["k", "network_type"])[["k_init", "lcc_init", "C_init", "I_mean", "I_amp", "I_sd", "A_mean", "A_amp", "extinct"]].mean().round(3).to_string())


# ------------------------------------------------------------------ Parameter-sweep heatmaps

def smooth(data, sigma, mask=None):
    """Gaussian smoothing; masked (undefined) cells are left out and shown as NaN."""
    if mask is None:
        return gaussian_filter(data, sigma=sigma)
    w = gaussian_filter((~mask).astype(float), sigma=sigma)
    v = gaussian_filter(np.where(mask, 0, data), sigma=sigma)
    out = v / np.where(w > 0, w, np.nan)
    out[mask] = np.nan
    return out


def plot_heatmap(ax, fig, data, title, xlabel, ylabel, label, cmap, sigma, mask=None):
    z = data if sigma == 0 else smooth(data, sigma, mask)
    if sigma == 0 and mask is not None:
        z = np.where(mask, np.nan, data)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("0.75")
    im = ax.imshow(z, extent=[0, 1, 0, 1], origin="lower", aspect="auto", cmap=cm)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, rotation=270, labelpad=24)
    ax.set_xlabel(rf"${xlabel}$")
    ax.set_ylabel(rf"${ylabel}$")
    ax.set_title(title, fontsize=13)


def fig_heatmaps():
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12})
    for sweep, xl, yl in [("lambdaRho", "\\lambda", "\\rho"), ("gammaEta", "\\gamma", "\\eta")]:
        d = {n: pd.read_csv(RES / f"{sweep}_{n}.csv", index_col=0).to_numpy() for n in ["amp", "sd", "seq_freq", "seq_amp", "final_infect", "final_aware"]}
        flat = (d["amp"] == 0)  # flat trajectories: the dominant frequency is undefined
        print(f"{sweep}: {flat.sum()} flat cells masked in the frequency panel")
        for sigma, suffix in [(1.5, ""), (0, "_unsmoothed")]:
            fig, axs = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True)
            axs = axs.flatten()
            specs = [("amp", "Amplitude of Infection Trend", "Amplitude", "inferno", None),
                     ("sd", "Std. Deviation of Infection Trend", "Std. Dev", "inferno", None),
                     ("seq_freq", "Dom. Frequency of Infection Trend", "Dom. Freq", "inferno", flat),
                     ("seq_amp", "Dom. Amplitude of Infection Trend", "Dom. Amp", "inferno", None),
                     ("final_infect", "Infection Rate", r"$I$", "viridis", None),
                     ("final_aware", "Adoption Rate", r"$A$", "viridis", None)]
            for ax, (n, title, lab, cmap, mask), s in zip(axs, specs, "abcdef"):
                plot_heatmap(ax, fig, d[n], title, xl, yl, lab, cmap, sigma, mask)
                ax.text(-0.2, 1.1, f"({s})", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="right")
            save(fig, f"{sweep}_heatmaps{suffix}")


# ------------------------------------------------------------------ Replicate table

def table2():
    rows = []
    for name, label in [("replicates", "fixed minimum period"), ("delay", "geometric recovery")]:
        s = pd.read_csv(RES / f"{name}_summary.csv")
        for (ba, f), g in s.groupby(["beta_a", "formula"]):
            rows.append(dict(recovery=label, beta_a=ba, formula=f, runs=len(g), extinct=int(g["extinct"].sum()),
                             I_mean=g["I_mean"].mean(), I_amp=g["I_amp"].mean(), I_amp_sd=g["I_amp"].std(),
                             I_sd=g["I_sd"].mean(), A_mean=g["A_mean"].mean(), A_amp=g["A_amp"].mean(),
                             A_amp_sd=g["A_amp"].std(), A_sd=g["A_sd"].mean(),
                             I_period_median=np.nanmedian(1 / g["I_freq"])))
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "table2.csv", index=False)
    print(tab.round(3).to_string())


FIGS = dict(timeseries=fig_timeseries, assortativity=fig_assortativity, xcorr=fig_xcorr,
            topology=fig_topology, heatmaps=fig_heatmaps, table2=table2)

if __name__ == "__main__":
    names = sys.argv[1:] or ["all"]
    for n in (FIGS if names == ["all"] else names):
        print(f"== {n}")
        FIGS[n]()
