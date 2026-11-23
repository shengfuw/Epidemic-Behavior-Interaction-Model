"""Seeded simulation experiments for EBIM, with network-structure metrics.

Usage: python run_experiments.py <experiment>
Experiments: sanity, pilot, replicates, psi_sens, delay, spotcheck, spotcheck_extended, timeseries, topology.
Each experiment writes CSVs to ../results/ and a log to logs/.
"""
import sys
import time
import random
import datetime
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from joblib import Parallel, delayed

from model import TwoLayerNetwork, fourier_analysis
from network_metrics import to_graph, snapshot, run_with_metrics, lagged_xcorr, nearest_peak

OUT = Path(__file__).resolve().parent.parent / "results"
LOGS = Path(__file__).resolve().parent / "logs"

# Baseline parameters (notebook cell 3 and Table 1 of the paper)
BASE = dict(N=1000, k=5, a=1.5, b=1.5, beta_u=0.8, beta_a=0.08, alpha=0.3, infection_period=50,
            gamma=0.7, eta=0.2, psi=0.01, lambda_=0.8, rho=0.5, network_type="ER",
            initial_infected_ratio=0.01, intial_aware_ratio=0.1, ws_p=0.1)
FORMULAS = ["complex", "simple", "simple_prevalence"]
ALPHA_GEOMETRIC = 1 / (49 + 1 / 0.3)  # same mean infectious duration (52.33 steps) with no minimum period


class Log:
    def __init__(self, name):
        LOGS.mkdir(exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = LOGS / f"{stamp}_{name}.log"
        self.f = open(self.path, "w")

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def make_model(formula, p):
    return TwoLayerNetwork(p["N"], p["k"], p["a"], p["b"], p["beta_a"], p["beta_u"], p["alpha"],
                           p["infection_period"], p["gamma"], p["eta"], formula, psi=p["psi"],
                           lambda_=p["lambda_"], rho=p["rho"], network_type=p["network_type"],
                           initial_infected_ratio=p["initial_infected_ratio"],
                           intial_aware_ratio=p["intial_aware_ratio"], ws_p=p["ws_p"])


def series_stats(y, window):
    """Same statistics as TwoLayerNetwork.get_statistics, plus the window mean.

    The dominant frequency is NaN for a flat window (the FFT would return the
    first bin, 0.001, by default).
    """
    seg = y[-window:]
    amp = seg.max() - seg.min()
    freq, fft_amp = fourier_analysis(seg, window)
    return dict(mean=seg.mean(), amp=amp, sd=seg.std(), freq=freq if amp > 0 else np.nan,
                fftamp=fft_amp, last10=y[-10:].mean())


def one_run(seed, formula, p, T, window, metrics=True, keep_series=False, tag=None):
    """One seeded run. Returns (summary dict, per-step DataFrame or None)."""
    random.seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    model = make_model(formula, p)
    init_G = to_graph(model)
    init = dict(k_init=2 * init_G.number_of_edges() / p["N"],
                lcc_init=max(len(c) for c in nx.connected_components(init_G)) / p["N"],
                C_init=nx.transitivity(init_G))
    if metrics:
        steps = run_with_metrics(model, T, record_from=T - window + 1)
    else:
        model.run_simulation(T, print_info=False, plot_network=False)
        steps = None
    I = np.array(model.infection_counts) / p["N"]
    A = np.array(model.awreness_counts) / p["N"]
    zeros = np.flatnonzero(I == 0)
    summ = dict(tag=tag, seed=seed, formula=formula, **{k: p[k] for k in ["beta_a", "psi", "lambda_", "rho", "gamma", "eta", "alpha", "infection_period", "network_type", "k"]},
                T=T, window=window, extinct=bool(len(zeros)), t_extinct=int(zeros[0]) if len(zeros) else np.nan,
                I_zero_in_window=bool((I[-window:] == 0).any()), **init)
    summ.update({f"I_{k}": v for k, v in series_stats(I, window).items()})
    summ.update({f"A_{k}": v for k, v in series_stats(A, window).items()})
    if steps is not None:
        for m in ["r_deg", "C", "r_adopt", "isolates"]:
            summ[f"{m}_mean"] = steps[m].mean()
        summ["r_deg_undef"] = steps["r_deg"].isna().mean()
        summ["r_adopt_undef"] = steps["r_adopt"].isna().mean()
        for col, v in [("tag", tag), ("seed", seed), ("formula", formula), ("beta_a", p["beta_a"])]:
            steps[col] = v
    elif keep_series:
        steps = pd.DataFrame(dict(t=np.arange(T + 1), I=I, A=A, tag=tag, seed=seed, formula=formula, beta_a=p["beta_a"]))
    summ["runtime_s"] = round(time.time() - t0, 1)
    return summ, steps


def run_batch(tasks, log, name, n_jobs=-1):
    """tasks: list of kwargs for one_run. Saves {name}_summary.csv and {name}_steps.csv.gz."""
    log(f"{name}: {len(tasks)} runs")
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs)(delayed(one_run)(**kw) for kw in tasks)
    summary = pd.DataFrame([r[0] for r in res])
    summary.to_csv(OUT / f"{name}_summary.csv", index=False)
    steps = [r[1] for r in res if r[1] is not None]
    if steps:
        pd.concat(steps, ignore_index=True).to_csv(OUT / f"{name}_steps.csv.gz", index=False)
    log(f"{name}: done in {time.time() - t0:.0f} s; saved to {OUT}")
    return summary


def describe(summary, log, by, cols):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    log(summary.groupby(by)[cols].agg(["mean", "std"]).round(4).to_string())


# ---------------------------------------------------------------- experiments

def sanity(log):
    """Checks of the metric code and model options before any experiment."""
    # 1. Metrics on hand-built graphs
    class FakeNode:
        def __init__(self, i):
            self.id, self.state_disease, self.state_awareness, self.physical_neighbors = i, 0, 0, []

    class FakeModel:
        def __init__(self, n, edges, adopt):
            self.nodes = [FakeNode(i) for i in range(n)]
            for u, v in edges:
                self.nodes[u].physical_neighbors.append(self.nodes[v])
                self.nodes[v].physical_neighbors.append(self.nodes[u])
            for i in adopt:
                self.nodes[i].state_awareness = 1

    star = FakeModel(6, [(0, i) for i in range(1, 6)], adopt=[0])
    s = snapshot(star)
    log("star: r_deg", s["r_deg"], "(expect -1), C", s["C"], "(expect 0), r_adopt", s["r_adopt"], "(expect -1)")
    assert np.isclose(s["r_deg"], -1) and s["C"] == 0 and np.isclose(s["r_adopt"], -1)

    cliques = [(i, j) for i in range(4) for j in range(i + 1, 4)] + [(i, j) for i in range(4, 8) for j in range(i + 1, 8)] + [(3, 4)]
    two = FakeModel(9, cliques, adopt=[0, 1, 2, 3])  # node 8 is an isolate
    s = snapshot(two)
    G = nx.Graph(cliques)
    nx.set_node_attributes(G, {i: int(i < 4) for i in G}, "adopt")
    log("two cliques: r_adopt", round(s["r_adopt"], 4), "nx", round(nx.attribute_assortativity_coefficient(G, "adopt"), 4),
        "| C", round(s["C"], 4), "nx", round(nx.transitivity(G), 4), "| isolates", s["isolates"])
    assert np.isclose(s["r_adopt"], nx.attribute_assortativity_coefficient(G, "adopt")) and s["isolates"] == 1

    ring = FakeModel(6, [(i, (i + 1) % 6) for i in range(6)], adopt=range(6))
    s = snapshot(ring)
    log("ring, all adopt: r_deg", s["r_deg"], "r_adopt", s["r_adopt"], "(expect nan, nan)")
    assert np.isnan(s["r_deg"]) and np.isnan(s["r_adopt"])

    # 2. snapshot agrees with the model's own graph on a real run
    random.seed(1); np.random.seed(1)
    m = make_model("complex", dict(BASE, N=200))
    m.run_simulation(50, print_info=False, plot_network=False)
    s, G = snapshot(m), m.get_network("physical")
    log("real run: r_deg", round(s["r_deg"], 4), "nx", round(nx.degree_assortativity_coefficient(G), 4),
        "| C", round(s["C"], 4), "nx", round(nx.transitivity(G), 4))
    assert np.isclose(s["r_deg"], nx.degree_assortativity_coefficient(G)) and np.isclose(s["C"], nx.transitivity(G))

    # 3. simple_prevalence with lambda = 0 equals simple under the same seed
    out = []
    for f in ["simple", "simple_prevalence"]:
        random.seed(7); np.random.seed(7)
        m = make_model(f, dict(BASE, N=300, lambda_=0.0))
        m.run_simulation(300, print_info=False, plot_network=False)
        out.append((m.infection_counts, m.awreness_counts))
    log("simple_prevalence(lambda=0) == simple:", out[0] == out[1])
    assert out[0] == out[1]

    # 4. Same seed in separate worker processes gives identical runs (rewiring sort fix)
    res = Parallel(n_jobs=2)(delayed(one_run)(111, "complex", dict(BASE, N=300), 400, 100, False, True) for _ in range(2))
    same = res[0][1][["I", "A"]].equals(res[1][1][["I", "A"]])
    log("same seed, two processes, identical I and A:", same)
    assert same

    # 5. lagged_xcorr recovers a known lag: x(t) = y(t - 20), so x lags y by 20 steps
    t = np.arange(1000)
    y = np.sin(2 * np.pi * t / 100)
    x = np.sin(2 * np.pi * (t - 20) / 100)
    lags, ccf = lagged_xcorr(x, y, 100)
    tau, val = nearest_peak(lags, ccf, 100)
    log("synthetic lag: tau", tau, "corr", round(val, 3), "(expect 20, ~1)")
    assert tau == 20
    log("all sanity checks passed")


def pilot(log):
    tasks = [dict(seed=s, formula=f, p=BASE, T=3000, window=1000) for s in [1, 2, 3] for f in FORMULAS]
    summ = run_batch(tasks, log, "pilot")
    log(summ[["seed", "formula", "runtime_s", "I_mean", "I_amp", "A_mean", "A_amp", "I_freq", "extinct",
              "r_deg_mean", "C_mean", "r_adopt_mean", "r_deg_undef", "r_adopt_undef"]].round(4).to_string())
    steps = pd.read_csv(OUT / "pilot_steps.csv.gz")
    log(steps.groupby("formula")[["r_deg", "C", "r_adopt"]].describe().T.round(4).to_string())


def replicates(log, M=50):
    tasks = [dict(seed=s, formula=f, p=dict(BASE, beta_a=ba), T=3000, window=1000, tag=f"beta_a={ba}")
             for ba in [0.08, 0.16] for f in FORMULAS for s in range(1, M + 1)]
    summ = run_batch(tasks, log, "replicates")
    describe(summ, log, ["beta_a", "formula"], ["I_mean", "I_amp", "I_sd", "A_mean", "A_amp", "A_sd", "I_freq",
                                                 "extinct", "r_deg_mean", "C_mean", "r_adopt_mean", "r_adopt_undef"])


def psi_sens(log, M=20):
    tasks = [dict(seed=s, formula=f, p=dict(BASE, psi=psi), T=3000, window=1000, tag=f"psi={psi}")
             for psi in [0.005, 0.02] for f in ["simple", "simple_prevalence"] for s in range(1, M + 1)]
    summ = run_batch(tasks, log, "psi_sens")
    describe(summ, log, ["psi", "formula"], ["I_mean", "I_amp", "A_mean", "A_amp", "extinct", "r_adopt_mean"])


def delay(log, M=20):
    p = dict(BASE, infection_period=0, alpha=ALPHA_GEOMETRIC)
    log(f"geometric recovery: infection_period=0, alpha={ALPHA_GEOMETRIC:.5f} (mean duration {1 / ALPHA_GEOMETRIC:.2f})")
    tasks = [dict(seed=s, formula=f, p=dict(p, beta_a=ba), T=3000, window=1000, metrics=False, keep_series=True, tag=f"beta_a={ba}")
             for ba in [0.08, 0.16] for f in FORMULAS for s in range(1, M + 1)]
    summ = run_batch(tasks, log, "delay")
    describe(summ, log, ["beta_a", "formula"], ["I_mean", "I_amp", "I_sd", "A_mean", "A_amp", "A_sd", "I_freq", "extinct"])


def spotcheck(log, M=10):
    """Re-run exact grid cells of the saved sweeps (T = 2000, 10 runs) and compare with the CSV means."""
    grid = np.linspace(0, 1, 30)
    cells = [("lambdaRho", 23, 15), ("lambdaRho", 6, 6), ("lambdaRho", 23, 26),  # (sweep, x index, y index)
             ("gammaEta", 21, 6), ("gammaEta", 0, 6)]
    tasks, keys = [], []
    for sweep, xi, yi in cells:
        if sweep == "lambdaRho":
            p = dict(BASE, lambda_=grid[xi], rho=grid[yi])
        else:
            p = dict(BASE, gamma=grid[xi], eta=grid[yi])
        for s in range(1, M + 1):
            tasks.append(dict(seed=s, formula="complex", p=p, T=2000, window=1000, metrics=False, tag=f"{sweep}:{xi}:{yi}"))
    summ = run_batch(tasks, log, "spotcheck")
    rng = np.random.default_rng(0)
    rows = []
    for sweep, xi, yi in cells:
        sub = summ[summ["tag"] == f"{sweep}:{xi}:{yi}"]
        for csv_name, col in [("amp", "I_amp"), ("sd", "I_sd"), ("final_infect", "I_last10"), ("final_aware", "A_last10")]:
            saved = pd.read_csv(OUT / f"{sweep}_{csv_name}.csv", index_col=0).iloc[yi, xi]
            v = sub[col].to_numpy()
            boots = [rng.choice(v, len(v)).mean() for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append(dict(sweep=sweep, x=round(grid[xi], 4), y=round(grid[yi], 4), stat=csv_name, saved=round(saved, 4),
                             new_mean=round(v.mean(), 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4), inside=lo <= saved <= hi))
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "spotcheck_comparison.csv", index=False)
    log(comp.to_string())


def spotcheck_extended(log, M=50):
    """Two sweep cells with 50 seeded runs each (seeds 101-150, T = 2000).

    Reports where each saved 10-run mean falls among 10-run means drawn without
    replacement from the 50 new runs.
    """
    grid = np.linspace(0, 1, 30)
    cells = [("lambdaRho", 23, 15, dict(BASE, lambda_=grid[23], rho=grid[15])),
             ("gammaEta", 0, 6, dict(BASE, gamma=grid[0], eta=grid[6]))]
    tasks = [dict(seed=s, formula="complex", p=p, T=2000, window=1000, metrics=False, tag=f"{sw}:{xi}:{yi}")
             for sw, xi, yi, p in cells for s in range(101, 101 + M)]
    summ = run_batch(tasks, log, "spotcheck_extended")
    rng = np.random.default_rng(3)
    m = {"amp": "I_amp", "sd": "I_sd", "final_infect": "I_last10", "final_aware": "A_last10"}
    rows = []
    for sw, xi, yi, _ in cells:
        sub = summ[summ.tag == f"{sw}:{xi}:{yi}"]
        for csv, col in m.items():
            saved = pd.read_csv(OUT / f"{sw}_{csv}.csv", index_col=0).iloc[yi, xi]
            v = sub[col].to_numpy()
            means10 = np.array([rng.choice(v, 10, replace=False).mean() for _ in range(5000)])
            pct = (means10 < saved).mean() * 100
            rows.append(dict(sweep=sw, x=round(grid[xi], 4), y=round(grid[yi], 4), stat=csv, saved=round(saved, 4),
                             mean50=round(v.mean(), 4), p2_5=round(np.percentile(means10, 2.5), 4),
                             p97_5=round(np.percentile(means10, 97.5), 4), saved_percentile=round(pct, 1)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "spotcheck_extended_comparison.csv", index=False)
    log("Where the saved 10-run mean falls among 10-run means drawn from 50 new seeded runs:")
    log(out.to_string())


def timeseries(log, seed=111):
    summ, steps = one_run(seed, "complex", BASE, T=3000, window=1000, tag="timeseries")
    steps.to_csv(OUT / "timeseries_seed111.csv", index=False)
    log(pd.Series(summ).to_string())


def topology(log, M=20):
    tasks = []
    for k in [4, 6]:
        for net in ["ER", "WS", "BA"]:
            p = dict(BASE, k=k, network_type=net)
            for s in [111] + list(range(1, M + 1)):
                tasks.append(dict(seed=s, formula="complex", p=p, T=3000, window=1000, metrics=False, keep_series=(s == 111), tag=f"{net}_k{k}"))
    summ = run_batch(tasks, log, "topology")
    rep = summ[summ["seed"] != 111]
    describe(rep, log, ["k", "network_type"], ["k_init", "lcc_init", "C_init", "I_mean", "I_amp", "I_sd", "A_mean", "A_amp", "extinct"])


EXPERIMENTS = dict(sanity=sanity, pilot=pilot, replicates=replicates, psi_sens=psi_sens, delay=delay,
                   spotcheck=spotcheck, spotcheck_extended=spotcheck_extended, timeseries=timeseries, topology=topology)

if __name__ == "__main__":
    name = sys.argv[1]
    log = Log(name)
    log(f"experiment: {name}; started {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"baseline: {BASE}")
    EXPERIMENTS[name](log)
    log(f"log file: {log.path}")
