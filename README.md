# Epidemic-Behavior Interaction Model (EBIM)

EBIM is an agent-based model of an SIS epidemic and a protective behavior that spread together on an adaptive network. Susceptible agents cut ties to infected neighbors and rewire. Behavior spreads either by threshold-based (complex) contagion, whose thresholds fall as prevalence rises, or by simple contagion.

## Files

| File | Purpose |
|---|---|
| `model.py` | The model: `Node`, `TwoLayerNetwork`, and `fourier_analysis`. |
| `main.ipynb` | Interactive runs and the two 30 × 30 parameter sweeps (λ × ρ and γ × η). Cell 1 is a copy of `model.py`, so a change to the model must be made in both places. |
| `network_metrics.py` | Network measures for each step (degree assortativity, transitivity, adoption assortativity) and the lagged cross-correlation. |
| `run_experiments.py` | Seeded experiments, one command each. Writes CSVs to `../results/` and a log to `logs/`. |
| `make_figures.py` | Draws the figures from the CSVs in `../results/`. Writes PDF and 300 dpi PNG to `../figures/`. |

## Setup

```sh
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install jupyter numpy pandas networkx matplotlib scipy tqdm joblib statsmodels python-dateutil PyWavelets
```

## Run

```sh
python run_experiments.py sanity        # checks of the metric code; run this first
python run_experiments.py replicates    # 50 seeds x 3 behavior models x 2 protection levels (about 20 min on 10 cores)
python make_figures.py all
```

The experiments are `sanity`, `pilot`, `replicates`, `psi_sens`, `delay`, `spotcheck`, `spotcheck_extended`, `timeseries`, and `topology`. The figures are `timeseries`, `assortativity`, `xcorr`, `topology`, `heatmaps`, and `table2`. Every run sets both `random` and `np.random` from its seed, so results are reproducible. The parameter-sweep CSVs (`{lambdaRho,gammaEta}_*.csv`) come from `main.ipynb`. Their runs are not seeded.
