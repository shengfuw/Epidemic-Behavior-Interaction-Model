"""Network-structure measures for EBIM runs.

The measures are recorded outside the model: `run_with_metrics` calls
`TwoLayerNetwork.update_one_round()` and takes a snapshot of the physical
layer after each step, so the model's update logic is unchanged.
"""
import numpy as np
import pandas as pd
import networkx as nx
from scipy.signal import find_peaks


def to_graph(model):
    """Physical layer as an undirected networkx graph with an 'adopt' node attribute."""
    G = nx.Graph()
    G.add_nodes_from((node.id, {"adopt": node.state_awareness}) for node in model.nodes)
    G.add_edges_from((node.id, nb.id) for node in model.nodes for nb in node.physical_neighbors if node.id < nb.id)
    return G


def snapshot(model):
    """Prevalence, adoption, and structure of the physical layer at the current step.

    Assortativity is NaN (undefined) when there are no edges, when all edge
    endpoints have the same degree, or when all edge endpoints share one
    adoption state. Undefined values are never replaced by zero.
    """
    G = to_graph(model)
    n = G.number_of_nodes()
    edges = np.array(G.edges(), dtype=int).reshape(-1, 2)
    deg = dict(G.degree())
    adopt = nx.get_node_attributes(G, "adopt")

    if len(edges) == 0:
        r_deg, r_adopt = np.nan, np.nan
    else:
        end_deg = np.array([deg[u] for u in edges.ravel()])
        end_adopt = np.array([adopt[u] for u in edges.ravel()])
        r_deg = nx.degree_assortativity_coefficient(G) if end_deg.min() != end_deg.max() else np.nan
        r_adopt = nx.attribute_assortativity_coefficient(G, "adopt") if end_adopt.min() != end_adopt.max() else np.nan

    return {
        "I": sum(node.state_disease == 1 for node in model.nodes) / n,
        "A": sum(node.state_awareness == 1 for node in model.nodes) / n,
        "r_deg": r_deg,
        "C": nx.transitivity(G),
        "r_adopt": r_adopt,
        "isolates": sum(1 for d in deg.values() if d == 0),
    }


def run_with_metrics(model, num_rounds, record_from=1):
    """Run the model for `num_rounds` steps and record a snapshot at every step t >= record_from."""
    records = []
    for _ in range(num_rounds):
        model.update_one_round()
        if model.round >= record_from:
            rec = snapshot(model)
            rec["t"] = model.round
            records.append(rec)
    return pd.DataFrame(records)


def _detrend(z):
    """Remove a least-squares line from z, using only its finite values."""
    z = np.asarray(z, dtype=float)
    t = np.arange(len(z))
    ok = np.isfinite(z)
    out = np.full_like(z, np.nan)
    if ok.sum() >= 2:
        slope, intercept = np.polyfit(t[ok], z[ok], 1)
        out[ok] = z[ok] - (slope * t[ok] + intercept)
    return out


def lagged_xcorr(x, y, max_lag, min_pairs=100):
    """Pearson correlation between x(t + tau) and y(t) for tau = -max_lag..max_lag.

    Positive tau means x lags y (x moves after y). Both series are linearly
    detrended first. Each lag uses only time points where both values are
    defined; lags with fewer than `min_pairs` such points, or with a constant
    series, return NaN.
    """
    x, y = _detrend(x), _detrend(y)
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    ccf = np.full(len(lags), np.nan)
    for i, tau in enumerate(lags):
        if tau >= 0:
            xs, ys = x[tau:], y[:n - tau]
        else:
            xs, ys = x[:n + tau], y[-tau:]
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() >= min_pairs and xs[ok].std() > 0 and ys[ok].std() > 0:
            ccf[i] = np.corrcoef(xs[ok], ys[ok])[0, 1]
    return lags, ccf


def nearest_peak(lags, ccf, max_abs_lag, prominence=0.1):
    """Signed local peak or trough of the CCF closest to tau = 0.

    Only extrema with |tau| <= max_abs_lag and the given prominence count.
    Returns (tau, value) or (nan, nan) when there is no clear peak.
    """
    ccf = np.asarray(ccf, dtype=float)
    filled = np.where(np.isfinite(ccf), ccf, 0.0)
    peaks, _ = find_peaks(filled, prominence=prominence)
    troughs, _ = find_peaks(-filled, prominence=prominence)
    cand = [i for i in np.concatenate([peaks, troughs]) if abs(lags[i]) <= max_abs_lag and np.isfinite(ccf[i])]
    if not cand:
        return np.nan, np.nan
    best = min(cand, key=lambda i: (abs(lags[i]), -abs(ccf[i])))
    return int(lags[best]), float(ccf[best])
