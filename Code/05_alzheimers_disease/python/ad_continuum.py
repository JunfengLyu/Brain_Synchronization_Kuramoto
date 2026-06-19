from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

sys.path.append(str(Path(__file__).resolve().parents[2] / "common" / "python"))
from kuramoto_style import COLORS, DATA_DIR, clean_axes, panel_label, save_figure, smooth_transition


GROUPS = ["CN", "EMCI", "LMCI", "AD"]
GROUP_COLORS = dict(zip(GROUPS, plt.cm.twilight(np.linspace(0.08, 0.88, len(GROUPS)))))


def load_csv_matrix(path: str) -> np.ndarray | None:
    try:
        mat = pd.read_csv(path, header=None).to_numpy(float)
    except Exception:
        return None
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] < 60:
        return None
    mat = np.nan_to_num(mat)
    mat = 0.5 * (mat + mat.T)
    np.fill_diagonal(mat, 0)
    if mat.max() > 0:
        mat = mat / mat.max()
    return mat


def load_group_matrices():
    data = {}
    for group in GROUPS:
        mats = []
        for path in glob.glob(str(DATA_DIR / group / "*.csv")):
            mat = load_csv_matrix(path)
            if mat is not None:
                mats.append(mat)
        n = min(m.shape[0] for m in mats)
        mats = np.asarray([m[:n, :n] for m in mats])
        data[group] = mats
    return data


def load_original_group_means(*, n_nodes: int | None = None, tail: bool = False):
    means = {}
    counts = {}
    for group in GROUPS:
        paths = sorted(glob.glob(str(DATA_DIR / group / "*.csv")))
        sum_mat = None
        valid = 0
        for path in paths:
            try:
                mat = pd.read_csv(path, header=None).to_numpy(float)
            except Exception:
                continue
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                continue
            if n_nodes is not None:
                if mat.shape[0] < n_nodes:
                    continue
                mat = mat[-n_nodes:, -n_nodes:] if tail else mat[:n_nodes, :n_nodes]
            mat = np.log1p(np.nan_to_num(mat))
            mat = 0.5 * (mat + mat.T)
            if sum_mat is None:
                sum_mat = np.zeros_like(mat, dtype=float)
            if mat.shape != sum_mat.shape:
                continue
            sum_mat += mat
            valid += 1
        if valid:
            mean_mat = sum_mat / valid
            if mean_mat.max() > 0:
                mean_mat = mean_mat / mean_mat.max()
            np.fill_diagonal(mean_mat, 0)
            means[group] = mean_mat
            counts[group] = valid
    return means, counts


def binary(mean_mat, density=0.18):
    tri = mean_mat[np.triu_indices_from(mean_mat, 1)]
    cutoff = np.quantile(tri[tri > 0], 1 - density)
    a = (mean_mat >= cutoff).astype(float)
    np.fill_diagonal(a, 0)
    return a


def graph_metrics(a):
    degree = a.sum(axis=1)
    w, v = np.linalg.eigh(a)
    eig = np.abs(v[:, np.argmax(w)])
    eig = eig / eig.max()

    n = a.shape[0]
    clustering = np.zeros(n)
    local_eff = np.zeros(n)
    for i in range(n):
        nb = np.where(a[i] > 0)[0]
        k = len(nb)
        if k >= 2:
            sub = a[np.ix_(nb, nb)]
            clustering[i] = sub.sum() / (k * (k - 1))
            local_eff[i] = clustering[i] / np.sqrt(k)
    modules = np.array_split(np.arange(n), 5)
    participation = np.zeros(n)
    for i in range(n):
        ki = max(degree[i], 1)
        participation[i] = 1 - sum((a[i, mod].sum() / ki) ** 2 for mod in modules)
    return {
        "Eigenvector Centrality": eig,
        "Clustering Coefficient": clustering,
        "Local Efficiency": local_eff,
        "Participation Coefficient": participation,
        "Degree": degree,
    }


def connectome_evolution(data):
    means = {g: data[g].mean(axis=0) for g in GROUPS}
    fig, axes = plt.subplots(1, 4, figsize=(8.2, 2.25))
    for i, g in enumerate(GROUPS):
        ax = axes[i]
        im = ax.imshow(means[g], cmap="turbo", vmin=0, vmax=1, interpolation="nearest")
        panel_label(ax, chr(ord("A") + i), -0.18, 1.05)
        ax.set_title(f"{g} (n={len(data[g])})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.015)
    cbar.set_label("Normalized structural connectivity")
    save_figure(fig, "11_ad_connectome_evolution.png")


def lesion_counts(means):
    cn = means["CN"]
    counts = {}
    for g in ["EMCI", "LMCI", "AD"]:
        loss = np.maximum(cn - means[g], 0)
        counts[g] = loss.sum(axis=1)
        if counts[g].max() > 0:
            counts[g] = counts[g] / counts[g].max()
    return counts


def lesion_regression(data):
    means, _ = load_original_group_means()
    cn = means["CN"]
    disease_groups = ["EMCI", "LMCI", "AD"]
    raw_diffs = {g: (cn - means[g]).sum(axis=1) for g in disease_groups}
    global_max = max(float(v.max()) for v in raw_diffs.values())
    lesions = {g: raw_diffs[g] / global_max for g in disease_groups}

    vals, vecs = np.linalg.eigh(cn)
    eigen = np.abs(vecs[:, np.argmax(vals)])

    w_cbrt = np.cbrt(cn)
    n = cn.shape[0]
    cluster = np.zeros(n)
    for i in range(n):
        k_i = np.sum(cn[i] > 0)
        if k_i > 1:
            triangles = w_cbrt[i, :] @ w_cbrt @ w_cbrt[:, i]
            cluster[i] = triangles / (k_i * (k_i - 1))

    threshold = np.percentile(cn, 70)
    g_bin = (cn > threshold).astype(float)
    efficiency = np.zeros(n)
    for i in range(n):
        neighbors = np.where(g_bin[i] > 0)[0]
        k_i = len(neighbors)
        if k_i > 1:
            sub = g_bin[np.ix_(neighbors, neighbors)]
            dist = shortest_paths_unweighted(sub)
            inv = np.divide(1.0, dist, out=np.zeros_like(dist), where=np.isfinite(dist) & (dist > 0))
            efficiency[i] = inv.sum() / (k_i * (k_i - 1))

    degree = np.diag(g_bin.sum(axis=1))
    lap = degree - g_bin
    eig_vals, eig_vecs = np.linalg.eigh(lap)
    order = np.argsort(eig_vals)
    v2 = eig_vecs[:, order[1]]
    v3 = eig_vecs[:, order[2]]
    communities = np.zeros(n, dtype=int)
    communities[(v2 > 0) & (v3 > 0)] = 1
    communities[(v2 > 0) & (v3 <= 0)] = 2
    communities[(v2 <= 0) & (v3 > 0)] = 3
    communities[(v2 <= 0) & (v3 <= 0)] = 4
    participation = np.zeros(n)
    for i in range(n):
        k_i = cn[i].sum()
        if k_i > 0:
            participation[i] = 1 - sum((cn[i, communities == c].sum() / k_i) ** 2 for c in range(1, 5))

    metrics = {
        "Eigenvector Centrality": eigen,
        "Clustering Coefficient": cluster,
        "Local Efficiency": efficiency,
        "Participation Coefficient": participation,
    }
    metric_names = list(metrics)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2))
    axes = axes.ravel()
    for idx, name in enumerate(metric_names):
        ax = axes[idx]
        x = metrics[name]
        handles = []
        labels = []
        for g in disease_groups:
            y = lesions[g]
            ax.scatter(x, y, s=18, color=GROUP_COLORS[g], alpha=0.45, edgecolors="none")
            slope, intercept = np.polyfit(x, y, 1)
            r = np.corrcoef(x, y)[0, 1]
            r_squared = r**2
            df_2 = len(x) - 2
            f_stat = (r_squared * df_2) / (1 - r_squared) if r_squared < 1 else np.inf
            p = stats.f.sf(f_stat, 1, df_2)
            xx = np.linspace(x.min(), x.max(), 100)
            h, = ax.plot(xx, slope * xx + intercept, lw=1.7, color=GROUP_COLORS[g])
            handles.append(h)
            p_str = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
            labels.append(f"{g}: F={f_stat:.1f}, p={p_str}")
        panel_label(ax, chr(ord("A") + idx), -0.13, 1.03)
        clean_axes(ax, f"Baseline {name}", "Lesion Count (Normalised)")
        ax.legend(handles, labels, frameon=False, loc="upper left", fontsize=6.7)
    fig.subplots_adjust(wspace=0.32, hspace=0.38)
    save_figure(fig, "12_topological_lesion_regression.png")


def shortest_paths_unweighted(adj):
    n = adj.shape[0]
    dist = np.full((n, n), np.inf)
    for source in range(n):
        dist[source, source] = 0
        frontier = [source]
        while frontier:
            current = frontier.pop(0)
            for nb in np.where(adj[current] > 0)[0]:
                if not np.isfinite(dist[source, nb]):
                    dist[source, nb] = dist[source, current] + 1
                    frontier.append(nb)
    return dist


def kuramoto_single(mean_mat, lam, omega, theta_init, *, noise_strength, scale, steps, burn, dt, noise_seed=42):
    rng = np.random.default_rng(noise_seed)
    n = mean_mat.shape[0]
    mat = np.asarray(mean_mat, dtype=float)
    np.fill_diagonal(mat, 0)
    theta = theta_init.copy()
    r_trace = []
    noise_factor = noise_strength * np.sqrt(dt)
    for step in range(steps):
        phase = theta[None, :] - theta[:, None]
        coupling = (mat * np.sin(phase)).sum(axis=1)
        drift = omega + scale * lam * coupling
        theta += drift * dt + noise_factor * rng.normal(size=n)
        if step >= burn:
            r_trace.append(abs(np.exp(1j * theta).mean()))
    return float(np.mean(r_trace))


def transition_curve_original(mean_mat, lambdas):
    vals = []
    for lam in lambdas:
        rng = np.random.default_rng(42)
        omega = rng.normal(0, 2.5, mean_mat.shape[0])
        theta = rng.uniform(0, 2 * np.pi, mean_mat.shape[0])
        vals.append(kuramoto_single(mean_mat, lam, omega, theta, noise_strength=0.2, scale=80.0, steps=4000, burn=2000, dt=0.01, noise_seed=42))
    return np.asarray(vals)


def rescue_curve_original(mean_mat, lambdas, omega_init, theta_init):
    return np.asarray([
        kuramoto_single(mean_mat, lam, omega_init, theta_init, noise_strength=0.25, scale=1.0, steps=3000, burn=1800, dt=0.01, noise_seed=42)
        for lam in lambdas
    ])


def transition_delay(data):
    lambdas = np.linspace(0, 0.05, 151)
    means, counts = load_original_group_means()
    curves = {g: transition_curve_original(means[g], lambdas) for g in GROUPS}
    fig, ax = plt.subplots(figsize=(5.45, 3.25))
    ax.axvspan(0.015, 0.030, color=COLORS["blue"], alpha=0.10, lw=0, label="Critical Regime (0.01-0.03)")
    for g in GROUPS:
        ax.plot(lambdas, curves[g], lw=1.5, color=GROUP_COLORS[g], label=f"{g} (n={counts[g]})")
    ax.set_xlabel("Cortical Coupling Factor, $\\lambda$")
    ax.set_ylabel("Order Parameter $r$")
    ax.set_xlim(0, 0.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.40)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.legend(loc="lower right", frameon=False, fontsize=7.2)
    save_figure(fig, "13_ad_phase_transition_delay.png")


def rescue_experiment(data):
    lambdas = np.arange(0.0, 0.6001, 0.01)
    twilight = plt.cm.twilight(np.linspace(0.08, 0.88, 4))
    means, _ = load_original_group_means(n_nodes=84, tail=True)
    n = 84
    m_cn = means["CN"]
    m_ad = means["AD"]
    vals, vecs = np.linalg.eigh(m_cn)
    eig = np.abs(vecs[:, np.argmax(vals)])
    hubs = np.argsort(eig)[::-1][: round(0.10 * n)]
    rng = np.random.default_rng(42)
    random_nodes = rng.permutation(n)[: len(hubs)]
    m_hub = m_ad.copy()
    m_hub[hubs, :] = m_cn[hubs, :]
    m_hub[:, hubs] = m_cn[:, hubs]
    m_rand = m_ad.copy()
    m_rand[random_nodes, :] = m_cn[random_nodes, :]
    m_rand[:, random_nodes] = m_cn[:, random_nodes]
    init_rng = np.random.default_rng(100)
    omega_init = init_rng.normal(0, 0.25, n)
    theta_init = init_rng.uniform(0, 2 * np.pi, n)
    curves = {
        "Healthy Baseline (CN)": (rescue_curve_original(m_cn, lambdas, omega_init, theta_init), twilight[0], "-^", 1.5, 3.2),
        "Targeted Rescue (Top 10 Hubs)": (rescue_curve_original(m_hub, lambdas, omega_init, theta_init), twilight[2], "-o", 1.5, 3.6),
        "Random Rescue (10 Non-Hubs)": (rescue_curve_original(m_rand, lambdas, omega_init, theta_init), twilight[1], "--s", 1.5, 3.2),
        "Pathological Baseline (AD)": (rescue_curve_original(m_ad, lambdas, omega_init, theta_init), twilight[3], ":", 1.5, 0),
    }
    fig, ax = plt.subplots(figsize=(5.45, 3.5))
    for label, (y, color, style, lw, ms) in curves.items():
        kwargs = {"color": color, "lw": lw, "label": label}
        if ms:
            kwargs.update({"ms": ms, "markerfacecolor": color})
        ax.plot(lambdas, y, style, **kwargs)
    ax.set_xlabel("Coupling Strength, $\\lambda$")
    ax.set_ylabel("Global Synchronization, $R$")
    ax.set_xlim(0, 0.60)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    save_figure(fig, "14_perturbation_rescue_experiment.png")


if __name__ == "__main__":
    group_data = load_group_matrices()
    connectome_evolution(group_data)
    lesion_regression(group_data)
    transition_delay(group_data)
    rescue_experiment(group_data)
