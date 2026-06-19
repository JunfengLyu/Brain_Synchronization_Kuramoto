from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


BASE = Path(__file__).resolve().parent
FIG_DIR = BASE.parent.parent / "Report" / "Figs"
GROUPS = ["CN", "EMCI", "LMCI", "AD"]


def twilight_colors(n):
    cmap = plt.get_cmap("twilight")
    return np.array([cmap(x)[:3] for x in np.linspace(0.08, 0.88, n)])


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=1.1, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def read_csv_matrix(path):
    arr = np.genfromtxt(path, delimiter=",")
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return None
    arr[~np.isfinite(arr)] = 0
    return arr


def load_group_means(groups=GROUPS, n_nodes=None, tail_mode=False):
    means, counts = {}, {}
    for group in groups:
        total, valid = None, 0
        for path in sorted((BASE / "data" / group).glob("*.csv")):
            mat = read_csv_matrix(path)
            if mat is None:
                continue
            if n_nodes is not None:
                if mat.shape[0] < n_nodes:
                    continue
                mat = mat[-n_nodes:, -n_nodes:] if tail_mode else mat[:n_nodes, :n_nodes]
            mat = np.log1p(np.maximum(mat, 0))
            mat = 0.5 * (mat + mat.T)
            np.fill_diagonal(mat, 0)
            if total is None:
                total = np.zeros_like(mat)
            if total.shape == mat.shape:
                total += mat
                valid += 1
        if valid == 0:
            raise RuntimeError(f"No valid CSV matrices for {group}")
        mean = total / valid
        if mean.max() > 0:
            mean = mean / mean.max()
        np.fill_diagonal(mean, 0)
        means[group] = mean
        counts[group] = valid
    return means, counts


def shortest_unweighted(adj):
    n = adj.shape[0]
    dist = np.full((n, n), np.inf)
    dist[adj > 0] = 1
    np.fill_diagonal(dist, 0)
    for k in range(n):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
    return dist


def graph_metrics(w):
    n = w.shape[0]
    vals, vecs = np.linalg.eig(w)
    eig = np.abs(vecs[:, np.argmax(vals.real)].real)
    eig = eig / max(eig.max(), np.finfo(float).eps)
    wc = np.cbrt(np.maximum(w, 0))
    clust = np.zeros(n)
    for i in range(n):
        k = np.count_nonzero(w[i] > 0)
        if k > 1:
            clust[i] = (wc[i] @ wc @ wc[:, i]) / (k * (k - 1))
    thr = np.percentile(w.ravel(), 70)
    adj = (w > thr).astype(float)
    np.fill_diagonal(adj, 0)
    eff = np.zeros(n)
    for i in range(n):
        nb = np.flatnonzero(adj[i])
        if len(nb) > 1:
            d = shortest_unweighted(adj[np.ix_(nb, nb)])
            inv = np.divide(1, d, out=np.zeros_like(d), where=np.isfinite(d) & (d > 0))
            eff[i] = inv.sum() / (len(nb) * (len(nb) - 1))
    lap = np.diag(adj.sum(axis=1)) - adj
    evals, evecs = np.linalg.eigh(lap)
    v2, v3 = evecs[:, min(1, n - 1)], evecs[:, min(2, n - 1)]
    comm = np.ones(n, dtype=int)
    comm[(v2 > 0) & (v3 <= 0)] = 2
    comm[(v2 <= 0) & (v3 > 0)] = 3
    comm[(v2 <= 0) & (v3 <= 0)] = 4
    part = np.zeros(n)
    for i in range(n):
        ki = w[i].sum()
        if ki > 0:
            part[i] = 1 - sum((w[i, comm == c].sum() / ki) ** 2 for c in range(1, 5))
    return {
        "Eigenvector_Centrality": eig,
        "Clustering_Coefficient": clust,
        "Local_Efficiency": eff,
        "Participation_Coefficient": part,
    }


def kuramoto_order_fixed(mat, lam, omega, theta, noise, scale, steps, burn, dt, seed):
    rng = np.random.default_rng(seed)
    theta = theta.copy()
    rvals = []
    for s in range(steps):
        phase = theta[None, :] - theta[:, None]
        drift = omega + scale * lam * (mat * np.sin(phase)).sum(axis=1)
        theta += drift * dt + noise * np.sqrt(dt) * rng.normal(size=theta.size)
        if s >= burn:
            rvals.append(abs(np.exp(1j * theta).mean()))
    return float(np.mean(rvals))


def kuramoto_order(mat, lam, omega_std, noise, scale, steps, burn, dt, seed):
    rng = np.random.default_rng(seed)
    n = mat.shape[0]
    return kuramoto_order_fixed(mat, lam, omega_std * rng.normal(size=n), rng.random(n) * 2 * np.pi, noise, scale, steps, burn, dt, seed)


def make_fig11(means, counts, save=False):
    fig, axes = plt.subplots(1, 4, figsize=(8.2, 2.25), constrained_layout=True)
    for i, (ax, group) in enumerate(zip(axes, GROUPS)):
        im = ax.imshow(means[group], vmin=0, vmax=1, cmap="turbo")
        ax.axis("off")
        ax.text(-0.18, 1.05, chr(ord("A") + i), transform=ax.transAxes, fontname="Arial", fontsize=11)
        ax.text(0.04, 0.95, f"{group} (n={counts[group]})", transform=ax.transAxes, color="w", fontsize=8.5, va="top")
    fig.colorbar(im, ax=axes, location="right", label="Normalized structural connectivity")
    if save:
        fig.savefig(FIG_DIR / "11_ad_connectome_evolution.png", dpi=300, bbox_inches="tight")
    return fig


def make_fig12(means, save=False):
    cn = means["CN"]
    disease = GROUPS[1:]
    raw = {g: np.maximum(cn - means[g], 0).sum(axis=1) for g in disease}
    ymax = max(v.max() for v in raw.values())
    metrics = graph_metrics(cn)
    cols = twilight_colors(4)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2), constrained_layout=True)
    for i, (ax, (name, x)) in enumerate(zip(axes.ravel(), metrics.items())):
        leg = []
        for j, group in enumerate(disease):
            y = raw[group] / max(ymax, np.finfo(float).eps)
            ax.scatter(x, y, s=14, color=cols[j + 1], alpha=0.45)
            p = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 80)
            ax.plot(xx, np.polyval(p, xx), lw=1.5, color=cols[j + 1])
            r = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else 0
            r2 = r * r
            fval = (r2 * (len(x) - 2)) / max(1 - r2, np.finfo(float).eps)
            pval = stats.f.sf(fval, 1, len(x) - 2)
            leg.append(f"{group}: F={fval:.1f}, p={pval:.2e}" if pval < 1e-3 else f"{group}: F={fval:.1f}, p={pval:.3f}")
        ax.text(-0.13, 1.03, chr(ord("A") + i), transform=ax.transAxes, fontname="Arial", fontsize=11)
        ax.set_xlabel("Baseline " + name.replace("_", " "))
        ax.set_ylabel("Lesion Count (Normalised)")
        ax.legend(leg, loc="upper left", frameon=False, fontsize=6.7)
        style_axes(ax)
    if save:
        fig.savefig(FIG_DIR / "12_topological_lesion_regression.png", dpi=300, bbox_inches="tight")
    return fig


def make_fig13(means, counts, save=False):
    lambdas = np.linspace(0, 0.05, 151)
    cols = twilight_colors(4)
    curves = np.zeros((4, len(lambdas)))
    for gi, group in enumerate(GROUPS):
        for ki, lam in enumerate(lambdas):
            curves[gi, ki] = kuramoto_order(means[group], lam, 2.5, 0.2, 80, 4000, 2000, 0.01, 42)
    fig, ax = plt.subplots(figsize=(5.45, 3.10))
    ax.axvspan(0.015, 0.030, color=(0.16, 0.65, 0.72), alpha=0.10, label="Critical Regime (0.01-0.03)")
    for group, curve, col in zip(GROUPS, curves, cols):
        ax.plot(lambdas, curve, lw=1.5, color=col, label=f"{group} (n={counts[group]})")
    ax.set(xlabel=r"Cortical Coupling Factor, $\lambda$", ylabel=r"Order Parameter $R$", xlim=(0, 0.05), ylim=(0, 1.05))
    ax.grid(True, ls="--", alpha=0.30)
    ax.legend(loc="lower right", frameon=False, fontsize=7.2)
    style_axes(ax)
    if save:
        fig.savefig(FIG_DIR / "13_ad_phase_transition_delay.png", dpi=300, bbox_inches="tight")
    return fig


def make_fig14(save=False):
    lambdas = np.arange(0, 0.6001, 0.01)
    means, _ = load_group_means(n_nodes=84, tail_mode=True)
    cn, ad = means["CN"], means["AD"]
    n = cn.shape[0]
    vals, vecs = np.linalg.eig(cn)
    eig = np.abs(vecs[:, np.argmax(vals.real)].real)
    top = np.argsort(eig)[::-1][: max(1, round(0.10 * n))]
    rng = np.random.default_rng(42)
    random_nodes = rng.choice(n, size=len(top), replace=False)
    hub = ad.copy()
    hub[top, :] = cn[top, :]
    hub[:, top] = cn[:, top]
    rand = ad.copy()
    rand[random_nodes, :] = cn[random_nodes, :]
    rand[:, random_nodes] = cn[:, random_nodes]
    rng = np.random.default_rng(100)
    omega0 = 0.25 * rng.normal(size=n)
    theta0 = rng.random(n) * 2 * np.pi
    mats = [cn, hub, rand, ad]
    curves = np.zeros((4, len(lambdas)))
    for i, mat in enumerate(mats):
        for k, lam in enumerate(lambdas):
            curves[i, k] = kuramoto_order_fixed(mat, lam, omega0, theta0, 0.25, 1, 3000, 1800, 0.01, 42)
    cols = twilight_colors(4)
    fig, ax = plt.subplots(figsize=(5.45, 3.25))
    styles = ["-^", "-o", "--s", ":"]
    labels = ["Healthy Baseline (CN)", "Targeted Rescue (Top 10 Hubs)", "Random Rescue (10 Non-Hubs)", "Pathological Baseline (AD)"]
    for y, sty, col, lab in zip(curves, styles, cols, labels):
        ax.plot(lambdas, y, sty, lw=1.5, ms=3.2, color=col, markerfacecolor=col, label=lab)
    ax.set(xlabel=r"Coupling Strength, $\lambda$", ylabel=r"Global Synchronization, $R$", xlim=(0, 0.6), ylim=(0, 1.05))
    ax.grid(True, alpha=0.15)
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    style_axes(ax)
    if save:
        fig.savefig(FIG_DIR / "14_perturbation_rescue_experiment.png", dpi=300, bbox_inches="tight")
    return fig


def main(save=False):
    means, counts = load_group_means()
    make_fig11(means, counts, save)
    make_fig12(means, save)
    make_fig13(means, counts, save)
    make_fig14(save)
    plt.show()


if __name__ == "__main__":
    main(save=False)
