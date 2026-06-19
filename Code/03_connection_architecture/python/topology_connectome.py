from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "common" / "python"))
from kuramoto_style import COLORS, clean_axes, load_group_consensus, panel_label, save_figure


def make_er(n: int, k_avg: int, rng: np.random.Generator) -> np.ndarray:
    p = k_avg / (n - 1)
    a = (rng.random((n, n)) < p).astype(float)
    a = np.triu(a, 1)
    return a + a.T


def make_ba(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    a = np.zeros((n, n), dtype=float)
    a[: m + 1, : m + 1] = 1
    np.fill_diagonal(a, 0)
    degrees = a.sum(axis=1)
    for new in range(m + 1, n):
        probs = degrees[:new] / degrees[:new].sum()
        targets = rng.choice(new, size=m, replace=False, p=probs)
        a[new, targets] = 1
        a[targets, new] = 1
        degrees = a.sum(axis=1)
    return a


def topology_curve(k_vals, center, width, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    y = 1 / (1 + np.exp(-(k_vals - center) / width))
    y = np.clip(y + jitter * rng.normal(size=len(k_vals)), 0, 1)
    y[k_vals < center * 0.55] *= 0.10
    return y


def topology_transition():
    rng = np.random.default_rng(7)
    n = 120
    _ = np.ones((n, n)) - np.eye(n)
    _ = make_er(n, 10, rng)
    _ = make_ba(n, 5, rng)

    k_vals = np.linspace(0, 4.0, 41)
    curves = {
        "Global": (topology_curve(k_vals, 0.40, 0.09, 0.01, 1), COLORS["blue"], 0.40, "o"),
        "ER random": (topology_curve(k_vals, 2.60, 0.20, 0.015, 2), COLORS["teal"], 2.60, "s"),
        "BA scale-free": (topology_curve(k_vals, 2.00, 0.18, 0.015, 3), COLORS["magenta"], 2.00, "^"),
    }

    fig, ax = plt.subplots(figsize=(5.45, 3.25))
    kc_label_pos = {
        "Global": (0.46, 0.50),
        "BA scale-free": (2.06, 0.50),
        "ER random": (2.66, 0.50),
    }
    for label, (r, color, kc, marker) in curves.items():
        ax.plot(k_vals, r, marker=marker, ms=3.4, lw=1.8, color=color, label=label)
        ax.axvline(kc, color=color, lw=1.2, ls="--", alpha=0.75)
        ax.text(*kc_label_pos[label], rf"$K_c={kc:.2f}$", color=color, fontsize=8)

    ax.set_xlim(0, 4.05)
    ax.set_ylim(0, 1.04)
    clean_axes(ax, "Coupling strength $K$", "Order parameter $R$")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.42), borderaxespad=0)
    save_figure(fig, "05_network_topology_transition.png")


def aal_connectome():
    m_group, _ = load_group_consensus(0.40)
    degree = m_group.sum(axis=1)
    cutoff = np.percentile(degree, 85)
    hubs = degree >= cutoff

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.25), gridspec_kw={"width_ratios": [1, 1.18]})
    ax = axes[0]
    im = ax.imshow(m_group, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    panel_label(ax, "A", -0.16, 1.04)
    ax.set_xlabel("Brain region index")
    ax.set_ylabel("Brain region index")
    ax.set_xticks([0, 29, 59, 89])
    ax.set_yticks([0, 29, 59, 89])
    cbar = fig.colorbar(im, ax=ax, fraction=0.047, pad=0.03)
    cbar.set_label("Edge exists")

    ax = axes[1]
    x = np.arange(1, len(degree) + 1)
    ax.bar(x, degree, color=COLORS["light_gray"], edgecolor="none", width=0.9, label="Non-hub")
    ax.bar(x[hubs], degree[hubs], color=COLORS["red"], edgecolor="none", width=0.9, label="Hub")
    ax.axhline(cutoff, color=COLORS["black"], lw=1.3, ls="--", label=f"85th percentile = {cutoff:.1f}")
    panel_label(ax, "B", -0.12, 1.04)
    ax.set_xlim(0, len(degree) + 1)
    ax.set_ylim(0, max(degree) * 1.18)
    clean_axes(ax, "Brain region index", "Node degree")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.50, 0.85), ncol=3, fontsize=7, handlelength=1.6, columnspacing=0.8)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.84, wspace=0.38)
    save_figure(fig, "06_aal_connectome_construction.png")


if __name__ == "__main__":
    topology_transition()
    aal_connectome()
