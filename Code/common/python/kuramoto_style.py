from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = PROJECT_ROOT / "Report" / "Figs"
ROOT_DATA_DIR = PROJECT_ROOT / "data_for_section3&4"
LEGACY_DATA_DIR = PROJECT_ROOT / "MatLab_codes" / "data_for_section3&4"
DATA_DIR = ROOT_DATA_DIR if ROOT_DATA_DIR.exists() else LEGACY_DATA_DIR

COLORS = {
    "teal": "#2AA7B8",
    "blue": "#3B42D8",
    "magenta": "#D12A7A",
    "red": "#D1435B",
    "green": "#2CA02C",
    "gray": "#505050",
    "light_gray": "#D8DEE2",
    "black": "#111111",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 1.1,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "mathtext.default": "it",
            "mathtext.fontset": "stix",
        }
    )


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="normal",
        color=COLORS["black"],
    )


def clean_axes(ax, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(direction="out", length=3.5, width=1.0, color=COLORS["black"])
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def despine_all(fig) -> None:
    for ax in fig.axes:
        if hasattr(ax, "spines"):
            clean_axes(ax)


def save_figure(fig, filename: str, *, tight: bool = True) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / filename
    fig.savefig(out, dpi=300, bbox_inches="tight" if tight else None, facecolor="white")
    plt.close(fig)
    return out


def load_group_consensus(threshold: float = 0.40):
    import scipy.io as sio

    mat = sio.loadmat(DATA_DIR / "rawdata.mat", squeeze_me=True)
    all_mats = np.asarray(mat["SCmatrices"], dtype=float)
    processed = 0.5 * (all_mats + np.swapaxes(all_mats, 1, 2))
    idx = np.arange(processed.shape[1])
    processed[:, idx, idx] = 0
    prob = (processed > 0).mean(axis=0)
    group = (prob >= threshold).astype(float)
    np.fill_diagonal(group, 0)
    return group, prob


def order_parameter(theta: np.ndarray) -> float:
    return float(np.abs(np.exp(1j * theta).mean()))


def simulate_network(
    adjacency: np.ndarray,
    lambdas: np.ndarray,
    *,
    omega_std: float = 1.0,
    noise_strength: float = 0.2,
    steps: int = 1200,
    burn: int = 650,
    dt: float = 0.03,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    n = adjacency.shape[0]
    degree = adjacency.sum(axis=1)
    degree[degree == 0] = 1.0
    norm_adj = adjacency / degree[:, None]
    omega = rng.normal(0, omega_std, n)
    theta0 = rng.uniform(0, 2 * np.pi, n)
    means, stds, snapshots = [], [], {}

    for lam in lambdas:
        theta = theta0.copy()
        r_trace = []
        sample_theta = []
        for step in range(steps):
            phase = theta[None, :] - theta[:, None]
            coupling = (norm_adj * np.sin(phase)).sum(axis=1)
            theta += dt * (omega + lam * 90 * coupling)
            if noise_strength:
                theta += np.sqrt(dt) * noise_strength * rng.normal(size=n)
            if step >= burn:
                r_trace.append(order_parameter(theta))
                if step % 8 == 0:
                    sample_theta.append(theta.copy())
        means.append(np.mean(r_trace))
        stds.append(np.std(r_trace))
        snapshots[float(lam)] = np.asarray(sample_theta)
    return np.asarray(means), np.asarray(stds), snapshots


def plv_matrix(theta_samples: np.ndarray) -> np.ndarray:
    phases = np.exp(1j * theta_samples)
    fc = np.abs(phases.conj().T @ phases) / max(1, theta_samples.shape[0])
    np.fill_diagonal(fc, 1)
    return fc


def smooth_transition(x, center, width, low=0.0, high=1.0):
    return low + (high - low) / (1 + np.exp(-(x - center) / width))


set_style()
