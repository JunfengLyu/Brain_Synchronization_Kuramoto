from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"


def percentile(v, p):
    v = np.sort(np.asarray(v, float).ravel())
    if len(v) == 0:
        return 0.0
    pos = (len(v) - 1) * p / 100
    lo, hi = int(np.floor(pos)), int(np.ceil(pos))
    return float(v[lo] if lo == hi else v[lo] + (pos - lo) * (v[hi] - v[lo]))


def load_aal_consensus(threshold=0.40):
    raw = sio.loadmat(DATA_DIR / "rawdata.mat")
    sc = np.asarray(raw["SCmatrices"], float)
    processed = 0.5 * (sc + np.swapaxes(sc, 1, 2))
    idx = np.arange(processed.shape[1])
    processed[:, idx, idx] = 0
    prob = (processed > 0).mean(axis=0)
    group = (prob >= threshold).astype(float)
    np.fill_diagonal(group, 0)
    return group, prob


def main(save=False):
    m_group, _ = load_aal_consensus()
    degree = m_group.sum(axis=1)
    cutoff = percentile(degree, 85)
    hubs = degree >= cutoff
    n = len(degree)
    print(f"Number of regions = {n}")
    print(f"Hub threshold (85th percentile) = {cutoff:.1f}")
    print(f"Number of hubs = {hubs.sum()}")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.25, 3.25), gridspec_kw={"width_ratios": [1, 1.2]})
    im = ax_a.imshow(m_group, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    ax_a.set_aspect("equal")
    ax_a.set_xlabel("Brain region index")
    ax_a.set_ylabel("Brain region index")
    ax_a.set_xticks([0, 29, 59, 89])
    ax_a.set_yticks([0, 29, 59, 89])
    ax_a.text(-0.16, 1.04, "A", transform=ax_a.transAxes, fontsize=12)
    cb = fig.colorbar(im, ax=ax_a, fraction=0.047, pad=0.03)
    cb.set_label("Edge exists")

    x = np.arange(1, n + 1)
    ax_b.bar(x, degree, color=[0.85, 0.87, 0.89], edgecolor="none", label="Non-hub")
    ax_b.bar(x[hubs], degree[hubs], color=[0.82, 0.26, 0.36], edgecolor="none", label="Hub")
    ax_b.axhline(cutoff, color=[0.05, 0.05, 0.05], lw=1.3, ls="--", label=f"85th percentile = {cutoff:.1f}")
    ax_b.set_xlabel("Brain region index")
    ax_b.set_ylabel("Node degree")
    ax_b.set_xlim(0, n + 1)
    ax_b.set_ylim(0, degree.max() * 1.18)
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False)
    ax_b.text(-0.12, 1.04, "B", transform=ax_b.transAxes, fontsize=12)
    for ax in (ax_a, ax_b):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.2, top=0.84, wspace=0.38)
    if save:
        fig.savefig(BASE / "06_connectome_construction.png", dpi=300, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    main(save=False)
