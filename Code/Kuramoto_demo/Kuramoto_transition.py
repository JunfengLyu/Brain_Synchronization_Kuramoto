import matplotlib.pyplot as plt
import numpy as np


def gaussian(x, sigma):
    return np.exp(-(x**2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)


def critical_k(sigma):
    return 2 / (np.pi * gaussian(0, sigma))


def psi_curve(k, r, sigma):
    x = np.linspace(-1, 1, 2501)
    kernel = np.sqrt(np.maximum(0, 1 - x * x))
    vals = gaussian(k * r[:, None] * x[None, :], sigma) * kernel[None, :]
    return k * np.trapz(vals, x, axis=1) - 1


def mean_field_r(k, sigma):
    if k <= critical_k(sigma):
        return 0.0
    grid = np.linspace(1e-4, 0.999, 700)
    vals = psi_curve(k, grid, sigma)
    hits = np.where(vals[:-1] * vals[1:] <= 0)[0]
    return float(grid[hits[-1]]) if len(hits) else float(grid[np.argmin(np.abs(vals))])


def simulate_sweep(n, k_values, sigma, dt=0.05, total=150, burn=75, seed=1):
    rng = np.random.default_rng(seed)
    omega = rng.normal(1.0, sigma, n)
    means, stds = [], []
    for k in k_values:
        theta = rng.uniform(0, 2 * np.pi, n)
        rs = []
        for step in range(round(total / dt)):
            z = np.mean(np.exp(1j * theta))
            theta += dt * (omega + k * abs(z) * np.sin(np.angle(z) - theta))
            if step >= round(burn / dt):
                rs.append(abs(np.mean(np.exp(1j * theta))))
        means.append(np.mean(rs))
        stds.append(np.std(rs))
    return np.asarray(means), np.asarray(stds)


def main(save=False):
    sigma = 0.5
    kc = critical_k(sigma)
    r = np.linspace(0, 1, 800)
    k_vals = [0.75 * kc, kc, 1.35 * kc]
    labels = [r"$K<K_c$", r"$K=K_c$", r"$K>K_c$"]
    colors = ["#6baed6", "#084594", "#d13a6f"]
    k_sample = np.linspace(0, 2, 41)
    n_list = [10, 100, 1000, 10000]
    sim_colors = ["#4cc0d4", "#4040d4", "#bf40cc", "#d93659"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for k, label, c in zip(k_vals, labels, colors):
        axes[0].plot(psi_curve(k, r, sigma), r, lw=2.5, color=c, label=label)
    pr = psi_curve(k_vals[-1], r, sigma)
    idx = np.where(pr[:-1] * pr[1:] <= 0)[0][-1]
    r_star = np.interp(0, pr[idx:idx+2], r[idx:idx+2])
    axes[0].axvline(0, color="k", lw=1.5)
    axes[0].scatter([0], [r_star], s=80, color=colors[-1], zorder=3)
    axes[0].set_xlabel(r"$\psi(K;R)$")
    axes[0].set_ylabel(r"$R$")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False)

    kd = np.linspace(0, 2, 240)
    axes[1].plot(kd, [mean_field_r(k, sigma) for k in kd], color="0.55", lw=3, label="MF prediction")
    for n, c in zip(n_list, sim_colors):
        m, s = simulate_sweep(n, k_sample, sigma, seed=1 + n)
        axes[1].errorbar(k_sample, m, yerr=s, fmt="o", ms=4, lw=0.6, capsize=2, color=c, label=fr"$N={n}$")
    axes[1].set_xlabel(r"$K$")
    axes[1].set_ylabel(r"$R$")
    axes[1].set_xlim(0, 2)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(frameon=False, loc="lower right")
    for lab, ax in zip(["A", "B"], axes):
        ax.text(-0.12, 1.05, lab, transform=ax.transAxes, fontsize=18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out")
    if save:
        fig.savefig("04_mean_field_bifurcation.png", dpi=300, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    main(save=False)
