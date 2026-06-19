import matplotlib.pyplot as plt
import numpy as np


TAU = 2 * np.pi


def wrap_pi(x):
    return (x + np.pi) % TAU - np.pi


def top_crossing(theta_old, theta_new):
    old = wrap_pi(theta_old - np.pi / 2)
    new = wrap_pi(theta_new - np.pi / 2)
    return ((old < 0) & (new >= 0) | (old > 0) & (new <= 0)) & (np.abs(new - old) < np.pi)


def step(theta, omega, k, dt):
    z = np.mean(np.exp(1j * theta))
    r, psi = abs(z), np.angle(z)
    return theta + dt * (omega + k * r * np.sin(psi - theta)), r


def simulate(n=64, sigma=0.50, k=1.20, dt=0.05, steps=1800, seed=1):
    rng = np.random.default_rng(seed)
    omega = np.sort(1.0 + sigma * rng.standard_normal(n))
    theta = np.ones(n) * np.pi / 2
    r_hist, spikes = [], []
    for s in range(steps):
        theta_new, r = step(theta, omega, k, dt)
        crossed = np.flatnonzero(top_crossing(theta, theta_new))
        for idx in crossed:
            spikes.append((s * dt, idx))
        theta = theta_new
        r_hist.append((s * dt, r))
    return omega, theta, np.asarray(r_hist), np.asarray(spikes)


def main(save=False):
    omega, theta, r_hist, spikes = simulate()
    fig = plt.figure(figsize=(13, 7.6))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, (3, 4))
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(theta)))

    circle = np.linspace(0, TAU, 400)
    ax1.plot(np.cos(circle), np.sin(circle), color="0.1", lw=1)
    ax1.scatter(np.cos(theta), np.sin(theta), s=35, c=colors, edgecolor="k", lw=0.3)
    z = np.mean(np.exp(1j * theta))
    ax1.arrow(0, 0, z.real, z.imag, width=0.01, color="k", length_includes_head=True)
    ax1.set_aspect("equal")
    ax1.axis("off")

    k_grid = np.linspace(0, 2, 200)
    kc = 2 * np.sqrt(2 / np.pi) * 0.5
    mf = np.where(k_grid > kc, np.sqrt(np.maximum(0, 1 - kc / np.maximum(k_grid, 1e-9))), 0)
    ax2.plot(k_grid, mf, color="0.4", lw=2)
    ax2.scatter([1.2], [np.mean(r_hist[-500:, 1])], c="crimson", s=45, zorder=3)
    ax2.set_xlabel("K")
    ax2.set_ylabel("R")
    ax2.set_ylim(0, 1.05)
    ax2.spines[["top", "right"]].set_visible(False)

    if len(spikes):
        ax3.scatter(spikes[:, 0], spikes[:, 1] + 1, s=3, color="k")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Oscillator index")
    ax3.spines[["top", "right"]].set_visible(False)
    if save:
        fig.savefig("Kuramoto_rotor_snapshot.png", dpi=300, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    main(save=False)
