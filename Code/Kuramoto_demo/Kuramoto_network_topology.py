import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def ba_network(n, m0, m, rng):
    a = np.zeros((n, n))
    a[:m0, :m0] = 1
    np.fill_diagonal(a, 0)
    deg = a.sum(axis=1)
    for new in range(m0, n):
        probs = deg[:new] / deg[:new].sum()
        targets = rng.choice(new, size=m, replace=False, p=probs)
        a[new, targets] = 1
        a[targets, new] = 1
        deg = a.sum(axis=1)
    return a


def rhs(_, theta, omega, k, a):
    phase = theta[None, :] - theta[:, None]
    return omega + (k / len(theta)) * (a * np.sin(phase)).sum(axis=1)


def order_parameter(theta_tail):
    return float(np.abs(np.exp(1j * theta_tail).mean(axis=1)).mean())


def scan_network(a, omega, k_range, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for k in k_range:
        sol = solve_ivp(lambda t, y: rhs(t, y, omega, k, a), (0, 200), 2*np.pi*rng.random(len(omega)),
                        t_eval=np.linspace(0, 200, 1200), rtol=1e-5, atol=1e-7)
        out.append(order_parameter(sol.y[:, -250:].T))
    return np.asarray(out)


def first_cross(k_range, r, threshold=0.2):
    idx = np.where(r > threshold)[0]
    return float(k_range[idx[0]]) if len(idx) else np.nan


def main(save=False):
    rng = np.random.default_rng(42)
    n = 100
    omega = 1 + 0.25 * rng.standard_normal(n)
    k_range = np.arange(0, 8.0001, 0.2)
    a_global = np.ones((n, n)) - np.eye(n)
    a_er = np.triu((rng.random((n, n)) < 0.10).astype(float), 1)
    a_er = a_er + a_er.T
    a_ba = ba_network(n, 5, 5, rng)
    curves = {
        "Global": (scan_network(a_global, omega, k_range, 10), plt.cm.plasma(0.16), "o"),
        "ER random": (scan_network(a_er, omega, k_range, 20), plt.cm.plasma(0.55), "s"),
        "BA scale-free": (scan_network(a_ba, omega, k_range, 30), plt.cm.plasma(0.86), "^"),
    }
    fig, ax = plt.subplots(figsize=(5.45, 3.25))
    for label, (r, c, marker) in curves.items():
        kc = first_cross(k_range, r)
        ax.plot(k_range, r, "-" + marker, color=c, ms=3.4, lw=1.8, label=label)
        ax.axvline(kc, ls="--", lw=1.2, color=c)
        ax.text(kc + 0.08, 0.5, fr"$K_c={kc:.2f}$", color=c, fontsize=8)
    ax.set_xlabel("Coupling strength K")
    ax.set_ylabel("Order parameter R")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.04)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out")
    if save:
        fig.savefig("05_network_topology_transition.png", dpi=300, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    main(save=False)
