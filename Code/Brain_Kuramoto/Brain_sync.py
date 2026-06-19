from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle
from scipy.io import loadmat


BASE = Path(__file__).resolve().parent
FIG_DIR = BASE.parent.parent / "Report" / "Figs"


def twilight_colors(n):
    cmap = plt.get_cmap("twilight")
    return np.array([cmap(x)[:3] for x in np.linspace(0.08, 0.88, n)])


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=1.1, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def load_aal_consensus(threshold=0.40):
    sc = loadmat(BASE / "data" / "rawdata.mat")["SCmatrices"]
    sc = 0.5 * (sc + np.transpose(sc, (0, 2, 1)))
    n = sc.shape[1]
    sc[:, np.arange(n), np.arange(n)] = 0
    prob = (sc > 0).mean(axis=0)
    mat = (prob >= threshold).astype(float)
    np.fill_diagonal(mat, 0)
    return mat, prob


def brain_modules(n):
    return {
        "Frontal": np.arange(0, min(28, n)),
        "Limbic_Subcortical": np.arange(28, min(46, n)),
        "Occipital": np.arange(46, min(58, n)),
        "Parietal": np.arange(58, min(72, n)),
        "Temporal": np.arange(72, n),
    }


def simulate_trace(mat, lam, steps, dt, noise, seed):
    rng = np.random.default_rng(seed)
    n = mat.shape[0]
    deg = mat.sum(axis=1)
    deg[deg == 0] = 1
    wmat = mat / deg[:, None]
    omega = rng.normal(size=n)
    theta = rng.random(n) * 2 * np.pi
    r = np.zeros(steps)
    for s in range(steps):
        phase = theta[None, :] - theta[:, None]
        theta += dt * (omega + lam * 90 * (wmat * np.sin(phase)).sum(axis=1))
        theta += noise * np.sqrt(dt) * rng.normal(size=n)
        r[s] = abs(np.exp(1j * theta).mean())
    return np.arange(steps) * dt, r


def scan_network(mat, lambdas, steps, burn, dt, noise, seed):
    rng = np.random.default_rng(seed)
    n = mat.shape[0]
    deg = mat.sum(axis=1)
    deg[deg == 0] = 1
    wmat = mat / deg[:, None]
    omega = rng.normal(size=n)
    theta0 = rng.random(n) * 2 * np.pi
    edge_den = max(mat.sum(), 1)
    rmean, rstd, rlink, snaps = [], [], [], []
    for lam in lambdas:
        theta = theta0.copy()
        rt, lt, sample = [], [], []
        for s in range(steps):
            phase = theta[None, :] - theta[:, None]
            theta += dt * (omega + lam * 90 * (wmat * np.sin(phase)).sum(axis=1))
            theta += noise * np.sqrt(dt) * rng.normal(size=n)
            if s >= burn:
                rt.append(abs(np.exp(1j * theta).mean()))
                lt.append((mat * np.cos(theta[None, :] - theta[:, None])).sum() / edge_den)
                if len(rt) % 8 == 1:
                    sample.append(theta.copy())
        rmean.append(np.mean(rt))
        rstd.append(np.std(rt))
        rlink.append(np.clip(np.mean(lt), 0, 1))
        snaps.append(np.asarray(sample))
    return np.asarray(rmean), np.asarray(rstd), np.asarray(rlink), snaps


def plv_matrix(theta_samples):
    z = np.exp(1j * theta_samples)
    fc = np.abs(z.conj().T @ z) / max(1, len(theta_samples))
    np.fill_diagonal(fc, 1)
    return fc


def local_order(theta_samples, nodes):
    nodes = np.asarray(nodes, dtype=int)
    nodes = nodes[(nodes >= 0) & (nodes < theta_samples.shape[1])]
    return np.abs(np.exp(1j * theta_samples[:, nodes]).mean(axis=1)).mean()


def frequency_tracking(mat, lam, target_nodes, modules, seed):
    rng = np.random.default_rng(seed)
    n = mat.shape[0]
    omega = 0.5 * rng.normal(size=n)
    omega[target_nodes] += 4.0
    theta = rng.random(n) * 2 * np.pi
    node_freq = np.zeros(n)
    count = 0
    for s in range(3000):
        coupling = np.cos(theta) * (mat @ np.sin(theta)) - np.sin(theta) * (mat @ np.cos(theta))
        drift = omega + lam * coupling
        theta += drift * 0.01 + 0.1 * np.sqrt(0.01) * rng.normal(size=n)
        if s >= 1500:
            node_freq += drift
            count += 1
    node_freq /= max(count, 1)
    mod_freq = np.array([node_freq[nodes[nodes < n]].mean() for nodes in modules.values()])
    return mod_freq, node_freq[target_nodes].mean(), node_freq


def percentile_matrix(x, pct):
    return np.percentile(x, pct, axis=0)


def make_fig07(mat, save=False):
    fig, ax = plt.subplots(figsize=(5.45, 3.10))
    labels = ["Robust synchronization", "Middle state", "Unstable state"]
    for lam, label, col in zip([0.040, 0.023, 0.010], labels, twilight_colors(3)):
        t, r = simulate_trace(mat, lam, 1300, 0.03, 0.2, 20 + len(ax.lines))
        ax.plot(t, r, lw=1.5, color=col, label=label)
    ax.set(xlabel="Time [a.u.]", ylabel=r"$R(t)$", xlim=(0, t[-1]), ylim=(0, 1.02))
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    style_axes(ax)
    if save:
        fig.savefig(FIG_DIR / "07_macroscopic_synchronization_dynamics.png", dpi=300, bbox_inches="tight")
    return fig


def make_fig08(mat, save=False):
    lambdas = np.linspace(0, 0.05, 70)
    rmean, rstd, rlink, snaps = scan_network(mat, lambdas, 900, 480, 0.03, 0.2, 61)
    fig = plt.figure(figsize=(5.40, 5.45))
    ax = fig.add_axes([0.12, 0.61, 0.80, 0.30])
    ax.axvspan(0.015, 0.030, color=(0.16, 0.65, 0.72), alpha=0.10)
    for y, ls, mk, col, lab in zip([rmean, rlink, 5 * rstd], ["-", "-", "--"], ["o", "s", None], twilight_colors(3), [r"$R$", r"$R_{link}$", r"$5\sigma_R$"]):
        ax.plot(lambdas, y, ls, marker=mk, ms=2.4, lw=1.5, color=col, label=lab)
    ax.text(-0.09, 1.05, "A", transform=ax.transAxes, fontname="Arial", fontsize=11)
    ax.set(xlabel=r"$\lambda$", ylabel="Synchronization level", xlim=(0, 0.05), ylim=(0, 1.05))
    ax.legend(loc="upper left", frameon=False)
    style_axes(ax)
    heat_size, heat_y = 0.215, 0.16
    last = None
    for j, (x, target) in enumerate(zip([0.12, 0.385, 0.65], [0.010, 0.0225, 0.035])):
        hax = fig.add_axes([x, heat_y, heat_size, heat_size])
        idx = np.argmin(abs(lambdas - target))
        im = hax.imshow(plv_matrix(snaps[idx]), vmin=0, vmax=1, cmap="viridis")
        hax.axis("off")
        if j == 0:
            hax.text(-0.20, 1.12, "B", transform=hax.transAxes, fontname="Arial", fontsize=11)
        hax.text(0.03, 0.95, rf"$\lambda={target:.4f}$", transform=hax.transAxes, color="w", fontsize=8, va="top")
        last = im
    cax = fig.add_axes([0.90, heat_y, 0.018, heat_size])
    fig.colorbar(last, cax=cax, label="PLV")
    if save:
        fig.savefig(FIG_DIR / "08_lambda_dependent_synchronization_states.png", dpi=300, bbox_inches="tight")
    return fig


def make_fig09(mat, modules, hubs, save=False):
    lambdas = np.linspace(0, 0.05, 56)
    _, _, _, snaps = scan_network(mat, lambdas, 720, 380, 0.03, 0.2, 74)
    names = list(modules)
    vals = np.array([[local_order(s, modules[name]) for s in snaps] for name in names])
    hub_vals = np.array([local_order(s, hubs) for s in snaps])
    global_vals = np.array([np.abs(np.exp(1j * s).mean(axis=1)).mean() for s in snaps])
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.9), constrained_layout=True)
    cols = twilight_colors(len(names))
    for y, name, col in zip(vals, names, cols):
        axes[0].plot(lambdas, y, "-o", ms=2.3, lw=1.5, color=col, label=name.replace("_", " & "))
    axes[0].text(-0.13, 1.05, "A", transform=axes[0].transAxes, fontname="Arial", fontsize=11)
    axes[0].set(xlabel=r"$\lambda$", ylabel="Intramodular synchrony", xlim=(0, 0.05), ylim=(0, 1.03))
    axes[0].legend(loc="lower right", frameon=False, fontsize=7)
    for y in vals:
        axes[1].plot(lambdas, y, lw=1.0, color=(0.82, 0.84, 0.86))
    axes[1].plot(lambdas, hub_vals, "-s", ms=2.8, lw=1.8, color=cols[-1], label="Top 10 hubs")
    axes[1].plot(lambdas, global_vals, "--", lw=1.5, color="k", label="Global")
    axes[1].text(-0.13, 1.05, "B", transform=axes[1].transAxes, fontname="Arial", fontsize=11)
    axes[1].set(xlabel=r"$\lambda$", xlim=(0, 0.05), ylim=(0, 1.03))
    axes[1].legend(loc="lower right", frameon=False)
    for ax in axes:
        style_axes(ax)
    if save:
        fig.savefig(FIG_DIR / "09_modules_and_hubs_synchronization.png", dpi=300, bbox_inches="tight")
    return fig


def make_fig10(mat, modules, hubs, save=False):
    lambdas = np.linspace(0, 0.12, 35)
    names = list(modules)
    rng = np.random.default_rng(42)
    non_hubs = np.setdiff1d(np.arange(mat.shape[0]), hubs)
    groups = [hubs, rng.choice(non_hubs, size=len(hubs), replace=False), modules["Frontal"]]
    group_names = ["Hub Nodes Perturbed", "Random Nodes Perturbed", "Frontal Module Perturbed"]
    fig = plt.figure(figsize=(8.6, 3.4))
    panel_pos = [[0.08, 0.23, 0.245, 0.67], [0.385, 0.23, 0.245, 0.67], [0.690, 0.23, 0.245, 0.67]]
    target_cols, mod_cols = twilight_colors(3), twilight_colors(len(names))
    for gi, target in enumerate(groups):
        ax = fig.add_axes(panel_pos[gi])
        mod_freq, target_freq, node_freq = [], [], []
        for lam in lambdas:
            mf, tf, nf = frequency_tracking(mat, lam, target, modules, 42)
            mod_freq.append(mf)
            target_freq.append(tf)
            node_freq.append(nf)
        mod_freq = np.asarray(mod_freq).T
        node_freq = np.asarray(node_freq).T
        low, high = percentile_matrix(node_freq, 5), percentile_matrix(node_freq, 95)
        ax.fill_between(lambdas, low, high, color=(0.82, 0.84, 0.86), alpha=0.35, lw=0.9)
        ax.plot(lambdas, low, color=(0.82, 0.84, 0.86), lw=0.9)
        ax.plot(lambdas, high, color=(0.82, 0.84, 0.86), lw=0.9)
        for mi, name in enumerate(names):
            if gi == 2 and name == "Frontal":
                continue
            ax.plot(lambdas, mod_freq[mi], lw=1.25, color=mod_cols[mi])
        target_line = mod_freq[names.index("Frontal")] if gi == 2 else np.asarray(target_freq)
        ax.plot(lambdas, target_line, lw=2.3, color=target_cols[gi])
        ax.text(-0.12, 1.03, chr(ord("A") + gi), transform=ax.transAxes, fontname="Arial", fontsize=11)
        ax.set(xlabel=r"Cortical Coupling Factor ($\lambda$)", xlim=(0, 0.12), ylim=(-0.5, 4.5))
        if gi == 0:
            ax.set_ylabel("Frequency")
        style_axes(ax)
        xz, yz = (0.04, 0.08), (-0.20, 1.50)
        ax.add_patch(Rectangle((xz[0], yz[0]), xz[1] - xz[0], yz[1] - yz[0], fill=False, lw=0.9))
        ins = ax.inset_axes([0.60, 0.56, 0.34, 0.33])
        for mi, name in enumerate(names):
            if gi == 2 and name == "Frontal":
                continue
            ins.plot(lambdas, mod_freq[mi], lw=1.25, color=mod_cols[mi])
        ins.plot(lambdas, target_line, lw=2.3, color=target_cols[gi])
        ins.set(xlim=xz, ylim=yz, xticks=[], yticks=[])
        for xy in [(xz[0], yz[0]), (xz[1], yz[0])]:
            fig.add_artist(ConnectionPatch(xyA=xy, coordsA=ax.transData, xyB=(0, 0), coordsB=ins.transAxes, color="0.1", lw=0.8))
    leg_ax = fig.add_axes([0.22, 0.02, 0.58, 0.05])
    leg_ax.axis("off")
    handles = [leg_ax.plot([], [], lw=2.4, color=c)[0] for c in target_cols]
    handles.append(leg_ax.plot([], [], "--", lw=1.8, color="0.35")[0])
    leg_ax.legend(handles, ["Hub Nodes", "Random Nodes", "Frontal Module Nodes", "Functional Modules"], loc="center", ncol=4, frameon=False, fontsize=7.2)
    if save:
        fig.savefig(FIG_DIR / "10_focal_perturbation_frequency_entrainment.png", dpi=300, bbox_inches="tight")
    return fig


def main(save=False):
    mat, _ = load_aal_consensus()
    modules = brain_modules(mat.shape[0])
    hubs = np.argsort(mat.sum(axis=1))[::-1][:10]
    make_fig07(mat, save)
    make_fig08(mat, save)
    make_fig09(mat, modules, hubs, save)
    make_fig10(mat, modules, hubs, save)
    plt.show()


if __name__ == "__main__":
    main(save=False)
