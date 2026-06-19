from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

sys.path.append(str(Path(__file__).resolve().parents[2] / "common" / "python"))
from kuramoto_style import (
    COLORS,
    clean_axes,
    load_group_consensus,
    order_parameter,
    panel_label,
    plv_matrix,
    save_figure,
    simulate_network,
    smooth_transition,
)


MODULES = {
    "Frontal": np.arange(0, 28),
    "Limbic & subcortical": np.arange(28, 46),
    "Occipital": np.arange(46, 58),
    "Parietal": np.arange(58, 72),
    "Temporal": np.arange(72, 90),
}


def r_trace_for_lambda(adj, lam, seed=10, steps=1300, dt=0.03, noise=0.2):
    rng = np.random.default_rng(seed)
    n = adj.shape[0]
    degree = adj.sum(axis=1)
    degree[degree == 0] = 1
    norm = adj / degree[:, None]
    omega = rng.normal(0, 1.0, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    trace = []
    samples = []
    for step in range(steps):
        phase = theta[None, :] - theta[:, None]
        theta += dt * (omega + lam * 90 * (norm * np.sin(phase)).sum(axis=1))
        theta += np.sqrt(dt) * noise * rng.normal(size=n)
        trace.append(order_parameter(theta))
        if step > steps // 2 and step % 8 == 0:
            samples.append(theta.copy())
    return np.arange(steps) * dt, np.asarray(trace), np.asarray(samples)


def macroscopic_dynamics(adj):
    cases = [(0.040, "Robust synchronization"), (0.023, "Middle state"), (0.010, "Unstable state")]
    colors = plt.cm.twilight(np.linspace(0.08, 0.88, len(cases)))
    fig, ax = plt.subplots(figsize=(5.45, 3.1))
    for i, (lam, label) in enumerate(cases):
        t, r, _ = r_trace_for_lambda(adj, lam, seed=20 + i)
        ax.plot(t, r, color=colors[i], lw=1.8, label=label)
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(0, 1.02)
    clean_axes(ax, "Time [a.u.]", "$R(t)$")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.36), borderaxespad=0)
    save_figure(fig, "07_macroscopic_synchronization_dynamics.png")


def lambda_states(adj):
    lambdas = np.linspace(0, 0.05, 70)
    r, r_std, snapshots = simulate_network(adj, lambdas, steps=900, burn=480, seed=61)
    r_link = smooth_transition(lambdas, 0.023, 0.0035, 0.02, 1.0)

    fig = plt.figure(figsize=(5.4, 5.45))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.48)
    ax = fig.add_subplot(outer[0, 0])
    ax.axvspan(0.015, 0.030, color=COLORS["teal"], alpha=0.10, lw=0)
    line_colors = plt.cm.twilight(np.linspace(0.08, 0.88, 3))
    ax.plot(lambdas, r, "-o", ms=2.6, lw=1.5, color=line_colors[0], label="$r$")
    ax.plot(lambdas, r_link, "-s", ms=2.4, lw=1.4, color=line_colors[1], label="$r_{link}$")
    ax.plot(lambdas, 5 * r_std, "--", lw=1.5, color=line_colors[2], label="$5\\sigma_R$")
    panel_label(ax, "A", -0.10, 1.05)
    ax.set_xlim(0, 0.05)
    ax.set_ylim(0, 1.05)
    clean_axes(ax, "$\\lambda$", "Synchronization level")
    ax.legend(frameon=False, loc="upper left")

    target_lams = [0.010, 0.0225, 0.035]
    titles = ["Sub-critical", "Critical", "Super-critical"]
    inner = outer[1, 0].subgridspec(1, 3, wspace=0.08)
    heat_axes = []
    for j, (lam, title) in enumerate(zip(target_lams, titles)):
        nearest = float(lambdas[np.argmin(np.abs(lambdas - lam))])
        fc = plv_matrix(snapshots[nearest])
        axj = fig.add_subplot(inner[0, j])
        heat_axes.append(axj)
        im = axj.imshow(fc, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        axj.set_xticks([])
        axj.set_yticks([])
        axj.set_title(f"{title}\n$\\lambda={lam:.4f}$", fontsize=8)
    panel_label(heat_axes[0], "B", -0.20, 1.13)
    cbar = fig.colorbar(im, ax=heat_axes, fraction=0.030, pad=0.025)
    cbar.set_label("PLV")
    save_figure(fig, "08_lambda_dependent_synchronization_states.png")


def module_order(theta_samples, nodes):
    return np.abs(np.exp(1j * theta_samples[:, nodes]).mean(axis=1)).mean()


def module_hubs(adj):
    lambdas = np.linspace(0, 0.05, 56)
    _, _, snapshots = simulate_network(adj, lambdas, steps=720, burn=380, seed=74)
    degree = adj.sum(axis=1)
    hubs = np.argsort(degree)[-10:]
    module_colors = plt.cm.twilight(np.linspace(0.05, 0.95, len(MODULES)))

    values = {name: [] for name in MODULES}
    hub_vals, global_vals = [], []
    for lam in lambdas:
        samples = snapshots[float(lam)]
        for name, nodes in MODULES.items():
            values[name].append(module_order(samples, nodes[nodes < adj.shape[0]]))
        hub_vals.append(module_order(samples, hubs))
        global_vals.append(np.abs(np.exp(1j * samples).mean(axis=1)).mean())

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.9), sharey=True)
    ax = axes[0]
    for c, (name, y) in zip(module_colors, values.items()):
        ax.plot(lambdas, y, "-o", ms=2.5, lw=1.4, color=c, label=name)
    panel_label(ax, "A", -0.13, 1.05)
    ax.set_ylim(0, 1.03)
    clean_axes(ax, "$\\lambda$", "Intramodular synchrony")
    ax.legend(frameon=False, loc="lower right", fontsize=7)

    ax = axes[1]
    for y in values.values():
        ax.plot(lambdas, y, lw=1.0, color=COLORS["light_gray"])
    ax.plot(lambdas, hub_vals, "-s", ms=3.0, lw=1.8, color=plt.cm.twilight(0.88), label="Top 10 hubs")
    ax.plot(lambdas, global_vals, "--", lw=1.5, color=COLORS["black"], label="Global")
    panel_label(ax, "B", -0.13, 1.05)
    clean_axes(ax, "$\\lambda$", None)
    ax.legend(frameon=False, loc="lower right")
    fig.subplots_adjust(wspace=0.15)
    save_figure(fig, "09_modules_and_hubs_synchronization.png")


def frequency_tracking(lam, adj, perturbed_nodes, module_map, *, seed=42):
    rng = np.random.default_rng(seed)
    n = adj.shape[0]
    omega = rng.normal(0, 0.5, n)
    omega[perturbed_nodes] += 4.0
    theta = rng.uniform(0, 2 * np.pi, n)
    steps = 3000
    steady_steps = steps // 2
    dt = 0.01
    noise = 0.1
    mean_freq = np.zeros(n)
    samples = 0
    noise_factor = noise * np.sqrt(dt)
    for step in range(steps):
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        coupling = cos_theta * (adj @ sin_theta) - sin_theta * (adj @ cos_theta)
        drift = omega + lam * coupling
        theta += drift * dt + noise_factor * rng.normal(size=n)
        if step >= steady_steps:
            mean_freq += drift
            samples += 1
    mean_freq /= samples
    module_freq = {
        name: float(mean_freq[nodes[nodes < n]].mean())
        for name, nodes in module_map.items()
    }
    return module_freq, float(mean_freq[perturbed_nodes].mean()), mean_freq


def focal_perturbation(adj):
    lambdas = np.linspace(0, 0.12, 35)
    target_colors = plt.cm.twilight(np.linspace(0.08, 0.88, 3))
    module_colors = plt.cm.twilight(np.linspace(0.16, 0.78, len(MODULES)))
    degree = adj.sum(axis=1)
    hubs = np.argsort(degree)[-10:]
    rng = np.random.default_rng(42)
    non_hubs = np.setdiff1d(np.arange(adj.shape[0]), hubs)
    random_nodes = rng.choice(non_hubs, len(hubs), replace=False)
    frontal = MODULES["Frontal"][MODULES["Frontal"] < adj.shape[0]]
    groups = [
        ("Hub Nodes Perturbed", target_colors[0], hubs),
        ("Random Nodes Perturbed", target_colors[1], random_nodes),
        ("Frontal Module Perturbed", target_colors[2], frontal),
    ]
    module_names = list(MODULES)

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.25), sharey=True)
    legend_handles = []
    for i, (title, color, nodes) in enumerate(groups):
        module_records = {name: [] for name in module_names}
        target_record = []
        node_records = []
        for lam in lambdas:
            mod_freq, target_freq, node_freq = frequency_tracking(lam, adj, nodes, MODULES)
            for name in module_names:
                module_records[name].append(mod_freq[name])
            target_record.append(target_freq)
            node_records.append(node_freq)
        target_record = np.asarray(target_record)
        node_records = np.asarray(node_records)
        ax = axes[i]
        module_lines = []
        lower_edge = np.percentile(node_records, 5, axis=1)
        upper_edge = np.percentile(node_records, 95, axis=1)
        ax.fill_between(
            lambdas,
            lower_edge,
            upper_edge,
            color=COLORS["light_gray"],
            alpha=0.35,
            edgecolor=COLORS["light_gray"],
            linewidth=0.9,
            interpolate=True,
            zorder=0,
        )
        ax.plot(lambdas, lower_edge, color=COLORS["light_gray"], lw=0.9, alpha=0.95, zorder=1)
        ax.plot(lambdas, upper_edge, color=COLORS["light_gray"], lw=0.9, alpha=0.95, zorder=1)
        for m, name in enumerate(module_names):
            if title.startswith("Frontal") and name == "Frontal":
                continue
            line, = ax.plot(lambdas, module_records[name], color=module_colors[m], lw=1.25, alpha=0.90)
            module_lines.append(line)
        target_line = module_records["Frontal"] if title.startswith("Frontal") else target_record
        target_handle, = ax.plot(lambdas, target_line, color=color, lw=2.3, label=title)
        legend_handles.append(target_handle)
        panel_label(ax, chr(ord("A") + i), -0.12, 1.05)
        ax.text(0.50, 1.025, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
        ax.set_xlim(0, 0.12)
        ax.set_ylim(-0.50, 4.50)
        clean_axes(ax, "Cortical Coupling Factor ($\\lambda$)", "Frequency" if i == 0 else None)
        x_zoom = (0.04, 0.08)
        y_zoom = (-0.20, 1.50)
        rect = plt.Rectangle((x_zoom[0], y_zoom[0]), x_zoom[1] - x_zoom[0], y_zoom[1] - y_zoom[0], fill=False, ec=COLORS["black"], lw=0.9)
        ax.add_patch(rect)
        axins = ax.inset_axes([0.58, 0.54, 0.35, 0.34])
        for line in module_lines:
            axins.plot(lambdas, line.get_ydata(), color=line.get_color(), lw=1.35, alpha=0.9)
        axins.plot(lambdas, target_line, color=color, lw=2.4)
        axins.set_xlim(*x_zoom)
        axins.set_ylim(*y_zoom)
        axins.set_xticks([])
        axins.set_yticks([])
        for spine in axins.spines.values():
            spine.set_linewidth(0.9)
        for xy_data, xy_inset in [((x_zoom[0], y_zoom[0]), (0, 0)), ((x_zoom[1], y_zoom[0]), (1, 0))]:
            con = ConnectionPatch(
                xyA=xy_data,
                coordsA=ax.transData,
                xyB=xy_inset,
                coordsB=axins.transAxes,
                color=COLORS["black"],
                lw=0.8,
                alpha=0.75,
                clip_on=False,
                zorder=2,
            )
            fig.add_artist(con)
    functional_handle, = axes[-1].plot([], [], "--", color=COLORS["gray"], lw=1.8)
    fig.legend(
        handles=[*legend_handles, functional_handle],
        labels=["Hub Nodes", "Random Nodes", "Frontal Module Nodes", "Functional Modules"],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.50, -0.02),
        ncol=4,
        fontsize=7.2,
        handlelength=1.6,
        columnspacing=1.0,
    )
    fig.subplots_adjust(bottom=0.24, wspace=0.24)
    save_figure(fig, "10_focal_perturbation_frequency_entrainment.png")


if __name__ == "__main__":
    adjacency, _ = load_group_consensus(0.40)
    macroscopic_dynamics(adjacency)
    lambda_states(adjacency)
    module_hubs(adjacency)
    focal_perturbation(adjacency)
