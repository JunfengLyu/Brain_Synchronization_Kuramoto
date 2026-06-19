import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


OUT_DIR = Path(__file__).resolve().parent
OUT_FILE = OUT_DIR / "04_mean_field_bifurcation.png"


def web_colormap(n):
    colors = []
    for i in range(n):
        hue = (190 + 155 * i / max(1, n - 1)) / 360
        lightness = 0.48
        saturation = 0.64
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def gaussian(x, sigma=0.5):
    return np.exp(-(x * x) / (2 * sigma * sigma)) / (np.sqrt(2 * np.pi) * sigma)


def critical_k(sigma=0.5):
    return 2 / (np.pi * gaussian(0, sigma))


def psi_curve(k, r, sigma=0.5):
    x = np.linspace(-1, 1, 2501)
    kernel = np.sqrt(np.maximum(0, 1 - x * x))
    values = gaussian(k * r[:, None] * x[None, :], sigma) * kernel[None, :]
    integral = np.trapezoid(values, x, axis=1)
    return k * integral - 1


def mean_field_r(k, sigma=0.5):
    kc = critical_k(sigma)
    if k <= kc:
        return 0.0

    grid = np.linspace(1e-4, 0.999, 800)
    vals = psi_curve(k, grid, sigma)
    signs = vals[:-1] * vals[1:]
    hits = np.where(signs <= 0)[0]

    if len(hits) == 0:
        return grid[np.argmin(np.abs(vals))]

    lo = grid[hits[-1]]
    hi = grid[hits[-1] + 1]
    flo = psi_curve(k, np.array([lo]), sigma)[0]

    for _ in range(42):
        mid = 0.5 * (lo + hi)
        fmid = psi_curve(k, np.array([mid]), sigma)[0]
        if flo * fmid <= 0:
            hi = mid
        else:
            lo = mid
            flo = fmid

    return 0.5 * (lo + hi)


def finite_sample_r(k, nu, sigma=0.5):
    if k <= critical_k(sigma):
        return 0.0

    def residual(r):
        if r <= 0:
            return -r
        locked = np.abs(nu) <= k * r
        if not np.any(locked):
            return -r
        contribution = np.sqrt(1 - (nu[locked] / (k * r)) ** 2)
        return np.mean(contribution) - r

    grid = np.linspace(1e-4, 0.999, 450)
    vals = np.array([residual(r) for r in grid])
    signs = vals[:-1] * vals[1:]
    hits = np.where(signs <= 0)[0]

    if len(hits) == 0:
        return 0.0

    lo = grid[hits[-1]]
    hi = grid[hits[-1] + 1]
    flo = residual(lo)

    for _ in range(36):
        mid = 0.5 * (lo + hi)
        fmid = residual(mid)
        if flo * fmid <= 0:
            hi = mid
        else:
            lo = mid
            flo = fmid

    return 0.5 * (lo + hi)


def simulate_kuramoto_sweep(n, k_values, sigma=0.5, dt=0.05, total_time=500, burn_time=250, seed=1):
    rng = np.random.default_rng(seed)
    omega = rng.normal(1.0, sigma, size=n)
    theta = rng.uniform(0, 2 * np.pi, size=(len(k_values), n))
    k_col = k_values[:, None]

    steps = round(total_time / dt)
    burn_steps = round(burn_time / dt)
    r_sum = np.zeros(len(k_values))
    r_sq_sum = np.zeros(len(k_values))
    count = 0

    for step in range(steps):
        c = np.cos(theta).mean(axis=1)
        s = np.sin(theta).mean(axis=1)
        r = np.hypot(c, s)
        psi = np.arctan2(s, c)
        theta += dt * (omega[None, :] + k_col * r[:, None] * np.sin(psi[:, None] - theta))

        if step >= burn_steps:
            r_sum += r
            r_sq_sum += r * r
            count += 1

    r_mean = r_sum / count
    r_std = np.sqrt(np.maximum(0, r_sq_sum / count - r_mean * r_mean))
    return r_mean, r_std


def setup_arrow_axes(ax, xlim, ylim, xlabel, ylabel, y_label_style="top"):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["bottom"].set_color("none")
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_position(("data", 0))
    ax.spines["left"].set_color("none")

    ax.tick_params(axis="both", direction="out", length=4.2, width=1.2, pad=5, labelsize=9)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    ax.annotate(
        "",
        xy=(xlim[1], 0),
        xytext=(xlim[0], 0),
        arrowprops=dict(arrowstyle="-|>", lw=1.5, color="black", shrinkA=0, shrinkB=0),
        zorder=0,
    )
    ax.annotate(
        "",
        xy=(0, ylim[1]),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", lw=1.5, color="black", shrinkA=0, shrinkB=0),
        zorder=0,
    )

    ax.text(
        xlim[1] + 0.04 * (xlim[1] - xlim[0]),
        0,
        xlabel,
        ha="left",
        va="center",
        fontsize=13,
        clip_on=False,
    )

    if y_label_style == "left":
        ax.text(
            -0.033 * (xlim[1] - xlim[0]),
            1.02 * ylim[1],
            ylabel,
            ha="right",
            va="center",
            fontsize=13,
            clip_on=False,
        )
    else:
        ax.text(
            0,
            ylim[1] + 0.06 * (ylim[1] - ylim[0]),
            ylabel,
            ha="center",
            va="bottom",
            fontsize=13,
            clip_on=False,
        )
    ax.text(
        -0.035 * (xlim[1] - xlim[0]),
        -0.055 * (ylim[1] - ylim[0]),
        r"$O$",
        ha="right",
        va="top",
        clip_on=False,
    )


def draw_left(ax, sigma=0.5):
    kc = critical_k(sigma)
    r = np.linspace(0, 0.95, 500)
    k_values = [0.82 * kc, kc, 1.22 * kc]
    labels = [r"$K<K_c$", r"$K=K_c$", r"$K>K_c$"]
    colors = ["#2aa9bd", "#3832d0", "#ce2c76"]

    for k, label, color in zip(k_values, labels, colors):
        psi = psi_curve(k, r, sigma)
        ax.plot(psi, r, color=color, lw=1.8, label=label)

    r_star = mean_field_r(k_values[-1], sigma)
    ax.axhline(r_star, color="0.55", lw=1.0, ls="--", alpha=0.65, zorder=0)
    ax.plot(0, r_star, marker="o", ms=5.2, color=colors[-1], markeredgecolor="black", markeredgewidth=0.55)
    ax.annotate(
        r"$R^*(K>K_c)$",
        xy=(0, r_star),
        xytext=(0.08, r_star + 0.08),
        arrowprops=dict(arrowstyle="->", lw=2.0, color="black"),
        fontsize=10,
        ha="left",
        va="bottom",
    )

    setup_arrow_axes(ax, (-0.56, 0.46), (0, 1.06), r"$\psi(K;R)$", r"$R$", y_label_style="left")
    ax.set_xticks([-0.4, -0.2, 0.2, 0.4])
    ax.set_yticks([0.5, 1.0])
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="lower left",
        bbox_to_anchor=(0.03, 0.08),
        handlelength=1.8,
        handletextpad=0.45,
        borderpad=0.2,
    )
    return k_values[-1], r_star


def draw_right(ax, k_high, r_star, sigma=0.5):
    kc = critical_k(sigma)
    k_max = 2.0
    k_grid = np.linspace(0, k_max, 130)
    r_theory = np.array([mean_field_r(k, sigma) for k in k_grid])

    ax.plot(k_grid, r_theory, color="0.45", alpha=0.8, lw=2.0, label="MF prediction")
    ax.axhline(r_star, color="0.55", lw=1.0, ls="--", alpha=0.65, zorder=0)
    ax.plot(k_high, r_star, marker="o", ms=6.0, color="0.22", zorder=4)

    n_values = [10, 100, 1000, 10000]
    colors = web_colormap(len(n_values))
    k_sample = np.linspace(0, k_max, 41)

    for n, color in zip(n_values, colors):
        r_sample, r_err = simulate_kuramoto_sweep(n, k_sample, sigma=sigma, seed=10 + n)
        ax.errorbar(
            k_sample,
            r_sample,
            yerr=r_err,
            fmt="o",
            ms=4.6,
            color=color,
            ecolor=color,
            elinewidth=0.55,
            capsize=1.4,
            markeredgecolor="black",
            markeredgewidth=0.25,
            alpha=0.9,
            label=fr"$N={n}$",
        )

    setup_arrow_axes(ax, (-0.05, k_max + 0.08), (0, 1.06), r"$K$", r"$R$", y_label_style="left")
    ax.set_xticks([0.5, 1.0, 1.5, 2.0])
    ax.set_yticks([0.5, 1.0])
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="center right",
        bbox_to_anchor=(0.98, 0.28),
        handletextpad=0.2,
        borderpad=0.2,
    )


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.17, top=0.88, wspace=0.40)
    k_high, r_star = draw_left(axes[0])
    draw_right(axes[1], k_high, r_star)

    panel_label_positions = [(-0.02, 1.06), (-0.14, 1.06)]
    for label, ax, pos in zip(["A", "B"], axes, panel_label_positions):
        ax.text(
            pos[0],
            pos[1],
            label,
            transform=ax.transAxes,
            fontname="Arial",
            fontstyle="normal",
            fontweight="normal",
            fontsize=14,
            ha="left",
            va="bottom",
        )

    fig.canvas.draw()
    y_fig = fig.transFigure.inverted().transform(axes[0].transData.transform((0, r_star)))[1]
    x0 = axes[0].get_position().x1
    x1 = axes[1].get_position().x0
    fig.add_artist(
        Line2D(
            [x0, x1],
            [y_fig, y_fig],
            transform=fig.transFigure,
            color="0.55",
            lw=1.0,
            ls="--",
            alpha=0.65,
            zorder=2,
        )
    )

    fig.savefig(OUT_FILE, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    print(OUT_FILE)


if __name__ == "__main__":
    main()
