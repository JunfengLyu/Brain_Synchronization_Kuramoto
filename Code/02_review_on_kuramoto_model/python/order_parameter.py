import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch


OUT_DIR = Path(__file__).resolve().parent
OUT_FILE = OUT_DIR / "02_order_parameter_schematic.png"


def web_colormap(n):
    """Match the HSL phase colors used in the interactive webpage."""
    colors = []
    for i in range(n):
        hue = (190 + 155 * i / max(1, n - 1)) / 360
        lightness = 0.48
        saturation = 0.64
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def order_parameter(theta):
    z = np.mean(np.exp(1j * theta))
    return np.abs(z), np.angle(z)


def simulate_kuramoto(omega, theta0, k, dt=0.02, total_time=200):
    theta = theta0.copy()
    steps = round(total_time / dt)

    for _ in range(steps):
        z = np.mean(np.exp(1j * theta))
        r = np.abs(z)
        psi = np.angle(z)
        theta += dt * (omega + k * r * np.sin(psi - theta))

    return theta


def draw_panel(ax, theta, colors, title):
    radius = 1.0
    R, Phi = order_parameter(theta)

    ax.add_patch(Circle((0, 0), radius, fill=False, lw=1.7, color="black", alpha=0.82))

    for angle, color in zip(theta, colors):
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        ax.add_patch(
            FancyArrowPatch(
                (0, 0),
                (0.92 * x, 0.92 * y),
                arrowstyle="-|>",
                mutation_scale=9,
                lw=0.9,
                color="0.68",
                alpha=0.88,
                zorder=1,
            )
        )
        ax.scatter(x, y, s=95, color=color, edgecolor="black", linewidth=0.75, zorder=3)

    zx = R * np.cos(Phi)
    zy = R * np.sin(Phi)
    ax.add_patch(
        FancyArrowPatch(
            (0, 0),
            (zx, zy),
            arrowstyle="-|>",
            mutation_scale=17,
            lw=2.0,
            color="black",
            zorder=4,
        )
    )

    label_angle = Phi - np.pi / 2 + np.deg2rad(10)
    label_radius = 0.16
    label_x = 0.5 * zx + label_radius * np.cos(label_angle)
    label_y = 0.5 * zy + label_radius * np.sin(label_angle)
    ax.text(
        label_x,
        label_y,
        r"$\tilde{z}(t)$",
        fontsize=15,
        ha="center",
        va="center",
        color="black",
    )

    ax.set_title(title, fontsize=16, pad=14)
    ax.set_aspect("equal")
    ax.set_xlim(-1.24, 1.24)
    ax.set_ylim(-1.24, 1.24)
    ax.axis("off")


def main():
    n = 10
    rng = np.random.default_rng(1)

    omega = rng.normal(loc=1.0, scale=np.sqrt(1 / 4), size=n)
    order = np.argsort(omega)
    omega = omega[order]
    theta0 = rng.uniform(0, 2 * np.pi, size=n)[order]

    colors = web_colormap(n)
    k_values = [0.1, 1.2]
    theta_by_k = [simulate_kuramoto(omega, theta0, k) for k in k_values]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.1), constrained_layout=True)
    for ax, theta, k in zip(axes, theta_by_k, k_values):
        draw_panel(ax, theta, colors, rf"$K={k:.1f}$")

    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT_FILE)


if __name__ == "__main__":
    main()
