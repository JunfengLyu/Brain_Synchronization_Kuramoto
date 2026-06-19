import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent
OUT_FILE = OUT_DIR / "03_phase_diagram.png"


def web_colormap(n):
    colors = []
    for i in range(n):
        hue = (190 + 155 * i / max(1, n - 1)) / 360
        lightness = 0.48
        saturation = 0.64
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def fixed_points(nu):
    if abs(nu) > 1:
        return []

    return [np.arcsin(nu)]


def main():
    kr = 1.0
    nu_m = 1.4
    step = 0.2

    nus = np.round(np.arange(nu_m, -nu_m - step / 2, -step), 10)
    colors = web_colormap(len(nus))
    phi = np.linspace(-np.pi, np.pi, 900)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))

    for nu, color in zip(nus, colors):
        phidot = nu - kr * np.sin(phi)
        has_fixed_point = abs(nu) <= kr
        ax.plot(
            phi,
            phidot,
            color=color,
            lw=1,
            ls="-" if has_fixed_point else "--",
        )

        for fp in fixed_points(nu / kr):
            ax.plot(
                fp,
                0,
                marker="o",
                ms=4.2,
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.45,
                zorder=3,
            )

    xlim = (-3.35, 3.35)
    ylim = (-2.65, 2.65)
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

    ax.tick_params(axis="both", direction="out", length=4.2, width=1.2, pad=6)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.set_xticks([-3, -2, -1, 1, 2, 3])
    ax.set_yticks([-2, -1, 1, 2])

    ax.annotate(
        "",
        xy=(xlim[1] - 0.02, 0),
        xytext=(xlim[0], 0),
        arrowprops=dict(arrowstyle="-|>", lw=1.5, color="black", shrinkA=0, shrinkB=0),
        zorder=0,
    )
    ax.annotate(
        "",
        xy=(0, ylim[1] - 0.02),
        xytext=(0, ylim[0]),
        arrowprops=dict(arrowstyle="-|>", lw=1.5, color="black", shrinkA=0, shrinkB=0),
        zorder=0,
    )

    ax.text(xlim[1] + 0.16, -0.08, r"$\phi$", ha="left", va="top", fontsize=12, clip_on=False)
    ax.text(0, ylim[1] + 0.2, r"$\dot{\phi}$", ha="center", va="bottom", fontsize=12, clip_on=False)
    ax.annotate(
        r"$O$",
        xy=(0, 0),
        xytext=(-10, -11),
        textcoords="offset points",
        ha="right",
        va="top",
        clip_on=False,
    )

    fig.savefig(OUT_FILE, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    print(OUT_FILE)


if __name__ == "__main__":
    main()
