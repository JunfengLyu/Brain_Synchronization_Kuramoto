from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

sys.path.append(str(Path(__file__).resolve().parents[2] / "common" / "python"))
from kuramoto_style import COLORS, clean_axes, panel_label, save_figure


def dzdt_complex(z, k, f, delta, omega):
    return 0.5 * ((k * z + f) - z**2 * (k * np.conj(z) + f)) - (delta + 1j * omega) * z


def rhs_real(_, y, k, f, delta, omega):
    z = y[0] + 1j * y[1]
    dz = dzdt_complex(z, k, f, delta, omega)
    return [dz.real, dz.imag]


def fixed_points(k, f, delta, omega):
    pts = []
    grid = np.linspace(-0.92, 0.92, 13)
    for x in grid:
        for y in grid:
            if x * x + y * y > 1:
                continue
            sol = root(lambda yy: rhs_real(0, yy, k, f, delta, omega), [x, y], method="hybr")
            if sol.success:
                z = sol.x[0] + 1j * sol.x[1]
                if abs(z) <= 1.01 and all(np.linalg.norm(sol.x - p) > 1e-4 for p in pts):
                    pts.append(sol.x)
    return np.asarray(pts)


def jacobian(z, k, f, delta, omega, eps=1e-5):
    y = np.array([z.real, z.imag])
    j = np.zeros((2, 2))
    for i in range(2):
        e = np.zeros(2)
        e[i] = eps
        j[:, i] = (np.asarray(rhs_real(0, y + e, k, f, delta, omega)) - np.asarray(rhs_real(0, y - e, k, f, delta, omega))) / (2 * eps)
    return j


def stable_point(k, f, delta, omega):
    pts = fixed_points(k, f, delta, omega)
    for p in pts:
        eig = np.linalg.eigvals(jacobian(p[0] + 1j * p[1], k, f, delta, omega))
        if np.all(np.real(eig) < 0):
            return p[0] + 1j * p[1], pts
    return pts[0, 0] + 1j * pts[0, 1], pts


def trajectory(z0, t_end, k, f, delta, omega, n=1800, max_step=np.inf):
    sol = solve_ivp(
        lambda t, y: rhs_real(t, y, k, f, delta, omega),
        (0, t_end),
        [z0.real, z0.imag],
        t_eval=np.linspace(0, t_end, n),
        rtol=1e-8,
        atol=1e-10,
        max_step=max_step,
    )
    return sol.t, sol.y[0] + 1j * sol.y[1]


def recovery_time(z_st, shift_hours, k, f, delta, omega, threshold=0.2, t_max_days=60):
    z0 = z_st * np.exp(1j * shift_hours * 2 * np.pi / 24)
    t, z = trajectory(z0, 24 * t_max_days, k, f, delta, omega, n=1200)
    d = np.abs(z - z_st)
    idx = np.where(d <= threshold)[0]
    return t[idx[0]] / 24 if len(idx) else np.nan


def phase_space():
    delta = 1.0
    f = 3.5 * delta
    omega = 1.4 * delta
    regimes = [(10.0 * delta, "$K=10\\Delta$"), (4.5 * delta, "$K=4.5\\Delta$")]
    twilight = plt.cm.twilight(np.linspace(0.08, 0.88, 4))
    shifts = [(-8.5, twilight[0], "8.5 h eastward"), (-9.5, twilight[1], "9.5 h eastward"), (9.0, twilight[2], "9.0 h westward"), (12.0, twilight[3], "12 h E/W")]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharex=True, sharey=True)
    theta = np.linspace(0, 2 * np.pi, 500)
    for i, (k, label) in enumerate(regimes):
        ax = axes[i]
        z_st, pts = stable_point(k, f, delta, omega)
        ax.plot(np.cos(theta), np.sin(theta), "--", lw=1.0, color=COLORS["light_gray"])
        ax.axhline(0, color=COLORS["light_gray"], lw=0.9)
        ax.axvline(0, color=COLORS["light_gray"], lw=0.9)
        for radius in [0.25, 0.55, 0.85]:
            for angle in np.linspace(0, 2 * np.pi, 18, endpoint=False):
                z0 = radius * np.exp(1j * angle)
                _, z_flow = trajectory(z0, 25, k, f, delta, omega, n=1200, max_step=0.035)
                ax.plot(z_flow.real, z_flow.imag, color=COLORS["light_gray"], lw=0.55, alpha=0.50, zorder=0)
        for hours, color, text in shifts:
            z0 = z_st * np.exp(1j * hours * 2 * np.pi / 24)
            _, z = trajectory(z0, 80, k, f, delta, omega, n=2600, max_step=0.035)
            ax.plot(z.real, z.imag, lw=1.5, color=color, label=text)
            ax.plot(z.real[0], z.imag[0], "o", ms=3.2, color=color)
        for p in pts:
            z = p[0] + 1j * p[1]
            eig = np.linalg.eigvals(jacobian(z, k, f, delta, omega))
            if np.all(np.real(eig) < 0):
                ax.plot(p[0], p[1], "p", ms=7, color=COLORS["black"])
            elif np.all(np.real(eig) > 0):
                ax.plot(p[0], p[1], "o", ms=5, mfc="white", mec=COLORS["black"])
            else:
                ax.plot(p[0], p[1], "+", ms=8, mew=1.5, color=COLORS["black"])
        panel_label(ax, chr(ord("A") + i), -0.13, 1.04)
        ax.text(0.05, 0.92, label, transform=ax.transAxes, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)
        clean_axes(ax, "$\\operatorname{Re}(z)$", "$\\operatorname{Im}(z)$" if i == 0 else None)
    axes[0].legend(frameon=False, loc="lower center", bbox_to_anchor=(1.04, -0.33), ncol=4)
    save_figure(fig, "15_circadian_phase_space_dynamics.png")


def recovery_curves():
    delta = 3.8e-3
    k, f, omega = 4.5 * delta, 3.5 * delta, 1.4 * delta
    z_st, _ = stable_point(k, f, delta, omega)
    twilight = plt.cm.twilight(np.linspace(0.08, 0.88, 4))
    cases = [(-3, twilight[0], "3 E", "-"), (-6, twilight[1], "6 E", "-"), (-9, twilight[2], "9 E", "-"), (12, twilight[3], "12 E/W", "-"), (3, twilight[0], "3 W", "--"), (6, twilight[1], "6 W", "--"), (9, twilight[2], "9 W", "--")]
    fig, ax = plt.subplots(figsize=(4.7, 3.35))
    for hours, color, label, ls in cases:
        z0 = z_st * np.exp(1j * hours * 2 * np.pi / 24)
        t, z = trajectory(z0, 14 * 24, k, f, delta, omega, n=900)
        d = np.abs(z - z_st)
        keep = np.arange(len(d))
        reached = np.where(d <= 0.2)[0]
        if len(reached):
            keep = keep[: reached[0] + 1]
        ax.plot(t[keep] / 24, d[keep], ls=ls, lw=1.6, color=color, label=label)
    ax.axhline(0.2, color=COLORS["black"], lw=1.1)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 2.0)
    clean_axes(ax, "Days", "$|z(t)-z_{st}|$")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    save_figure(fig, "16_circadian_recovery_trajectories.png")


def parameter_dependence():
    delta = 3.8e-3
    k_ref, f_ref, omega_ref = 4.5 * delta, 3.5 * delta, 1.4 * delta
    twilight = plt.cm.twilight(np.linspace(0.08, 0.88, 4))
    cases = [(-3, twilight[0], "3 E", "-"), (-6, twilight[1], "6 E", "-"), (-9, twilight[2], "9 E", "-"), (12, twilight[3], "12 E/W", "-"), (3, twilight[0], "3 W", "--"), (6, twilight[1], "6 W", "--"), (9, twilight[2], "9 W", "--")]
    scans = [
        ("$K/\\Delta$", np.linspace(2.0, 15.0, 72), lambda x: (x * delta, f_ref, omega_ref), (2, 24)),
        ("$F/\\Delta$", np.linspace(1.5, 5.8, 70), lambda x: (k_ref, x * delta, omega_ref), (0, 45)),
        ("$\\Omega/\\Delta$", np.linspace(-3.7, 3.7, 78), lambda x: (k_ref, f_ref, x * delta), (0, 34)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.8))
    for j, (xlabel, xs, params, ylim) in enumerate(scans):
        ax = axes[j]
        for shift, color, label, ls in cases:
            rec = []
            for x in xs:
                k, f, omega = params(x)
                z_st, _ = stable_point(k, f, delta, omega)
                rec.append(recovery_time(z_st, shift, k, f, delta, omega, threshold=0.2, t_max_days=28))
            ax.plot(xs, rec, ls=ls, lw=1.25, color=color, label=label)
        panel_label(ax, chr(ord("A") + j), -0.15, 1.04)
        ax.set_ylim(*ylim)
        clean_axes(ax, xlabel, "Recovery time (days)" if j == 0 else None)
        if j == 1:
            ax.legend(frameon=False, loc="upper right", fontsize=6.4, ncol=2)
    fig.subplots_adjust(wspace=0.28)
    save_figure(fig, "17_circadian_parameter_dependence.png")


if __name__ == "__main__":
    phase_space()
    recovery_curves()
    parameter_dependence()
