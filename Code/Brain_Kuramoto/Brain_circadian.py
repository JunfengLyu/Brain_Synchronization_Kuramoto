from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


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


def dzdt_complex(z, k, f, delta, omega):
    return 0.5 * ((k * z + f) - z * z * (k * np.conj(z) + f)) - (delta + 1j * omega) * z


def rhs_real(_, y, k, f, delta, omega):
    z = y[0] + 1j * y[1]
    dz = dzdt_complex(z, k, f, delta, omega)
    return np.array([dz.real, dz.imag])


def rhs_norm(y, k, f, delta, omega):
    y = np.asarray(y)
    if np.sum(y * y) > 1.08:
        return 1e3 + float(np.sum(y * y))
    r = rhs_real(0, y, k, f, delta, omega)
    return float(r @ r)


def stable_point_fast(k, f, delta, omega):
    starts = np.array([[0.5, 0], [0, 0], [-0.5, 0], [0, 0.5], [0, -0.5], [0.8, 0.2], [-0.2, 0.8]])
    best, best_val = starts[0], np.inf
    for start in starts:
        res = minimize(lambda yy: rhs_norm(yy, k, f, delta, omega), start, method="Nelder-Mead", options={"maxiter": 800, "xatol": 1e-10, "fatol": 1e-12})
        val = rhs_norm(res.x, k, f, delta, omega)
        if np.linalg.norm(res.x) <= 1.02 and val < best_val:
            best, best_val = res.x, val
    return best[0] + 1j * best[1]


def fixed_points_all(k, f, delta, omega):
    pts = []
    grid = np.linspace(-0.92, 0.92, 11)
    for x in grid:
        for y in grid:
            if x * x + y * y > 1:
                continue
            res = minimize(lambda yy: rhs_norm(yy, k, f, delta, omega), [x, y], method="Nelder-Mead", options={"maxiter": 600, "xatol": 1e-10, "fatol": 1e-12})
            p = res.x
            if np.linalg.norm(p) <= 1.02 and rhs_norm(p, k, f, delta, omega) < 1e-9:
                if not pts or np.all(np.linalg.norm(np.asarray(pts) - p, axis=1) > 1e-4):
                    pts.append(p)
    if not pts:
        z = stable_point_fast(k, f, delta, omega)
        pts.append([z.real, z.imag])
    return np.asarray(pts)


def jacobian(z, k, f, delta, omega):
    eps = 1e-5
    y = np.array([z.real, z.imag])
    mat = np.zeros((2, 2))
    for i in range(2):
        e = np.zeros(2)
        e[i] = eps
        mat[:, i] = (rhs_real(0, y + e, k, f, delta, omega) - rhs_real(0, y - e, k, f, delta, omega)) / (2 * eps)
    return mat


def trajectory(z0, t_end, k, f, delta, omega, n, max_step=np.inf):
    kwargs = {"rtol": 1e-8, "atol": 1e-10}
    if np.isfinite(max_step):
        kwargs["max_step"] = max_step
    t_eval = np.linspace(0, t_end, n)
    sol = solve_ivp(rhs_real, (0, t_end), [z0.real, z0.imag], t_eval=t_eval, args=(k, f, delta, omega), **kwargs)
    z = sol.y[0] + 1j * sol.y[1]
    return sol.t, z


def recovery_time(zst, shift_hours, k, f, delta, omega, threshold, max_days):
    z0 = zst * np.exp(1j * shift_hours * 2 * np.pi / 24)
    t, z = trajectory(z0, 24 * max_days, k, f, delta, omega, 850)
    hit = np.flatnonzero(np.abs(z - zst) <= threshold)
    return np.nan if len(hit) == 0 else t[hit[0]] / 24


def make_fig15(save=False):
    delta = 1.0
    f, omega = 3.5 * delta, 1.4 * delta
    regimes = [10.0 * delta, 4.5 * delta]
    shifts = [-8.5, -9.5, 9.0, 12.0]
    cols = twilight_colors(len(shifts))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), constrained_layout=True)
    theta = np.linspace(0, 2 * np.pi, 500)
    for ri, (ax, k) in enumerate(zip(axes, regimes)):
        zst = stable_point_fast(k, f, delta, omega)
        pts = fixed_points_all(k, f, delta, omega)
        ax.plot(np.cos(theta), np.sin(theta), "--", lw=1.0, color=(0.82, 0.84, 0.86))
        ax.axhline(0, color=(0.82, 0.84, 0.86), lw=0.9)
        ax.axvline(0, color=(0.82, 0.84, 0.86), lw=0.9)
        for rad in [0.25, 0.55, 0.85]:
            for ang in np.linspace(0, 2 * np.pi, 18):
                _, z = trajectory(rad * np.exp(1j * ang), 25, k, f, delta, omega, 1200, 0.035)
                ax.plot(z.real, z.imag, color=(0.82, 0.84, 0.86), lw=0.55)
        for shift, col in zip(shifts, cols):
            z0 = zst * np.exp(1j * shift * 2 * np.pi / 24)
            _, z = trajectory(z0, 80, k, f, delta, omega, 2600, 0.035)
            ax.plot(z.real, z.imag, lw=1.5, color=col)
            ax.plot(z[0].real, z[0].imag, "o", ms=3.2, color=col)
        for p in pts:
            z = p[0] + 1j * p[1]
            ev = np.linalg.eigvals(jacobian(z, k, f, delta, omega))
            if np.all(ev.real < 0):
                ax.plot(z.real, z.imag, "p", ms=7, color="k")
            elif np.all(ev.real > 0):
                ax.plot(z.real, z.imag, "o", ms=5, markerfacecolor="w", markeredgecolor="k")
            else:
                ax.plot(z.real, z.imag, "+", ms=8, mew=1.5, color="k")
        ax.text(-0.13, 1.04, chr(ord("A") + ri), transform=ax.transAxes, fontname="Arial", fontsize=11)
        ax.text(0.05, 0.92, rf"$K={10 if ri == 0 else 4.5}\Delta$", transform=ax.transAxes, fontsize=9)
        ax.set_aspect("equal")
        ax.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02), xlabel=r"$\operatorname{Re}(z)$")
        if ri == 0:
            ax.set_ylabel(r"$\operatorname{Im}(z)$")
        style_axes(ax)
    if save:
        fig.savefig(FIG_DIR / "15_circadian_phase_space_dynamics.png", dpi=300, bbox_inches="tight")
    return fig


def make_fig16(save=False):
    delta = 3.8e-3
    k, f, omega = 4.5 * delta, 3.5 * delta, 1.4 * delta
    zst = stable_point_fast(k, f, delta, omega)
    cases = [-3, -6, -9, 12, 3, 6, 9]
    labels = ["3 E", "6 E", "9 E", "12 E/W", "3 W", "6 W", "9 W"]
    styles = ["-", "-", "-", "-", "--", "--", "--"]
    base = twilight_colors(4)
    cols = [base[0], base[1], base[2], base[3], base[0], base[1], base[2]]
    fig, ax = plt.subplots(figsize=(4.7, 3.35))
    for case, label, ls, col in zip(cases, labels, styles, cols):
        t, z = trajectory(zst * np.exp(1j * case * 2 * np.pi / 24), 14 * 24, k, f, delta, omega, 900)
        d = np.abs(z - zst)
        stop = np.flatnonzero(d <= 0.2)
        end = stop[0] + 1 if len(stop) else len(d)
        ax.plot(t[:end] / 24, d[:end], ls, lw=1.6, color=col, label=label)
    ax.axhline(0.2, color="k", lw=1.1)
    ax.set(xlabel="Days", ylabel=r"$|z(t)-z_{st}|$", xlim=(0, 14), ylim=(0, 2.0))
    ax.legend(loc="upper right", ncol=2, frameon=False)
    style_axes(ax)
    if save:
        fig.savefig(FIG_DIR / "16_circadian_recovery_trajectories.png", dpi=300, bbox_inches="tight")
    return fig


def make_fig17(save=False):
    delta = 3.8e-3
    k_ref, f_ref, omega_ref = 4.5 * delta, 3.5 * delta, 1.4 * delta
    cases = [-3, -6, -9, 12, 3, 6, 9]
    labels = ["3 E", "6 E", "9 E", "12 E/W", "3 W", "6 W", "9 W"]
    styles = ["-", "-", "-", "-", "--", "--", "--"]
    base = twilight_colors(4)
    cols = [base[0], base[1], base[2], base[3], base[0], base[1], base[2]]
    scans = [
        (r"$K/\Delta$", np.linspace(2.0, 15.0, 72), (2, 24), 1),
        (r"$F/\Delta$", np.linspace(1.5, 5.8, 70), (0, 45), 2),
        (r"$\Omega/\Delta$", np.linspace(-3.7, 3.7, 78), (0, 34), 3),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.8), constrained_layout=True)
    for si, (ax, (xlabel, xs, ylim, mode)) in enumerate(zip(axes, scans)):
        for case, label, ls, col in zip(cases, labels, styles, cols):
            rec = []
            for x in xs:
                if mode == 1:
                    k, f, omega = x * delta, f_ref, omega_ref
                elif mode == 2:
                    k, f, omega = k_ref, x * delta, omega_ref
                else:
                    k, f, omega = k_ref, f_ref, x * delta
                zst = stable_point_fast(k, f, delta, omega)
                rec.append(recovery_time(zst, case, k, f, delta, omega, 0.2, 28))
            ax.plot(xs, rec, ls, lw=1.25, color=col, label=label)
        ax.text(-0.15, 1.04, chr(ord("A") + si), transform=ax.transAxes, fontname="Arial", fontsize=11)
        ax.set(xlabel=xlabel, ylim=ylim)
        if si == 0:
            ax.set_ylabel("Recovery time (days)")
        if si == 1:
            ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=6.4)
        style_axes(ax)
    if save:
        fig.savefig(FIG_DIR / "17_circadian_parameter_dependence.png", dpi=300, bbox_inches="tight")
    return fig


def main(save=False):
    make_fig15(save)
    make_fig16(save)
    make_fig17(save)
    plt.show()


if __name__ == "__main__":
    main(save=False)
