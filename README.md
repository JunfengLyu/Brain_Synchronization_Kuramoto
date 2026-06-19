# Brain Synchronization with the Kuramoto Model

This repository contains the code, figures, data, and interactive project page for a course project on emergent synchronization in brain networks. The project starts from the classical Kuramoto model, extends it to heterogeneous network coupling, and then applies the framework to empirical brain connectomes, Alzheimer's disease progression, and circadian resynchronization.

The online companion page is served through GitHub Pages from `docs/`.

## Project Structure

The main reproducible entry points are kept in the repository root:

- `01_Kuramoto_demo.py` / `01_Kuramoto_demo.m`: classical Kuramoto dynamics, order parameter, mean-field bifurcation, and heterogeneous network topology comparison.
- `02_Brain_Kuramoto.py` / `02_Brain_Kuramoto.m`: AAL connectome construction, whole-brain synchronization, hub perturbation, and Alzheimer's disease continuum simulations.
- `03_Circadian_clock.py` / `03_Circadian_clock.m`: forced Kuramoto/Ott-Antonsen circadian resynchronization analysis.
- `data_for_section3&4/`: AAL and ADNI-derived structural connectivity datasets used by the brain-connectome and AD sections.
- `Report/Figs/`: final report figures saved as 300 dpi PNG files.
- `docs/`: interactive GitHub Pages site with paper text, figures, browser demos, and minimal code snippets.

The original section-level scripts remain in `Code/` for traceability. The root scripts are the recommended interfaces for reproducing the report figures.

## Core Models

### Classical Kuramoto Model

For `N` coupled oscillators, the phase dynamics are

```text
d theta_i / dt = omega_i + (K / N) sum_j sin(theta_j - theta_i).
```

The complex order parameter is

```text
z(t) = (1 / N) sum_j exp(i theta_j) = R(t) exp(i Theta(t)),
```

where `R(t)` measures the population synchronization level.

Under the mean-field approximation and the rotating coordinate `phi_i = theta_i - omega_bar t`, the reduced one-dimensional equation is

```text
d phi_i / dt = nu_i - K R sin(phi_i).
```

Locked oscillators satisfy `|nu_i| <= K R`, and the Gaussian mean-field critical coupling is

```text
K_c = 2 / (pi g(omega_bar)).
```

### Heterogeneous Networks

To model topology-dependent synchronization, homogeneous all-to-all coupling is replaced by a network adjacency matrix:

```text
d theta_i / dt = omega_i + (1 / N) sum_j K_ij sin(theta_j - theta_i).
```

The project compares global coupling, Erdős-Rényi random graphs, and Barabási-Albert scale-free networks. Scale-free topology synchronizes at lower coupling because hub nodes relay phase information more efficiently.

### AAL Brain Connectome

Brain regions from the AAL-90 atlas are modeled as oscillators connected by an empirical structural matrix `M_group`. The stochastic whole-brain Kuramoto equation is

```text
d theta_i = omega_i dt + lambda sum_j M_ij sin(theta_j - theta_i) dt + sigma dW_i.
```

This model is used to examine macroscopic synchrony, edge synchronization, modular locking, and the topological role of central hubs.

### Alzheimer's Disease Continuum

Group-average structural connectomes are constructed for CN, EMCI, LMCI, and AD cohorts. The same Kuramoto dynamics are simulated across disease stages to test whether structural degradation delays the transition to global synchronization. The perturbation-rescue experiment replaces selected AD network edges with CN-like edges to compare targeted hub repair against random repair.

### Circadian Resynchronization

The circadian section uses a forced Kuramoto model of SCN oscillators:

```text
d theta_i / dt = omega_i + (K / N) sum_j sin(theta_j - theta_i)
                 + F sin(sigma t - theta_i + p(t)).
```

Assuming a Lorentzian frequency distribution, the Ott-Antonsen reduction yields a macroscopic complex order-parameter equation:

```text
dot z = 1/2 [(K z + F) - z^2 (K conj(z) + F)] - (Delta + i Omega) z.
```

The reduced phase portrait explains east-west jet-lag asymmetry through the geometry of stable fixed points, saddles, and separatrices.

## Reproducing Figures

Run the Python versions from the repository root:

```bash
python3 01_Kuramoto_demo.py
python3 02_Brain_Kuramoto.py
python3 03_Circadian_clock.py
```

Or run the MATLAB versions from the repository root:

```matlab
run("01_Kuramoto_demo.m")
run("02_Brain_Kuramoto.m")
run("03_Circadian_clock.m")
```

Generated figures are saved to `Report/Figs/`.

## Interactive Page

The GitHub Pages site in `docs/` includes:

- Paper-style sections for Introduction, Kuramoto review, AAL connectome, brain synchronization, AD analysis, and circadian resynchronization.
- All report figures.
- Browser-based Kuramoto demos for oscillator phases and transition/bifurcation exploration.
- Minimal MATLAB/Python code blocks corresponding to the root scripts.

