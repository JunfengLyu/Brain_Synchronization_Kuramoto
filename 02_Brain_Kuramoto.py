from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    topology = load_module(
        "topology_connectome",
        ROOT / "Code" / "03_connection_architecture" / "python" / "topology_connectome.py",
    )
    brain = load_module(
        "brain_sync",
        ROOT / "Code" / "04_brain_network_synchronization" / "python" / "brain_sync.py",
    )
    ad = load_module(
        "ad_continuum",
        ROOT / "Code" / "05_alzheimers_disease" / "python" / "ad_continuum.py",
    )

    topology.aal_connectome()
    adjacency, _ = brain.load_group_consensus(0.40)
    brain.macroscopic_dynamics(adjacency)
    brain.lambda_states(adjacency)
    brain.module_hubs(adjacency)
    brain.focal_perturbation(adjacency)
    group_data = ad.load_group_matrices()
    ad.connectome_evolution(group_data)
    ad.lesion_regression(group_data)
    ad.transition_delay(group_data)
    ad.rescue_experiment(group_data)
    print("Generated brain connectome and AD figures 06-14 in Report/Figs.")


if __name__ == "__main__":
    main()
