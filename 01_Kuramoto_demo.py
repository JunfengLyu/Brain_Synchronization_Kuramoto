from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "Report" / "Figs"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_review_figures() -> None:
    review_dir = ROOT / "Code" / "02_review_on_kuramoto_model" / "python"
    targets = [
        ("order_parameter", "02_order_parameter_schematic.png"),
        ("phase_diagram", "03_phase_diagram.png"),
        ("mean_field", "04_mean_field_bifurcation.png"),
    ]
    for module_name, filename in targets:
        module = load_module(module_name, review_dir / f"{module_name}.py")
        module.OUT_FILE = FIG_DIR / filename
        module.main()


def run_topology_figure() -> None:
    topology = load_module(
        "topology_connectome",
        ROOT / "Code" / "03_connection_architecture" / "python" / "topology_connectome.py",
    )
    topology.topology_transition()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    run_review_figures()
    run_topology_figure()
    print("Generated Kuramoto figures 02-05 in Report/Figs.")


if __name__ == "__main__":
    main()
