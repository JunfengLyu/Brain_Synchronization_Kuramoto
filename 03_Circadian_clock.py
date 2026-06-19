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
    circadian = load_module(
        "circadian_recovery",
        ROOT / "Code" / "06_circadian_resynchronization" / "python" / "circadian_recovery.py",
    )
    circadian.phase_space()
    circadian.recovery_curves()
    circadian.parameter_dependence()
    print("Generated circadian-clock figures 15-17 in Report/Figs.")


if __name__ == "__main__":
    main()
