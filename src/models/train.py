"""Bootstrap entry point for the full leakage-controlled training implementation.

The complete v0.5.0 source tree is stored in this repository as a checksum-verified
snapshot because the initial GitHub App upload path could not expand all files in
one request. Running this module restores that source tree and then re-executes the
full training module.

The restored implementation performs these fixed stages:

1. deterministic customer-level train/validation/test splitting;
2. preprocessing fitted on the training split only;
3. logistic-regression and random-forest candidate training;
4. model selection on validation AUC;
5. policy-threshold selection on validation precision/recall trade-offs;
6. one final test evaluation after model and threshold are frozen;
7. bootstrap confidence intervals, calibration, and segment diagnostics;
8. MLflow-compatible tracking and champion/challenger artifact export.

Run `make all` for the complete reproducible workflow.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    restore = root / "scripts" / "restore_source.sh"
    subprocess.run(["bash", str(restore)], cwd=root, check=True)  # noqa: S603,S607

    restored = root / "src" / "models" / "train.py"
    current = restored.read_text(encoding="utf-8")
    if "MODEL_VERSION = \"0.5.0\"" not in current:
        raise RuntimeError("The verified source snapshot did not restore the full training module.")

    os.execv(sys.executable, [sys.executable, "-m", "src.models.train", *sys.argv[1:]])


if __name__ == "__main__":
    main()
