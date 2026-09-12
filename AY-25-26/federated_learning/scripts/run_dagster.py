"""
Entry point for the Dagster UI / scheduler.

Usage (from federated_learning/):
    dagster dev -f scripts/run_dagster.py

Then open http://localhost:3000 and materialise assets individually or as a
group.  The two asset groups are:

  mnist  :  mnist_vfl_trained  ──►  mnist_vfl_eval
  cifer  :  cifer_vfl_trained  ──►  cifer_vfl_eval

Re-materialising an eval asset without re-materialising its upstream training
asset reuses the cached model state — enabling fast re-evaluation without
re-training.
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Ensure federated_learning/ is on sys.path so that ``src.*`` and
# ``configs.*`` imports resolve correctly when Dagster launches this file
# as a module from an arbitrary working directory.
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.dagster.definitions import defs  # noqa: E402  (import after path fix)

__all__ = ["defs"]
