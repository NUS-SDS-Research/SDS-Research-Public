"""
VFL Dagster Definitions.

Wires all assets into a single Definitions object consumed by
``dagster dev -f scripts/run_dagster.py``.

Asset groups
------------
  mnist  :  mnist_vfl_trained  ──►  mnist_vfl_eval
  cifer  :  cifer_vfl_trained  ──►  cifer_vfl_eval

Each group can be materialised independently or together.
Re-materialising an eval asset without re-materialising its training
asset reuses the cached model state — enabling fast re-evaluation.
"""
from __future__ import annotations

from dagster import Definitions

from src.dagster.assets import (
    mnist_vfl_trained,
    mnist_vfl_eval,
    cifer_vfl_trained,
    cifer_vfl_eval,
)

defs = Definitions(
    assets=[
        mnist_vfl_trained,
        mnist_vfl_eval,
        cifer_vfl_trained,
        cifer_vfl_eval,
    ],
)
