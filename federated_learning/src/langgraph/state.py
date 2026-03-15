"""
VFL LangGraph State.

VFLState is the single mutable object that flows through every node in the
LangGraph StateGraph.  All fields must be JSON-serialisable so that LangGraph
can checkpoint the state.

Metric history uses operator.add as the reducer so that each node can simply
return a list of new entries and LangGraph will append them — the node never
needs to read the full history list to avoid overwriting it.
"""
from __future__ import annotations

import operator
from typing import TypedDict, Annotated


class VFLState(TypedDict):
    # ── Model checkpoint (serialisable dict of state_dicts) ─────────────
    model_state: dict          # VFLTrainer.get_model_state()

    # ── Training progress ────────────────────────────────────────────────
    round: int                 # current round (incremented inside train_round_node)
    num_rounds: int            # total rounds to run (set once at graph invocation)

    # ── Metrics log — operator.add reducer appends new entries per node ──
    metrics_history: Annotated[list[dict], operator.add]

    # ── Best-round tracking ───────────────────────────────────────────────
    # MNIST:   best_val_acc tracks generalisation; best_model_state unused ({})
    # CiferAI: best_recall tracks fraud detection (M4); best_model_state stores
    #          the deepcopy checkpoint at that round for final re-evaluation
    best_val_acc: float
    best_recall: float
    best_round: int
    best_model_state: dict     # M4 checkpoint weights (empty dict when unused)

    # ── Privacy budget (None when DP is disabled) ────────────────────────
    dp_epsilon: float | None