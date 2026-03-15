"""
VFL LangGraph Graph Builder.

Constructs and compiles a LangGraph StateGraph that orchestrates VFL training
as a stateful round loop with conditional early-stopping support.

Graph topology
--------------
  START → train_round → evaluate → should_stop? ──► END
                ↑______________________↓ (continue)

Usage
-----
    from src.langgraph.vfl_graph import build_vfl_graph

    graph = build_vfl_graph(
        trainer=trainer,
        train_loaders=(loader_a, loader_b, loader_server),
        val_loaders=(val_a, val_b, val_server),
        schedulers=[sched_a, sched_b, sched_top],
    )
    final_state = graph.invoke(initial_state)
"""
from __future__ import annotations

from typing import Optional

from langgraph.graph import StateGraph, START, END

from src.langgraph.state import VFLState
from src.langgraph.nodes import make_train_round_node, make_evaluate_node
from src.vfl.training_loop import VFLTrainer


def _should_stop(state: VFLState) -> str:
    """
    Routing function for the conditional edge after evaluate.

    Returns 'stop'     → graph terminates (END)
    Returns 'continue' → loop back to train_round
    """
    if state["round"] >= state["num_rounds"]:
        return "stop"
    return "continue"


def build_vfl_graph(
    trainer: VFLTrainer,
    train_loaders: tuple,
    val_loaders: tuple,
    schedulers: Optional[list] = None,
    evaluate_node_fn=None,
):
    """
    Build and compile the VFL training StateGraph.

    Parameters
    ----------
    trainer : VFLTrainer
        Pre-constructed trainer instance (models, optimizers, criterion, device).
    train_loaders : tuple
        (loader_a, loader_b, loader_server) — training DataLoaders.
    val_loaders : tuple
        (val_a, val_b, val_server) — validation DataLoaders.
    schedulers : list, optional
        LR schedulers to step after each training round.
    evaluate_node_fn : callable, optional
        Custom evaluate node function ``(state) -> dict``.  Defaults to
        ``make_evaluate_node(trainer, *val_loaders)`` (MNIST/generic).
        Pass ``make_cifer_evaluate_node(...)`` for CiferAI fraud metrics + M4.

    Returns
    -------
    CompiledStateGraph
        Ready to invoke with ``graph.invoke(initial_state)``.
    """
    builder = StateGraph(VFLState)

    # ── Resolve evaluate node ────────────────────────────────────────────
    if evaluate_node_fn is None:
        evaluate_node_fn = make_evaluate_node(trainer, *val_loaders)

    # ── Register nodes ───────────────────────────────────────────────────
    builder.add_node(
        "train_round",
        make_train_round_node(trainer, *train_loaders, schedulers=schedulers),
    )
    builder.add_node(
        "evaluate",
        evaluate_node_fn,
    )

    # ── Wire edges ───────────────────────────────────────────────────────
    builder.add_edge(START, "train_round")
    builder.add_edge("train_round", "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        _should_stop,
        {"stop": END, "continue": "train_round"},
    )

    return builder.compile()
