# Federated Learning as a Service (FLaaS)

A research implementation of **Vertical Federated Learning (VFL)** built across six incremental stages — from raw data engineering through to a production-grade Dagster pipeline with differential privacy, experiment tracking, and LangGraph orchestration.

---

## What Is This?

In Vertical Federated Learning, two parties each hold **different feature columns for the same set of records**, and train a shared model without exchanging raw data. Only 128-dimensional embedding vectors cross the party boundary.

```
Party A (identity/pixel features)              Party B (transaction/pixel features)
        │                                               │
   bottom_a(x_a) → emb_a ──────────────────────────── emb_b ← bottom_b(x_b)
                              ↓                ↓
                         top_model([emb_a, emb_b]) → loss
                              ↓
                     ← grad_a      grad_b →
                              (back-propagated to each party's bottom model)
```

Raw features never leave their owning party. Labels are held exclusively by the active party (server).

---

## Datasets

| Dataset | Task | Split |
|---------|------|-------|
| **MNIST** | 10-class digit classification | Party A: top 14 rows of 28×28 image · Party B: bottom 14 rows |
| **CiferAI Fraud Detection** | Binary fraud classification (0.12% fraud rate) | Party A: identity/temporal columns (`step`, `nameOrig`, `nameDest`) · Party B: transaction columns (`amount`, balances, `type`, `isFlaggedFraud`) |

---

## Architecture

```
federated_learning/
├── configs/
│   └── vfl_config.py              # Master config (plain dataclasses — no framework coupling)
├── src/
│   ├── datasets/
│   │   ├── cifer_dataset.py       # HuggingFace streaming, freq-encoding, minority oversampling
│   │   ├── mnist_dataset.py       # MNIST vertical spatial split (top/bottom halves)
│   │   └── dataloader_factory.py  # Aligned DataLoaders via shared torch.Generator seed
│   ├── models/
│   │   ├── bottom_models.py       # TabularBottomModel (MLP+BN) | CNNBottomModel → 128-dim
│   │   └── top_model.py           # VFLTopModel: concat(emb_a, emb_b) → num_classes
│   ├── vfl/
│   │   ├── gradient_bridge.py     # 3-step VFL protocol; DP clip+noise hook
│   │   ├── training_loop.py       # VFLTrainer: train_one_epoch, evaluate, checkpoint
│   │   └── dp_accountant.py       # DPBudgetAccountant (Opacus RDPAccountant wrapper)
│   ├── langgraph/
│   │   ├── state.py               # VFLState TypedDict
│   │   ├── nodes.py               # train_round_node, evaluate_node, should_stop_node
│   │   └── vfl_graph.py           # build_vfl_graph() — compiled StateGraph
│   └── dagster/
│       ├── assets.py              # @asset definitions (mnist + cifer groups)
│       └── definitions.py         # Definitions(assets=[...])
├── scripts/
│   ├── run_mnist_vfl.py           # MNIST entry point (pure PyTorch + MLflow)
│   ├── run_cifer_vfl.py           # CiferAI entry point (pure PyTorch + MLflow)
│   ├── run_mnist_langgraph.py     # MNIST via LangGraph StateGraph
│   ├── run_cifer_langgraph.py     # CiferAI via LangGraph StateGraph
│   └── run_dagster.py             # dagster dev entry point
└── tests/
    ├── test_datasets.py
    ├── test_models.py
    └── test_training_loop.py      # 40 tests including 4 DP-specific tests
```

---

## Stages

### Stage 1 & 2 — VFL Data Engineering + Core Simulation

Built the full VFL protocol and both dataset pipelines from scratch.

**Key design decisions:**
- `torch.Generator` shared seed → identical DataLoader shuffle order across parties without a shared sampler
- `detach().requires_grad_(True)` cut-layer → simulates network boundary; `local_emb.backward(received_grad)` propagates correctly into bottom model weights
- `drop_last=True` on all loaders → prevents single-sample batches that break `BatchNorm1d`
- Frequency encoding for `nameOrig`/`nameDest` (millions of unique values — one-hot would OOM)
- `build_aligned_pair()` as sole factory → encoders/scalers fit on train only, no data leakage

**MNIST baseline result:** 92.7% val accuracy at round 10 (25-round run with `CosineAnnealingLR`)

**CiferAI baseline challenges fixed:**
- Streaming load (`datasets` streaming API) — avoids downloading the full 1.8 GB dataset for a 100k-row sample
- Proper train/val split — validation on held-out data, not training data
- Minority oversampling (94 → 8,878 fraud rows, 10% target ratio) — model learns a fraud signal
- Class-weighted `CrossEntropyLoss` + F1/recall/precision metrics — accuracy alone is meaningless at 0.12% fraud rate

---

### Stage 3 — Opacus Differential Privacy

Added **(ε, δ)-DP** at the VFL cut-layer to defend against gradient inversion attacks.

**Why embedding-level noise, not Opacus `PrivacyEngine`?**
Opacus `PrivacyEngine` clips parameter gradients during `optimizer.step()`. VFL's manual gradient protocol (`local_emb.backward(received_grad)`) does not go through `optimizer.step()`, so `PrivacyEngine` cannot instrument it. The attack surface is the *embedding channel*, not parameter updates — embedding-level noise directly defends against the threat.

**Mechanism** (inside `gradient_bridge.detach_for_transmission()`):
```python
# Per-sample L2 norm clipping (bounds sensitivity to clip_norm C)
norms = emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
emb   = emb * (clip_norm / norms).clamp(max=1.0)

# Gaussian mechanism: noise_std = noise_multiplier * clip_norm
emb   = emb + torch.randn_like(emb) * (noise_multiplier * clip_norm)
```

Privacy budget is tracked via `opacus.accountants.RDPAccountant` in `DPBudgetAccountant`.

**Backward compatible:** calling `detach_for_transmission(emb)` with no DP args is identical to the pre-DP behaviour — all 40 tests pass without modification.

---

### Stage 4 — MLflow Experiment Tracking

All training runs are logged to MLflow with full hyperparameters, per-round metrics, and (when DP is enabled) the final ε value.

```bash
# Launch the MLflow UI
cd federated_learning
mlflow ui --port 5000
```

Run names follow the convention `{dataset}-{orchestrator}-dp{on|off}-r{rounds}-s{seed}`, e.g. `cifer-lg-dpon-r25-s42`.

---

### Stage 5 — LangGraph Orchestration

VFL training is expressed as a `StateGraph` where each node is a training round. State carries model weights, round number, metrics history, and the M4 best-recall checkpoint.

```
START → train_round → evaluate → should_stop? → END
              ↑_________________________| (continue)
```

**M4 checkpoint:** the best-recall model state is saved in-memory at every round. Because CiferAI recall can peak early and then collapse as Adam's variance estimate saturates, the final model returned is the best-recall checkpoint — not the final-round weights.

---

### Stage 6 — Dagster Pipeline

VFL training is exposed as a Dagster asset pipeline with full lineage tracking and UI materialisation.

```
Asset group: mnist          Asset group: cifer
─────────────────           ──────────────────
mnist_vfl_trained           cifer_vfl_trained
       │                           │
mnist_vfl_eval              cifer_vfl_eval
```

Re-materialising an eval asset reuses the cached upstream model state — enabling fast re-evaluation without re-training.

```bash
cd federated_learning
dagster dev -f scripts/run_dagster.py
# → http://localhost:3000
```

---

## Experimental Results Summary

### MNIST — Key Runs

| Run | Orchestrator | DP | Best Val Acc | Best Round | Notes |
|-----|--------------|----|-------------|------------|-------|
| `mnist-pt-dpoff-r25-s42` | Pure PyTorch | Off | 92.7% | 10 | Overfits after round 10 |
| `mnist-lg-dpoff-r25-s42` | LangGraph | Off | 92.7% | 10 | Deterministic match to pure PyTorch |
| `mnist-lg-dpon-r25-s42` | LangGraph | On | ~55% | — | DP destroys MNIST signal (SNR ≈ 0.09); embedding too small to absorb noise |

### CiferAI Fraud Detection — Key Runs

| Run | Orchestrator | DP | Best Recall | Best Round | Notes |
|-----|--------------|----|------------|------------|-------|
| `cifer-lg-dpoff-r25-s42` | LangGraph | Off | 39.1% | 11 | M4 checkpoint; model collapses to no-fraud by round 13 |
| `cifer-lg-dpon-r25-s42` | LangGraph | On (ε=2.23) | **91.3%** | 2 | DP noise acts as stochastic regulariser — prevents majority-class mode collapse |

**Key finding — non-monotonic DP privacy-utility tradeoff:**

| Dataset | DP Effect | Explanation |
|---------|-----------|-------------|
| MNIST | Strongly harmful | High signal-to-noise ratio; Gaussian noise overwhelms the 128-dim embedding signal |
| CiferAI | Beneficial | Prevents Adam variance accumulation from collapsing the model to predict no-fraud; noise acts as regularisation |

This confirms that DP's effect on model utility is **dataset- and architecture-dependent**, not universally harmful.

---

## Installation

```bash
git clone <repo-url>
cd federated_learning
```

### Auto-activation (recommended — one-time setup)

Run the setup script once after cloning. It installs a shell hook so that every new terminal opened inside this repository **automatically** creates (on first run) and activates the virtual environment.

**macOS / Linux (zsh or bash):**
```bash
bash .scripts/setup_autoactivate.sh
source ~/.zshrc        # or ~/.bashrc
```

From then on, every new terminal will print:
```
[FLaas] venv active → Python 3.x.x
```

**Windows (CMD):**
```bat
.scripts\setup_activate.bat
```

### Manual activation

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate.bat
```

### Dependency install (if not using auto-activation)

```bash
pip install -r requirements.txt
```

---

## Running the Pipelines

### Pure PyTorch (with MLflow logging)
```bash
python scripts/run_mnist_vfl.py
python scripts/run_cifer_vfl.py
```

### LangGraph StateGraph
```bash
python scripts/run_mnist_langgraph.py
python scripts/run_cifer_langgraph.py
```

### Dagster UI
```bash
dagster dev -f scripts/run_dagster.py
# Open http://localhost:3000 → materialise assets individually or as a group
```

### MLflow UI
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### Test Suite
```bash
pytest tests/ -v
# 40/40 tests expected
```

---

## Enabling Differential Privacy

DP is disabled by default. To enable it in any script:

```python
cfg = VFLConfig(dataset="mnist")
cfg.dp.enabled = True
cfg.dp.clip_norm = 1.0          # L2 sensitivity bound C
cfg.dp.noise_multiplier = 1.0   # σ: noise_std = σ × C
cfg.dp.delta = 1e-5             # δ for (ε, δ)-DP guarantee
```

At the end of training the privacy budget is printed:
```
Privacy budget — ε=2.2274 at δ=1e-05 (after 25 rounds)
```

In the Dagster UI, set `dp_enabled = true` in the asset config panel before materialising.

---

## What Can Be Improved

### Model Quality

| Area | Issue | Suggested Approach |
|------|-------|-------------------|
| **MNIST early stopping** | Val accuracy peaks at round ~10 and then overfits. Final-round accuracy (91.9%) is below the round-10 peak (92.7%). | Add patience-based early stopping (patience = 5 rounds). The LangGraph graph already supports conditional branching — add an `early_stop` edge. |
| **CiferAI precision** | Best recall (91.3%) comes at the cost of very low precision (~0.1%). The model flags thousands of false positives. | Calibrate the `fraud_threshold` against a held-out calibration set. Platt scaling or isotonic regression on softmax outputs would align model confidence to the real 0.12% fraud prior. |
| **CiferAI stability** | Recall peaks early (round 2–3) and then collapses as Adam's variance estimate saturates. M4 checkpoint is a workaround, not a fix. | Use a learning rate warm-up for the first 3 rounds, or switch to AdamW with weight decay to regularise momentum accumulation. |
| **MNIST under DP** | DP reduces MNIST accuracy from 92.7% to ~55% — the 128-dim embedding is too small to absorb Gaussian noise at `noise_multiplier=1.0`. | Increase `embedding_dim` to 256–512 under DP, or reduce `noise_multiplier` to 0.3–0.5 to trade a weaker ε guarantee for higher utility. |

### Privacy & Security

| Area | Issue | Suggested Approach |
|------|-------|-------------------|
| **Gradient channel** | DP protects the embedding channel, but `dL/d_embedding` gradients sent back to passive parties are unprotected. A curious server could attempt feature reconstruction from these. | Apply clip+noise to outgoing gradients in `gradient_bridge.apply_gradient()` with a separate accountant. |
| **ε calibration** | `noise_multiplier=1.0` gives ε ≈ 2.23 after 25 rounds. Whether this is meaningful depends on the threat model. | Sweep `noise_multiplier` ∈ {0.5, 1.0, 2.0} and plot the ε–accuracy Pareto frontier for both datasets. |
| **Membership inference** | No membership inference attack evaluation is included. | Implement a shadow model attack (Shokri et al. 2017) to empirically verify that DP reduces membership inference accuracy toward the random-guess baseline. |

### Architecture & Engineering

| Area | Issue | Suggested Approach |
|------|-------|-------------------|
| **Flower integration** | The `src/flower/` client/server stubs exist but are not used — all training runs in-process via `VFLTrainer`. | Wire the Flower `ClientApp` / `ServerApp` to `VFLTrainer` and add a `run_mnist_flower.py` script. This is the path toward real multi-process or multi-machine VFL. |
| **GPU support** | `device="cpu"` is hardcoded in all Dagster assets and LangGraph scripts. | Detect MPS/CUDA at runtime (`torch.backends.mps.is_available()`) and thread it through `VFLConfig.device`. Add a `device` field to the Dagster `Config` classes. |
| **Dagster IO manager** | The default pickle IO manager stores tensors in a temp SQLite database — not portable across machines or Python versions. | Configure a `FilesystemIOManager` to save model state dicts as `.pt` files for portability. |
| **CiferAI data freshness** | The dataset is streamed from HuggingFace on every run with no local caching. | Add a `cache_dir` option to `CiferConfig` that saves preprocessed numpy arrays on first run and reads from cache on subsequent runs. |
| **Test coverage** | The 40-test suite covers the core VFL protocol but has no integration tests for LangGraph graphs or Dagster assets. | Add `tests/test_langgraph.py` (2-round smoke test) and `tests/test_dagster_assets.py` using `dagster.build_asset_context()`. |

### Research Directions

| Direction | Description |
|-----------|-------------|
| **Asymmetric DP** | Apply different `clip_norm`/`noise_multiplier` per party based on feature sensitivity. Identity columns (Party A) may warrant stronger protection than transaction columns (Party B). |
| **Evidently drift monitoring** | `evidently>=0.4.0` is already listed in `requirements.txt` (commented out). Add a drift-detection Dagster asset that compares val embedding distributions across materialisation runs. |
| **LLM run summaries** | Add a final LangGraph node that calls the Claude API to generate a natural-language summary of each training run, linked to the MLflow experiment. |
| **Secure aggregation** | Replace plaintext embedding exchange with a homomorphic encryption layer (e.g. TenSEAL) so the server never sees individual party embeddings. |
| **Multi-party VFL** | The current architecture is fixed at two parties. Generalise `VFLTopModel` and `VFLTrainer` to accept N parties and test with a 3-party MNIST split. |
