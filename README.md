# catastrophic-forgetting-mitigation

![tests](https://github.com/yashghule9/catastrophic-forgetting-mitigation/actions/workflows/tests.yml/badge.svg)

Gradient-free unsupervised domain adaptation and catastrophic-forgetting mitigation for a
frozen Vision Transformer (BeiT), via density-based **mean shift clustering** and
**weighted convex prototype updates**, with a hard-enforced bound on source-accuracy
retention.

## Motivation

Fine-tuning a Vision Transformer's weights on unlabeled target-domain data is the usual
way to close a domain gap — and the usual way to catastrophically forget the source task
while doing it. This project takes a different approach: **the backbone never moves**.
Only a small set of per-class **prototypes** — points in the frozen backbone's embedding
space — are adapted, using unsupervised density estimation (mean shift) to find robust
target-class modes, and a deterministic accept/reject/rollback guard that validates every
update against a held-out source split before committing it. If an update would push
source accuracy below `baseline - epsilon`, it is shrunk or rejected — never applied
blindly.

## Method

```
                 ┌──────────────────────┐
   source images │   Frozen BeiT (ViT)  │  target images (unlabeled)
   ───────────►  │  f_theta, no_grad,   │  ◄───────────
                 │  never backprop'd    │
                 └──────────┬───────────┘
                            │ CLS-token embeddings
              ┌─────────────┴─────────────┐
              ▼                           ▼
     per-class mean  ──► P^(0)     nearest-prototype
     (source-train)                pseudo-labeling (P_current)
                                            │ confidence filter (margin)
                                            ▼
                                 per-class target subsets Z_c^t
                                            │
                                   mean shift (from scratch):
                                   largest-basin density mode
                                            │
                                            ▼
     P_candidate[c] = (1-α)·P_current[c] + α·MeanShift(Z_c^t)
                                            │
                                            ▼
                     forgetting guard: Acc(source_val; P_candidate)
                        >= Acc(source_val; P^(0)) - epsilon ?
                    accept ──────────────┘└────────────── shrink α / reject
                                            │
                                    repeat for T rounds
                                            ▼
                                        P_final
```

**Why mean shift, not a plain mean?** Pseudo-labeling is noisy — a handful of target
points get assigned to the wrong class by imperfect prototypes. A plain per-class mean is
dragged by every such outlier; mean shift's largest-density-basin mode is not, because
outliers tend to form their own small, separate basin rather than perturbing the dominant
one.

**Why guard every update?** Because "the model should stay useful on the old domain" is a
requirement, not a hope. The guard makes it a testable invariant: every round is validated
against a fixed source-validation split before being committed, with a deterministic
backtracking line search on the adaptation rate `α` when the naive update would violate
the bound.

## Key equations

- Initial prototypes: `P^(0)_c = mean({ f_theta(x) : (x, y) in D_s^train, y = c })`
- Update rule: `P^(t+1)_c = (1 - α) · P^(t)_c + α · MeanShift(Z_c^t)`
- Forgetting constraint: `Acc(D_s^val; P^(t+1)) ≥ Acc(D_s^val; P^(0)) - ε`
- Metrics: `adaptation_gain = target_acc_adapted - target_acc_baseline`,
  `forgetting_delta = source_val_acc_baseline - source_val_acc_adapted`,
  `retention_rate = source_val_acc_adapted / source_val_acc_baseline`,
  `worst_case_accuracy = min(source_val_acc_adapted, target_acc_adapted)`

## Repo structure

```
src/cfm/
├── config.py               # ExperimentConfig — every knob in one place
├── data/office_home.py      # dataset scanning, class-index resolution, seeded splits
├── features/backbone.py     # FrozenBeit — frozen, no_grad, CLS-token embeddings
├── features/cache.py        # embedding cache (the only GPU-heavy stage runs once)
├── prototypes/init.py       # P^(0): per-class mean of source embeddings
├── prototypes/update.py     # the convex update rule itself
├── clustering/mean_shift.py # mean shift implemented from scratch
├── adapt/pseudo_label.py    # nearest-prototype pseudo-labeling + margin confidence
├── adapt/forgetting_guard.py# accept/reject/rollback + backtracking line search
├── adapt/loop.py            # orchestrates the T-round adaptation loop
└── eval/{classifier,metrics}.py

scripts/   # thin CLIs: validate_data, extract_features, init_prototypes, adapt,
           # evaluate, make_plots, run_pipeline (cache-aware end-to-end driver)
configs/   # YAML experiment configs (domain pairs, hyperparameters)
tests/     # fully offline — synthetic Gaussian-blob domains, no GPU/network/data
```

## Quickstart

```bash
pip install -e ".[dev]"
pytest                                  # offline correctness tests (no data/GPU needed)

# Real experiment on Office-Home (see data/README.md for manual download steps):
python scripts/run_pipeline.py --config configs/realworld_to_clipart.yaml
```

`run_pipeline.py` is cache-aware: it validates the data layout, extracts and caches BeiT
embeddings for both domains (skipped on reruns if already cached), computes `P^(0)` and
baseline accuracies, runs the guarded adaptation loop, evaluates `P_final`, and renders the
result plots — end to end, from one command.

## Configuration

All hyperparameters live in `src/cfm/config.py::ExperimentConfig`, loaded from a YAML file
(`configs/*.yaml`) with optional CLI overrides (`--epsilon 0.02`, etc). Key fields:

| Field | Default | Meaning |
|---|---|---|
| `epsilon` | `0.013` | max allowed source-val accuracy drop |
| `alpha_max` | `0.3` | initial adaptation rate per round |
| `alpha_shrink_factor` | `0.5` | backtracking shrink factor when the bound is violated |
| `num_rounds` | `10` | adaptation rounds |
| `confidence_threshold` | `0.05` | min pseudo-label margin to trust a target point |
| `bandwidth_quantile` | `0.3` | k-NN-quantile heuristic for mean-shift bandwidth |
| `guard_strategy` | `global_backtrack` | or `per_class_greedy` (finer-grained, more validation calls) |
| `distance` | `cosine` | nearest-prototype decision rule (`cosine` or `euclidean`) |

## Results

*Populate this table by running `scripts/run_pipeline.py` — it is intentionally left as a
template rather than filled with placeholder numbers.*

| Source → Target | Source-val acc (baseline) | Target acc (baseline) | Source-val acc (adapted) | Target acc (adapted) | Adaptation gain | Forgetting delta | Retention rate |
|---|---|---|---|---|---|---|---|
| Real World → Clipart | — | — | — | — | — | — | — |
| Art → Product | — | — | — | — | — | — | — |

The adaptation-vs-forgetting tradeoff plot and per-round α/acceptance plot are written to
`results/figures/` by `scripts/make_plots.py`.

## Design decisions & simplifications

- **CLS-token pooling** is the default embedding (standard for ViT-family checkpoints);
  mean-patch pooling is available as a configurable alternative.
- **Cosine distance** is the default decision rule, consistent with L2-normalized
  prototypes/embeddings used everywhere else in the pipeline; euclidean is kept as an
  ablation.
- **Global backtracking** is the default forgetting-guard strategy (one shared `α` per
  round, simple to reason about and test). `per_class_greedy` is available for
  finer-grained retention at the cost of more validation calls per round — still cheap,
  since validation never touches the backbone.
- **Fixed round count** (`num_rounds`) rather than an adaptive stopping rule, for
  determinism and easy plotting; an early-stop-on-small-prototype-delta extension is a
  natural next step.
- Mean shift is implemented from scratch per the project's requirements; `scikit-learn`
  is used only in tests, as an independent correctness oracle on synthetic blobs — never
  imported by the library at runtime.

## Limitations

- The method's ceiling is inherently lower than full fine-tuning, since it never adjusts
  the backbone — the tradeoff is exactly what buys the forgetting guarantee.
- Mean-shift bandwidth selection is a heuristic; results are somewhat sensitive to it
  under very large or very small per-class target sample counts.
- Pseudo-label noise grows with the severity of the domain shift, which is exactly when
  the confidence filter matters most and is hardest to tune.
- Each class prototype is a single point — the update rule implicitly assumes a
  (roughly) unimodal per-class distribution in the embedding space.

## Citations

- Venkateswara et al., *Deep Hashing Network for Unsupervised Domain Adaptation*, CVPR 2017
  (Office-Home dataset).
- Bao et al., *BEiT: BERT Pre-Training of Image Transformers*, ICLR 2022.
- Comaniciu & Meer, *Mean Shift: A Robust Approach Toward Feature Space Analysis*, PAMI 2002.

## License

MIT — see [LICENSE](LICENSE).
