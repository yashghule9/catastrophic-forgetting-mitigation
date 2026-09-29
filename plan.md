# Gradient-Free UDA + Anti-Forgetting for ViTs (BeiT) via Mean Shift Prototype Updates

## Context

The repo `catastrophic-forgetting-mitigation` currently contains only a two-line README — this is a greenfield build. The goal, per the user's spec, is a **gradient-free** unsupervised domain adaptation pipeline for a frozen Vision Transformer (BeiT): instead of fine-tuning backbone weights on unlabeled target data (which risks catastrophic forgetting of the source task and is compute-heavy), we adapt only a small set of **class prototypes** in the frozen backbone's embedding space, using density-based Mean Shift clustering to find robust target-class modes, and a deterministic accept/reject/rollback guard to hard-enforce a bound on source-accuracy degradation.


This is being built for **learning** (the user wants to understand and implement each algorithmic piece — mean shift, prototype convex updates, forgetting-constrained validation — themselves) and for **impact** as a **portfolio-quality open-source repo**: clean package structure, reproducible scripts, tests, and a README with real results on a standard UDA benchmark.

Decisions already made with the user (do not re-litigate):
- **Benchmark:** Office-Home (4 domains: Art, Clipart, Product, Real World; 65 classes). Primary pair: **Real World → Clipart** (largest visual shift, Real World has the most images for a stable source). Made configurable for other pairs (e.g. Art → Product) as a secondary/ablation experiment.
- **Compute:** single consumer GPU (8–24GB VRAM). Backbone forward passes are the only GPU-heavy step and are cached to disk, so everything else (mean shift, prototype updates, the adaptation loop, evaluation) runs cheaply on CPU from cached embeddings.
- **Backbone:** `microsoft/beit-base-patch16-224` via HuggingFace `transformers`, frozen (`requires_grad_(False)` everywhere, all inference under `torch.no_grad()`), CLS-token pooling as the embedding.
- **Impact deliverable:** portfolio-quality open-source repo (clean structure, README results table, reproducible CLI, unit tests). A results plot (adaptation-vs-forgetting tradeoff) is included as a cheap, high-value bonus even though a separate writeup/demo wasn't requested.

## Repo Structure (src-layout, package name `cfm`)

```
catastrophic-forgetting-mitigation/
├── pyproject.toml, LICENSE (MIT), Makefile, .gitignore
├── configs/default.yaml, configs/realworld_to_clipart.yaml, configs/art_to_product.yaml
├── data/
│   ├── README.md                # manual Office-Home download + expected layout instructions
│   ├── raw/OfficeHome/{Art,Clipart,Product,Real World}/<Class>/*.jpg   (user-provided, gitignored)
│   └── cache/{embeddings,splits,prototypes}/                          (generated, gitignored)
├── scripts/
│   ├── validate_data.py, extract_features.py, init_prototypes.py
│   ├── adapt.py, evaluate.py, make_plots.py, run_pipeline.py (cache-aware end-to-end driver)
├── src/cfm/
│   ├── config.py                 # ExperimentConfig dataclass, YAML+argparse loader
│   ├── data/office_home.py       # dataset scan + fixed 65-class list + seeded stratified splits
│   ├── data/validate.py
│   ├── features/backbone.py      # FrozenBeit wrapper (frozen, no_grad, CLS pooling)
│   ├── features/cache.py         # save/load_embeddings (.pt: embeddings, labels, paths, domain)
│   ├── prototypes/init.py        # compute_class_prototypes (per-class mean, L2-normalized)
│   ├── prototypes/update.py      # convex_update(p, mode, alpha)
│   ├── clustering/mean_shift.py  # from-scratch mean shift + bandwidth estimation
│   ├── adapt/pseudo_label.py     # nearest-prototype pseudo-labeling + margin-confidence filter
│   ├── adapt/forgetting_guard.py # accept/reject/rollback + alpha backtracking line search
│   ├── adapt/loop.py             # orchestrates T adaptation rounds
│   ├── eval/classifier.py        # nearest-prototype predict/accuracy (cosine default, euclidean option)
│   ├── eval/metrics.py           # adaptation gain, forgetting delta, retention rate, worst-case acc
│   └── utils/{seed,logging,distances}.py
├── tests/                        # all synthetic — no GPU, no network, no Office-Home needed
│   ├── conftest.py                       # make_synthetic_domains() fixture
│   ├── test_mean_shift.py                # vs sklearn.cluster.MeanShift on blobs (dev-only dep)
│   ├── test_prototype_update.py
│   ├── test_forgetting_guard.py
│   ├── test_pseudo_label.py
│   ├── test_classifier.py
│   └── test_end_to_end_synthetic.py
└── notebooks/exploration.ipynb   # optional
```

## Core Algorithms

**1. Feature extraction (`features/backbone.py`)** — BeiT loaded once, frozen, `torch.no_grad()`; `extract_features.py` batches every domain through it and caches `{embeddings, labels, paths, domain, model_name}` to `data/cache/embeddings/`. Source domain gets one seeded 80/20 train/val split persisted to disk — this is what makes `Acc(D_s^val; ·)` reproducible across every later run. This is the only stage touching the GPU.

**2. Prototype init (`prototypes/init.py`)** — `P0[c] = normalize(mean of source-train embeddings with label c))`. Immediately followed by computing baseline accuracies (source-val, target — target labels used only for reporting, never for adaptation) — first checkpoint deliverable.

**3. Mean shift, from scratch (`clustering/mean_shift.py`)** — Gaussian-kernel mode-seeking ascent (flat kernel also supported) seeded from every point in a per-class target subset; bandwidth via a k-NN-quantile heuristic (mirrors sklearn's `estimate_bandwidth`); converged trajectories merged into modes; `mean_shift_mode_for_class` returns the **largest-basin mode** (not a plain mean) — this is what makes mean shift worth using here: it's robust to the minority of points that a noisy nearest-prototype pseudo-labeling step mis-assigns. Correctness verified against `sklearn.cluster.MeanShift` on synthetic blobs in tests (sklearn is a dev/test-only dependency, never used at runtime — the algorithm itself is hand-implemented per the spec).

**4. Pseudo-labeling (`adapt/pseudo_label.py`)** — nearest-prototype assignment for unlabeled target embeddings; confidence = **margin** between best and second-best prototype similarity (more robust than an absolute threshold under domain shift); only confidently-assigned points (`confidence ≥ τ`, `min_points_per_class` floor) feed each class's mean-shift subset.

**5. Iterative adaptation loop with hard-enforced forgetting bound (`adapt/loop.py`, `adapt/forgetting_guard.py`)** — deterministic, per round:
```
for r in 1..T:
    pseudo-label target set using current P
    for each class c with enough confident points:
        mode_c = mean_shift_mode_for_class(confident target subset)
        P_candidate[c] = normalize((1-alpha) * P_current[c] + alpha * mode_c)
    if Acc(source_val; P_candidate) >= baseline_acc - epsilon:
        accept: P_current = P_candidate
    else:
        shrink alpha (backtracking line search) and retry;
        if alpha hits min_alpha and still violates the bound: reject round, P_current unchanged
    log round (accepted?, alpha used, source_val_acc, target_acc — target_acc is monitoring only)
```
This is a global backtracking line search: cheap because `Acc()` is just cached-embedding matrix ops (no backbone forward pass), so retries cost nothing. A per-class greedy variant (finer-grained retention, `O(C)` accuracy checks/round) is documented as a Phase 4e stretch goal, not required for v1.

**6. Evaluation (`eval/classifier.py`, `eval/metrics.py`)** — cosine-distance nearest-prototype classifier (default; justified since embeddings/prototypes are L2-normalized, matching the pseudo-labeling and update math; euclidean kept as a configurable ablation). Metrics: adaptation gain, forgetting delta, retention rate, worst-case accuracy — exactly the three metrics in the spec — logged per round to CSV/JSON for plotting.

## Config

Single flat `ExperimentConfig` dataclass (YAML + argparse overrides — deliberately simpler than Hydra for a learning-focused repo), key fields: `source_domain`, `target_domain`, `model_name`, `epsilon=0.013`, `alpha_max=0.3`, `alpha_shrink_factor=0.5`, `min_alpha`, `confidence_threshold`, `bandwidth_quantile`, `mean_shift_kernel`, `num_rounds=10`, `guard_strategy` (`global_backtrack` default, `per_class_greedy` stretch), `seed`.

## Testing Strategy

All tests run on synthetic Gaussian-blob "domains" (no GPU, no network, no Office-Home download needed) — `test_mean_shift` (vs sklearn), `test_prototype_update` (convex-combination edge cases: alpha=0/1/0.5), `test_forgetting_guard` (engineered bad update must be shrunk/rejected and must never corrupt `P_current` on full rejection), `test_pseudo_label` (margin computation on hand-built matrices), `test_classifier` (cosine vs euclidean on toy data with known answers), `test_end_to_end_synthetic` (full loop; asserts every logged round satisfies the epsilon bound and that target accuracy improves under an easy synthetic shift). Optional GitHub Actions CI workflow runs this full suite on every push (no data/GPU dependency, so it's a legitimate green badge).

## Phased Build Order

0. Scaffolding (`pyproject.toml`, src-layout, pytest config, `.gitignore`, README skeleton).
1. Data + features: `office_home.py`, `validate_data.py`, `data/README.md`, `FrozenBeit`, `extract_features.py`.
2. Prototypes + eval harness → **baseline numbers** (first real checkpoint, goes into README results table row 1).
3. Mean shift module, fully unit-tested against sklearn, built independent of the rest of the pipeline.
4. Iterative adaptation, built in sub-steps for learning: (a) convex update + forgetting guard tested standalone on hand-fed embeddings, (b) add pseudo-labeling, (c) swap in mean-shift mode instead of raw mean, (d) wire full multi-round loop + end-to-end test, (e) optional per-class-greedy guard ablation.
5. Real experiments on Office-Home (Real World → Clipart, +1–2 more pairs), `make_plots.py`, populate README results table/plots/limitations/design-decisions sections.

## Verification

- Unit tests (`pytest`) pass fully offline on synthetic data at every phase from Phase 3 onward — this is the primary correctness gate and should be run continuously during development, not just at the end.
- Phase 1 sanity check: inspect extracted-embedding shapes and confirm same-class cosine similarity > different-class similarity on a small subset before trusting anything downstream.
- Phase 2 checkpoint: baseline source-val and target accuracy numbers should be sane for a nearest-prototype classifier on frozen BeiT features (non-adapted target accuracy should be visibly lower than source-val accuracy, demonstrating the domain-shift problem the rest of the project addresses).
- Phase 5 end-to-end run: execute `scripts/run_pipeline.py` for Real World → Clipart and confirm (a) every round in the CSV log satisfies `source_val_acc >= baseline - epsilon`, (b) final target accuracy shows a positive adaptation gain over the non-adapted baseline, (c) `make_plots.py` produces the adaptation-vs-forgetting tradeoff figure used in the README.
