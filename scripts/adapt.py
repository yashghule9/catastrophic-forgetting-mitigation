#!/usr/bin/env python
"""Run the T-round gradient-free adaptation loop: pseudo-label the target
domain with the current prototypes, refine per-class target modes via mean
shift, and commit a forgetting-bound-guarded convex update each round."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from cfm.adapt.loop import run_adaptation
from cfm.config import add_config_arg, config_from_args
from cfm.eval.metrics import AdaptationMetrics
from cfm.pipeline import (
    load_prototypes,
    prototypes_path,
    save_prototypes,
    source_split_embeddings,
    target_embeddings,
)
from cfm.utils.logging import RoundLogger
from cfm.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    args = parser.parse_args()
    config = config_from_args(args)
    set_seed(config.seed)

    p0 = load_prototypes(prototypes_path(config, tag="P0"))
    _, _, z_val, y_val = source_split_embeddings(config)
    z_target, y_target = target_embeddings(config)

    logger = RoundLogger()
    result = run_adaptation(
        p0=p0, z_source_val=z_val, y_source_val=y_val, z_target=z_target,
        config=config, y_target=y_target, logger=logger,
    )

    final_path = prototypes_path(config, tag="Pfinal")
    save_prototypes(final_path, result.p_final)
    print(f"Saved P_final ({result.p_final.shape}) -> {final_path}")

    log_path = Path(config.results_root) / "logs" / f"{config.run_tag}_seed{config.seed}.csv"
    logger.to_csv(log_path)
    print(f"Per-round log -> {log_path}")

    source_val_acc_adapted = logger.rows[-1]["source_val_acc"] if logger.rows else result.baseline_source_val_acc
    target_acc_adapted = logger.rows[-1]["target_acc"] if logger.rows else None

    baseline_path = Path(config.results_root) / f"baseline_{config.run_tag}.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {
        "source_val_acc_baseline": result.baseline_source_val_acc,
        "target_acc_baseline": target_acc_adapted,
    }

    metrics = AdaptationMetrics(
        source_val_acc_baseline=baseline["source_val_acc_baseline"],
        source_val_acc_adapted=source_val_acc_adapted,
        target_acc_baseline=baseline["target_acc_baseline"],
        target_acc_adapted=target_acc_adapted,
    )

    summary_path = Path(config.results_root) / f"summary_{config.run_tag}.json"
    summary_path.write_text(json.dumps({
        "source_domain": config.source_domain,
        "target_domain": config.target_domain,
        "epsilon": config.epsilon,
        "num_rounds": config.num_rounds,
        "guard_strategy": config.guard_strategy,
        **metrics.to_dict(),
    }, indent=2))

    print(f"Adaptation gain:   {metrics.adaptation_gain:+.4f}")
    print(f"Forgetting delta:  {metrics.forgetting_delta:+.4f} (bound: epsilon={config.epsilon})")
    print(f"Retention rate:    {metrics.retention_rate:.4f}")
    print(f"Worst-case acc:    {metrics.worst_case_accuracy:.4f}")
    print(f"-> {summary_path}")


if __name__ == "__main__":
    main()
