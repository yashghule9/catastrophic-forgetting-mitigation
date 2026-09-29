#!/usr/bin/env python
"""Render the adaptation-vs-forgetting tradeoff plot (source-val / target
accuracy per round, with the epsilon bound drawn in) and an alpha/acceptance
plot, from the per-round CSV log written by adapt.py."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import _bootstrap  # noqa: F401
from cfm.config import add_config_arg, config_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    args = parser.parse_args()
    config = config_from_args(args)

    log_path = Path(config.results_root) / "logs" / f"{config.run_tag}_seed{config.seed}.csv"
    if not log_path.exists():
        raise SystemExit(f"No round log at {log_path}. Run scripts/adapt.py first.")
    df = pd.read_csv(log_path)

    fig_dir = Path(config.results_root) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: adaptation-vs-forgetting tradeoff ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["round"], df["source_val_acc"], marker="o", label="Source-val accuracy")
    if "target_acc" in df.columns and df["target_acc"].notna().any():
        ax.plot(df["round"], df["target_acc"], marker="o", label="Target accuracy")
    baseline = df["baseline_source_val_acc"].iloc[0]
    bound = baseline - config.epsilon
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, label="Source-val baseline")
    ax.axhline(bound, color="red", linestyle=":", linewidth=1, label=f"Forgetting bound (baseline - {config.epsilon})")
    ax.set_xlabel("Adaptation round")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Adaptation vs. forgetting: {config.source_domain} -> {config.target_domain}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    tradeoff_path = fig_dir / f"tradeoff_{config.run_tag}.png"
    fig.savefig(tradeoff_path, dpi=150)
    plt.close(fig)

    # --- Plot 2: alpha used / acceptance per round ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = df["accepted"].map({True: "tab:green", False: "tab:red"})
    ax.bar(df["round"], df["alpha_used"], color=colors)
    ax.set_xlabel("Adaptation round")
    ax.set_ylabel("Alpha used (0 = round rejected)")
    ax.set_title("Adaptation rate per round (green = accepted, red = rejected)")
    fig.tight_layout()
    alpha_path = fig_dir / f"alpha_{config.run_tag}.png"
    fig.savefig(alpha_path, dpi=150)
    plt.close(fig)

    print(f"Saved plots:\n  {tradeoff_path}\n  {alpha_path}")


if __name__ == "__main__":
    main()
