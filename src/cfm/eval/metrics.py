"""The three metrics named in the project spec, computed from already-measured
accuracy numbers so there's exactly one place that defines what each means."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptationMetrics:
    source_val_acc_baseline: float
    source_val_acc_adapted: float
    target_acc_baseline: float
    target_acc_adapted: float

    @property
    def adaptation_gain(self) -> float:
        """Percentage-point increase in target accuracy vs. the non-adapted baseline."""
        return self.target_acc_adapted - self.target_acc_baseline

    @property
    def forgetting_delta(self) -> float:
        """Acc_s^initial - Acc_s^adapted. Positive means forgetting occurred;
        non-positive means source performance held or improved."""
        return self.source_val_acc_baseline - self.source_val_acc_adapted

    @property
    def retention_rate(self) -> float:
        """Adapted source accuracy as a fraction of the baseline (1.0 = no forgetting)."""
        if self.source_val_acc_baseline == 0:
            return 1.0
        return self.source_val_acc_adapted / self.source_val_acc_baseline

    @property
    def worst_case_accuracy(self) -> float:
        """Minimum accuracy across the adapted source-val and target slices —
        a single number capturing whether the model is reliable everywhere."""
        return min(self.source_val_acc_adapted, self.target_acc_adapted)

    def satisfies_forgetting_bound(self, epsilon: float) -> bool:
        return self.forgetting_delta <= epsilon

    def to_dict(self) -> dict:
        return {
            "source_val_acc_baseline": self.source_val_acc_baseline,
            "source_val_acc_adapted": self.source_val_acc_adapted,
            "target_acc_baseline": self.target_acc_baseline,
            "target_acc_adapted": self.target_acc_adapted,
            "adaptation_gain": self.adaptation_gain,
            "forgetting_delta": self.forgetting_delta,
            "retention_rate": self.retention_rate,
            "worst_case_accuracy": self.worst_case_accuracy,
        }
