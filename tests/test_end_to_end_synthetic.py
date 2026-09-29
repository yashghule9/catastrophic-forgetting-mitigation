from cfm.adapt.loop import run_adaptation
from cfm.config import ExperimentConfig
from cfm.eval.classifier import accuracy
from cfm.prototypes.init import compute_class_prototypes
from cfm.utils.logging import RoundLogger
from conftest import make_synthetic_domains

EXPECTED_LOG_FIELDS = {
    "round", "alpha_init", "alpha_used", "accepted", "num_classes_updated",
    "source_val_acc", "baseline_source_val_acc", "target_acc", "forgetting_bound_ok",
}


def _run(guard_strategy: str, epsilon: float = 0.013):
    domains = make_synthetic_domains(
        num_classes=8, dim=4, samples_per_class=50, class_spacing=3.0,
        shift_scale=3.0, noise_std=1.0, seed=7,
    )
    p0 = compute_class_prototypes(domains.z_source_train, domains.y_source_train, domains.num_classes)

    config = ExperimentConfig(
        epsilon=epsilon, alpha_max=0.4, num_rounds=8, guard_strategy=guard_strategy,
        confidence_threshold=0.02, min_points_per_class=2,
    )
    logger = RoundLogger()
    result = run_adaptation(
        p0=p0, z_source_val=domains.z_source_val, y_source_val=domains.y_source_val,
        z_target=domains.z_target, config=config, y_target=domains.y_target, logger=logger,
    )
    return domains, config, result, logger


def test_every_round_satisfies_the_forgetting_bound():
    _domains, config, result, logger = _run("global_backtrack")
    baseline_acc = result.baseline_source_val_acc
    for row in logger.rows:
        assert row["source_val_acc"] >= baseline_acc - config.epsilon - 1e-9
        assert row["forgetting_bound_ok"]


def test_target_accuracy_improves_over_the_non_adapted_baseline():
    domains, config, result, _logger = _run("global_backtrack")
    target_acc_baseline = accuracy(domains.z_target, domains.y_target, result.p0, config.distance)
    target_acc_final = accuracy(domains.z_target, domains.y_target, result.p_final, config.distance)
    assert target_acc_final > target_acc_baseline


def test_log_schema_matches_expected_fields():
    _, _, _, logger = _run("global_backtrack")
    assert len(logger.rows) > 0
    for row in logger.rows:
        assert set(row.keys()) == EXPECTED_LOG_FIELDS


def test_per_class_greedy_strategy_also_respects_the_bound():
    _domains, config, result, logger = _run("per_class_greedy")
    baseline_acc = result.baseline_source_val_acc
    for row in logger.rows:
        assert row["source_val_acc"] >= baseline_acc - config.epsilon - 1e-9


def test_tighter_epsilon_never_degrades_source_val_below_bound():
    # A near-zero epsilon should still never be violated, even if it means
    # most/all rounds get rejected.
    domains, config, result, _logger = _run("global_backtrack", epsilon=1e-4)
    baseline_acc = result.baseline_source_val_acc
    final_acc = accuracy(domains.z_source_val, domains.y_source_val, result.p_final, config.distance)
    assert final_acc >= baseline_acc - config.epsilon - 1e-9
