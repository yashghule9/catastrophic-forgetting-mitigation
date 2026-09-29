"""Office-Home layout validation. Warns (does not fail) on count mismatches
since different mirrors of the dataset vary slightly; fails only on structural
problems (a missing domain, or domains disagreeing on their class sets) that
would silently break class-index alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from cfm.data.office_home import (
    DOMAINS,
    EXPECTED_DOMAIN_COUNTS,
    EXPECTED_NUM_CLASSES,
    domain_dir,
    list_domain_samples,
    resolve_class_list,
)

COUNT_TOLERANCE = 0.15  # warn if a domain's image count is off by more than 15%


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    class_list: list[str] | None = None
    domain_counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = []
        if self.class_list is not None:
            lines.append(f"Resolved {len(self.class_list)} shared classes across domains.")
        for domain, count in self.domain_counts.items():
            lines.append(f"  {domain}: {count} images")
        for w in self.warnings:
            lines.append(f"[warn] {w}")
        for e in self.errors:
            lines.append(f"[error] {e}")
        lines.append("VALID" if self.ok else "INVALID")
        return "\n".join(lines)


def validate_office_home(data_root: str | Path, domains: list[str] | None = None) -> ValidationReport:
    domains = domains or DOMAINS
    data_root = Path(data_root)
    errors: list[str] = []
    warnings: list[str] = []

    for domain in domains:
        d = domain_dir(data_root, domain)
        if not d.is_dir():
            errors.append(f"Missing domain folder: {d}")

    if errors:
        return ValidationReport(ok=False, errors=errors, warnings=warnings)

    try:
        class_list = resolve_class_list(data_root, domains)
    except ValueError as e:
        return ValidationReport(ok=False, errors=[str(e)], warnings=warnings)

    if len(class_list) != EXPECTED_NUM_CLASSES:
        warnings.append(
            f"Expected {EXPECTED_NUM_CLASSES} shared classes, found {len(class_list)}."
        )

    domain_counts: dict[str, int] = {}
    for domain in domains:
        samples = list_domain_samples(data_root, domain, class_list)
        domain_counts[domain] = len(samples)
        expected = EXPECTED_DOMAIN_COUNTS.get(domain)
        if expected and abs(len(samples) - expected) / expected > COUNT_TOLERANCE:
            warnings.append(
                f"{domain}: found {len(samples)} images, expected ~{expected} "
                f"(published count) — mirror may be incomplete."
            )
        if len(samples) == 0:
            errors.append(f"{domain}: no images found under resolved classes.")

    return ValidationReport(
        ok=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        class_list=class_list,
        domain_counts=domain_counts,
    )
