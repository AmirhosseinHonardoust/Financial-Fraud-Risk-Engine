from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _policy_row(name: str, row: dict[str, Any], rationale: str) -> dict[str, Any]:
    """Return a compact policy row from a threshold-search result."""
    keys = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "specificity",
        "flagged_rate",
        "cost",
        "normalized_cost",
        "tp",
        "fp",
        "tn",
        "fn",
    ]
    result: dict[str, Any] = {"policy": name, "rationale": rationale}
    for key in keys:
        result[key] = row.get(key)
    return result


def _min_by(rows: Iterable[dict[str, Any]], key_fn) -> dict[str, Any] | None:
    rows = list(rows)
    return min(rows, key=key_fn) if rows else None


def _max_by(rows: Iterable[dict[str, Any]], key_fn) -> dict[str, Any] | None:
    rows = list(rows)
    return max(rows, key=key_fn) if rows else None


def build_threshold_policy_candidates(
    threshold_results: list[dict[str, Any]],
    *,
    target_recall: float = 0.95,
    target_precision: float = 0.70,
    max_review_rate: float = 0.10,
) -> list[dict[str, Any]]:
    """Build analyst-friendly threshold policy candidates.

    The raw threshold search is useful, but analysts usually need a small set of
    interpretable choices. These candidates summarize common operating modes:
    minimize business cost, maximize F1, prioritize recall, prioritize precision,
    and respect a review-capacity budget.
    """
    if not threshold_results:
        return []

    rows = [dict(row) for row in threshold_results]
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, float | None]] = set()

    def add(name: str, row: dict[str, Any] | None, rationale: str) -> None:
        if row is None:
            return
        threshold = _safe_float(row.get("threshold"))
        marker = (name, threshold)
        if marker in seen:
            return
        seen.add(marker)
        candidates.append(_policy_row(name, row, rationale))

    cost_optimized = _min_by(
        rows,
        lambda r: (
            _safe_float(r.get("cost")) if _safe_float(r.get("cost")) is not None else float("inf"),
            -(_safe_float(r.get("recall")) or 0.0),
            -(_safe_float(r.get("precision")) or 0.0),
        ),
    )
    add(
        "cost_optimized",
        cost_optimized,
        "Minimizes expected business cost under the configured "
        "false-positive and false-negative costs.",
    )

    balanced_f1 = _max_by(
        rows,
        lambda r: (
            _safe_float(r.get("f1")) or 0.0,
            -(_safe_float(r.get("cost")) or 0.0),
        ),
    )
    add(
        "balanced_f1",
        balanced_f1,
        "Maximizes F1 to balance precision and recall.",
    )

    high_recall_pool = [r for r in rows if (_safe_float(r.get("recall")) or 0.0) >= target_recall]
    high_recall = _min_by(
        high_recall_pool,
        lambda r: (
            _safe_float(r.get("cost")) if _safe_float(r.get("cost")) is not None else float("inf"),
            _safe_float(r.get("flagged_rate")) or 1.0,
            -(_safe_float(r.get("precision")) or 0.0),
        ),
    )
    add(
        "high_recall",
        high_recall,
        f"Maintains recall of at least {target_recall:.0%} while minimizing cost.",
    )

    high_precision_pool = [
        r for r in rows if (_safe_float(r.get("precision")) or 0.0) >= target_precision
    ]
    high_precision = _max_by(
        high_precision_pool,
        lambda r: (
            _safe_float(r.get("recall")) or 0.0,
            -(_safe_float(r.get("flagged_rate")) or 1.0),
        ),
    )
    add(
        "high_precision",
        high_precision,
        f"Maintains precision of at least {target_precision:.0%} "
        "while preserving as much recall as possible.",
    )

    review_capacity_pool = [
        r for r in rows if (_safe_float(r.get("flagged_rate")) or 1.0) <= max_review_rate
    ]
    review_capacity = _min_by(
        review_capacity_pool,
        lambda r: (
            _safe_float(r.get("cost")) if _safe_float(r.get("cost")) is not None else float("inf"),
            -(_safe_float(r.get("recall")) or 0.0),
        ),
    )
    add(
        "review_capacity",
        review_capacity,
        f"Keeps the flagged/review rate at or below {max_review_rate:.0%}.",
    )

    return candidates


def build_threshold_policy_summary(
    threshold_results: list[dict[str, Any]],
    best_threshold: dict[str, Any],
    *,
    cost_false_positive: float,
    cost_false_negative: float,
) -> dict[str, Any]:
    """Create a serializable threshold-policy summary."""
    candidates = build_threshold_policy_candidates(threshold_results)

    return {
        "purpose": "Threshold policy candidates for fraud-risk review.",
        "recommended_policy": "cost_optimized",
        "selected_threshold": best_threshold,
        "cost_assumptions": {
            "false_positive_cost": float(cost_false_positive),
            "false_negative_cost": float(cost_false_negative),
            "interpretation": (
                "A false negative is configured as more expensive because missing fraud is "
                "usually more costly than reviewing a legitimate transaction."
            ),
        },
        "policy_candidates": candidates,
        "notes": [
            "These policies are decision-support artifacts, not automatic approval rules.",
            "Thresholds should be reviewed with business, compliance, and operations "
            "stakeholders before deployment.",
            "The demo dataset is synthetic; threshold values should not be reused "
            "for real banking data without validation.",
        ],
    }


def save_threshold_policy_artifacts(
    threshold_results: list[dict[str, Any]],
    best_threshold: dict[str, Any],
    *,
    metrics_dir: str | Path,
    cost_false_positive: float,
    cost_false_negative: float,
) -> dict[str, Path]:
    """Save JSON, CSV, and Markdown threshold-policy artifacts."""
    metrics_path = Path(metrics_dir)
    metrics_path.mkdir(parents=True, exist_ok=True)

    summary = build_threshold_policy_summary(
        threshold_results,
        best_threshold,
        cost_false_positive=cost_false_positive,
        cost_false_negative=cost_false_negative,
    )

    json_path = metrics_path / "threshold_policy.json"
    csv_path = metrics_path / "threshold_policy.csv"
    md_path = metrics_path / "threshold_policy.md"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    candidates = summary["policy_candidates"]
    if candidates:
        fieldnames = list(candidates[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(candidates)
    else:
        csv_path.write_text("policy,rationale\n", encoding="utf-8")

    md_path.write_text(_render_markdown_policy(summary), encoding="utf-8")

    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": md_path,
    }


def _format_rate(value: Any) -> str:
    number = _safe_float(value)
    return "n/a" if number is None else f"{number:.3f}"


def _render_markdown_policy(summary: dict[str, Any]) -> str:
    candidates = summary.get("policy_candidates", [])
    cost = summary.get("cost_assumptions", {})

    lines = [
        "# Fraud Threshold Policy",
        "",
        "This file summarizes candidate decision thresholds for fraud-risk review.",
        "",
        "## Cost assumptions",
        "",
        f"- False-positive cost: `{cost.get('false_positive_cost')}`",
        f"- False-negative cost: `{cost.get('false_negative_cost')}`",
        f"- Interpretation: {cost.get('interpretation')}",
        "",
        "## Policy candidates",
        "",
        "| Policy | Threshold | Precision | Recall | FPR | Flagged rate | Cost | Rationale |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in candidates:
        lines.append(
            (
                "| {policy} | {threshold} | {precision} | {recall} | "
                "{fpr} | {flagged} | {cost} | {rationale} |"
            ).format(
                policy=row.get("policy", "n/a"),
                threshold=_format_rate(row.get("threshold")),
                precision=_format_rate(row.get("precision")),
                recall=_format_rate(row.get("recall")),
                fpr=_format_rate(row.get("false_positive_rate")),
                flagged=_format_rate(row.get("flagged_rate")),
                cost=_format_rate(row.get("cost")),
                rationale=str(row.get("rationale", "")).replace("|", "/"),
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )

    for note in summary.get("notes", []):
        lines.append(f"- {note}")

    lines.append("")
    return "\n".join(lines)
