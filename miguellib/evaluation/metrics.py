from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd


@dataclass(frozen=True)
class EvaluationMetrics:
    constraint_compliance: float
    category_diversity: float
    avg_similarity: Optional[float]

    def to_dict(self) -> dict:
        return {
            "constraint_compliance": self.constraint_compliance,
            "category_diversity": self.category_diversity,
            "avg_similarity": self.avg_similarity,
        }


def constraint_compliance_rate(
    recs: pd.DataFrame,
    budget: Optional[float],
    allowed_categories: Optional[Iterable[str]],
) -> float:
    if recs.empty:
        return 0.0

    mask = pd.Series(True, index=recs.index)

    if budget is not None and "price" in recs.columns:
        prices = pd.to_numeric(recs["price"], errors="coerce")
        mask &= prices <= float(budget)

    if allowed_categories and "category" in recs.columns:
        allowed = set(allowed_categories)
        mask &= recs["category"].isin(allowed)

    return float(mask.mean())


def category_diversity_ratio(recs: pd.DataFrame) -> float:
    if recs.empty or "category" not in recs.columns:
        return 0.0
    return float(recs["category"].nunique() / len(recs))


def average_similarity(recs: pd.DataFrame) -> Optional[float]:
    if recs.empty or "similarity" not in recs.columns:
        return None
    sims = pd.to_numeric(recs["similarity"], errors="coerce")
    if sims.dropna().empty:
        return None
    return float(sims.mean())


def summarize_metrics(
    recs: pd.DataFrame,
    budget: Optional[float],
    allowed_categories: Optional[List[str]],
) -> EvaluationMetrics:
    return EvaluationMetrics(
        constraint_compliance=constraint_compliance_rate(
            recs, budget=budget, allowed_categories=allowed_categories
        ),
        category_diversity=category_diversity_ratio(recs),
        avg_similarity=average_similarity(recs),
    )
