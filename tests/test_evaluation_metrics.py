import pandas as pd
import pytest

from miguellib.evaluation.metrics import (
    average_similarity,
    category_diversity_ratio,
    constraint_compliance_rate,
)


def test_constraint_compliance_rate():
    recs = pd.DataFrame(
        {
            "price": [10, 60, 30],
            "category": ["Moisturizer", "Moisturizer", "Cleanser"],
        }
    )
    rate = constraint_compliance_rate(recs, budget=50, allowed_categories=["Moisturizer"])
    assert rate == 1 / 3


def test_category_diversity_ratio():
    recs = pd.DataFrame({"category": ["A", "A", "B", "C"]})
    assert category_diversity_ratio(recs) == 3 / 4


def test_average_similarity():
    recs = pd.DataFrame({"similarity": [0.1, 0.2, 0.3]})
    assert average_similarity(recs) == pytest.approx(0.2)
