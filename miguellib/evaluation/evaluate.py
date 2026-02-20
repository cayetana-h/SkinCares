from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from miguellib.evaluation.metrics import summarize_metrics
from miguellib.ml_system.artifacts import find_project_root
from miguellib.models.recommender_ranker import load_artifacts, rank_products
from miguellib.models.user_profile import build_user_vector


@dataclass(frozen=True)
class EvaluationScenario:
    name: str
    liked_product_ids: List[str]
    explicit_prefs: Dict[str, object]
    constraints: Dict[str, object]
    top_n: int = 10


def _load_tokens(root: Path) -> pd.DataFrame:
    tokens_path = root / "miguellib" / "datasets" / "datasets" / "products_tokens.csv"
    return pd.read_csv(tokens_path, dtype={"product_id": str})


def run_scenario(scenario: EvaluationScenario) -> Dict[str, object]:
    vectors, product_index, metadata_df = load_artifacts()
    tokens_df = _load_tokens(find_project_root())

    user_vector = build_user_vector(
        scenario.liked_product_ids,
        scenario.explicit_prefs,
        vectors,
        product_index,
    )

    recs = rank_products(
        user_vector=user_vector,
        product_vectors=vectors,
        metadata_df=metadata_df,
        constraints=scenario.constraints,
        top_n=scenario.top_n,
        tokens_df=tokens_df,
        product_index=product_index,
    )

    budget = scenario.constraints.get("budget") if scenario.constraints else None
    categories = scenario.constraints.get("categories") if scenario.constraints else None

    metrics = summarize_metrics(recs, budget=budget, allowed_categories=categories)

    return {
        "scenario": scenario.name,
        "top_n": scenario.top_n,
        "metrics": metrics.to_dict(),
    }


def default_scenarios(root: Path) -> List[EvaluationScenario]:
    vectors, product_index, metadata_df = load_artifacts()

    sample_ids = metadata_df["product_id"].astype(str).head(3).tolist()
    liked = sample_ids[:2] if len(sample_ids) >= 2 else []

    return [
        EvaluationScenario(
            name="baseline_moisturizer_budget",
            liked_product_ids=liked,
            explicit_prefs={"skin_type": "dry", "budget": 50.0},
            constraints={
                "budget": 50.0,
                "categories": ["Moisturizer"],
                "banned_ingredients": [],
            },
            top_n=10,
        )
    ]


def run_all(scenarios: Optional[List[EvaluationScenario]] = None) -> Dict[str, object]:
    root = find_project_root()
    scenarios = scenarios or default_scenarios(root)

    results = [run_scenario(s) for s in scenarios]
    return {"scenarios": results}


def write_report(path: Path, report: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    root = find_project_root()
    report = run_all()
    output_path = root / "artifacts" / "evaluation_report.json"
    write_report(output_path, report)
    print(f"Wrote evaluation report: {output_path}")


if __name__ == "__main__":
    main()
