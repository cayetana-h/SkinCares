from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd


@dataclass(frozen=True)
class ValidationIssue:
    message: str


class DataValidationError(ValueError):
    def __init__(self, issues: Iterable[ValidationIssue]):
        messages = "\n".join(f"- {issue.message}" for issue in issues)
        super().__init__(f"Data validation failed:\n{messages}")
        self.issues = list(issues)


REQUIRED_PRODUCTS_COLUMNS = {"product_id", "category", "price"}
REQUIRED_TOKENS_COLUMNS = {"product_id", "ingredient_tokens"}


def _assert_columns(df: pd.DataFrame, required: set[str], label: str, issues: List[ValidationIssue]) -> None:
    missing = required - set(df.columns)
    if missing:
        issues.append(ValidationIssue(f"{label} missing columns: {sorted(missing)}"))


def _assert_non_empty(df: pd.DataFrame, label: str, issues: List[ValidationIssue]) -> None:
    if df.empty:
        issues.append(ValidationIssue(f"{label} has no rows"))


def _assert_non_empty_strings(series: pd.Series, label: str, issues: List[ValidationIssue]) -> None:
    if series.isna().all():
        issues.append(ValidationIssue(f"{label} is entirely empty"))
        return
    if series.fillna("").str.strip().eq("").mean() > 0.5:
        issues.append(ValidationIssue(f"{label} has >50% empty strings"))


def _assert_prices(series: pd.Series, issues: List[ValidationIssue]) -> None:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().all():
        issues.append(ValidationIssue("price column is non-numeric"))
    elif (numeric < 0).any():
        issues.append(ValidationIssue("price column contains negative values"))


def validate_artifact_inputs(root: Path) -> None:
    data_products = root / "miguellib" / "datasets" / "datasets" / "products_clean.csv"
    data_tokens = root / "miguellib" / "datasets" / "datasets" / "products_tokens.csv"
    groups_path = root / "features" / "ingredient_groups.json"

    issues: List[ValidationIssue] = []

    if not data_products.exists():
        issues.append(ValidationIssue(f"Missing file: {data_products}"))
    if not data_tokens.exists():
        issues.append(ValidationIssue(f"Missing file: {data_tokens}"))
    if not groups_path.exists():
        issues.append(ValidationIssue(f"Missing file: {groups_path}"))

    if issues:
        raise DataValidationError(issues)

    products = pd.read_csv(data_products, dtype={"product_id": str})
    tokens = pd.read_csv(data_tokens, dtype={"product_id": str})

    _assert_columns(products, REQUIRED_PRODUCTS_COLUMNS, "products_clean.csv", issues)
    _assert_columns(tokens, REQUIRED_TOKENS_COLUMNS, "products_tokens.csv", issues)
    _assert_non_empty(products, "products_clean.csv", issues)
    _assert_non_empty(tokens, "products_tokens.csv", issues)

    if "price" in products.columns:
        _assert_prices(products["price"], issues)
    if "category" in products.columns:
        _assert_non_empty_strings(products["category"], "category", issues)
    if "ingredient_tokens" in tokens.columns:
        _assert_non_empty_strings(tokens["ingredient_tokens"], "tokens.ingredient_tokens", issues)

    if issues:
        raise DataValidationError(issues)
