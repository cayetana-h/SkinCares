from pathlib import Path

import pandas as pd
import pytest

from miguellib.ml_system.data_validation import DataValidationError, validate_artifact_inputs


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_json(path: Path, content: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_validate_artifact_inputs_success(tmp_path: Path):
    root = tmp_path
    products = pd.DataFrame(
        {
            "product_id": ["1"],
            "category": ["Moisturizer"],
            "price": [10.0],
        }
    )
    tokens = pd.DataFrame({"product_id": ["1"], "ingredient_tokens": ["water, glycerin"]})

    _write_csv(root / "miguellib" / "datasets" / "datasets" / "products_clean.csv", products)
    _write_csv(root / "miguellib" / "datasets" / "datasets" / "products_tokens.csv", tokens)
    _write_json(root / "features" / "ingredient_groups.json")

    validate_artifact_inputs(root)


def test_validate_artifact_inputs_missing_columns(tmp_path: Path):
    root = tmp_path
    products = pd.DataFrame({"product_id": ["1"], "price": [10.0]})
    tokens = pd.DataFrame({"product_id": ["1"]})

    _write_csv(root / "miguellib" / "datasets" / "datasets" / "products_clean.csv", products)
    _write_csv(root / "miguellib" / "datasets" / "datasets" / "products_tokens.csv", tokens)
    _write_json(root / "features" / "ingredient_groups.json")

    with pytest.raises(DataValidationError) as exc:
        validate_artifact_inputs(root)

    assert "missing columns" in str(exc.value)


def test_validate_artifact_inputs_empty_tokens(tmp_path: Path):
    root = tmp_path
    products = pd.DataFrame(
        {
            "product_id": ["1"],
            "category": ["Moisturizer"],
            "price": [10.0],
        }
    )
    tokens = pd.DataFrame({"product_id": ["1"], "ingredient_tokens": [""]})

    _write_csv(root / "miguellib" / "datasets" / "datasets" / "products_clean.csv", products)
    _write_csv(root / "miguellib" / "datasets" / "datasets" / "products_tokens.csv", tokens)
    _write_json(root / "features" / "ingredient_groups.json")

    with pytest.raises(DataValidationError) as exc:
        validate_artifact_inputs(root)

    assert "ingredient_tokens" in str(exc.value)
