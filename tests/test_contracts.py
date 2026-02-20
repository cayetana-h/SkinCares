import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "miguellib" / "datasets"
FEATURES_DIR = ROOT / "features"


def test_cosmetics_processed_schema():
    path = DATASETS_DIR / "cosmetics_processed.csv"
    assert path.exists(), "Missing cosmetics_processed.csv in miguellib/datasets"

    df = pd.read_csv(path, nrows=5)
    required = {"Label", "Brand", "Name", "Price", "Rank", "Ingredients"}
    assert required.issubset(set(df.columns))


def test_synonyms_json_is_dict():
    path = DATASETS_DIR / "synonyms.json"
    assert path.exists(), "Missing synonyms.json in miguellib/datasets"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict)


def test_ingredient_groups_json_schema():
    path = FEATURES_DIR / "ingredient_groups.json"
    assert path.exists(), "Missing ingredient_groups.json in features/"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict)
