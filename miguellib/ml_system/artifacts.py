# miguellib/ml_system/artifacts.py

import json
from pathlib import Path
import numpy as np


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "artifacts").exists():
            return p
    raise FileNotFoundError("Could not find project root (folder containing 'artifacts/').")


def load_artifacts():
    root = find_project_root()

    vectors = np.load(root / "artifacts" / "product_vectors.npy")

    with open(root / "artifacts" / "product_index.json", "r") as f:
        product_index = json.load(f)

    schema = None
    schema_path = root / "artifacts" / "feature_schema.json"
    if schema_path.exists():
        with open(schema_path, "r") as f:
            schema = json.load(f)

    index_to_id = {v: k for k, v in product_index.items()}

    return vectors, product_index, index_to_id, schema
