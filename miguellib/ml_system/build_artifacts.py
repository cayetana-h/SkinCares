from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from miguellib.ml_system.artifacts import find_project_root
from miguellib.ml_system.data_validation import validate_artifact_inputs
from miguellib.ml_system.manifest import build_manifest, write_manifest
from miguellib.models import vectorizer


def _default_data_paths(root: Path) -> List[Path]:
    return [
        root / "miguellib" / "datasets" / "datasets" / "products_clean.csv",
        root / "miguellib" / "datasets" / "datasets" / "products_tokens.csv",
        root / "features" / "ingredient_groups.json",
    ]


def _default_code_paths(root: Path) -> List[Path]:
    return [
        root / "miguellib" / "models" / "vectorizer.py",
        root / "miguellib" / "models" / "user_profile.py",
        root / "miguellib" / "models" / "similarity.py",
    ]


def _default_artifact_paths(root: Path) -> List[Path]:
    return [
        root / "artifacts" / "product_vectors.npy",
        root / "artifacts" / "product_index.json",
        root / "artifacts" / "feature_schema.json",
        root / "artifacts" / "tfidf.joblib",
    ]


def build(schema_version: str = "v1") -> Path:
    root = find_project_root()

    data_paths = _default_data_paths(root)
    code_paths = _default_code_paths(root)
    artifact_paths = _default_artifact_paths(root)

    validate_artifact_inputs(root)
    vectorizer.run()

    for path in data_paths + code_paths + artifact_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required path missing: {path}")

    manifest = build_manifest(
        root=root,
        data_paths=data_paths,
        code_paths=code_paths,
        artifact_paths=artifact_paths,
        schema_version=schema_version,
        extra={"artifacts_dir": "artifacts"},
    )

    manifest_path = root / "artifacts" / "manifest.json"
    write_manifest(manifest, manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vector artifacts and manifest.")
    parser.add_argument(
        "--schema-version",
        default="v1",
        help="Schema version label recorded in manifest.",
    )
    args = parser.parse_args()

    manifest_path = build(schema_version=args.schema_version)
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
