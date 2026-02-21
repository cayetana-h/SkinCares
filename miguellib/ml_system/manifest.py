from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class Manifest:
    created_at: str
    data_hashes: Dict[str, str]
    code_hashes: Dict[str, str]
    artifact_hashes: Dict[str, str]
    schema_version: str
    extra: Dict[str, str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "created_at": self.created_at,
            "data_hashes": self.data_hashes,
            "code_hashes": self.code_hashes,
            "artifact_hashes": self.artifact_hashes,
            "schema_version": self.schema_version,
            "extra": self.extra,
        }


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_paths(paths: Iterable[Path], root: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for path in paths:
        rel = str(path.relative_to(root))
        hashes[rel] = _hash_file(path)
    return hashes


def build_manifest(
    root: Path,
    data_paths: Iterable[Path],
    code_paths: Iterable[Path],
    artifact_paths: Iterable[Path],
    schema_version: str,
    extra: Optional[Dict[str, str]] = None,
) -> Manifest:
    created_at = datetime.now(timezone.utc).isoformat()
    data_hashes = _hash_paths(data_paths, root)
    code_hashes = _hash_paths(code_paths, root)
    artifact_hashes = _hash_paths(artifact_paths, root)
    return Manifest(
        created_at=created_at,
        data_hashes=data_hashes,
        code_hashes=code_hashes,
        artifact_hashes=artifact_hashes,
        schema_version=schema_version,
        extra=extra or {},
    )


def write_manifest(manifest: Manifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)


def load_manifest(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
