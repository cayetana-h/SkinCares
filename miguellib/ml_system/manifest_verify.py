from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from miguellib.ml_system.artifacts import find_project_root
from miguellib.ml_system.manifest import load_manifest


@dataclass(frozen=True)
class VerificationIssue:
    message: str


class ManifestVerificationError(ValueError):
    def __init__(self, issues: Iterable[VerificationIssue]):
        messages = "\n".join(f"- {issue.message}" for issue in issues)
        super().__init__(f"Manifest verification failed:\n{messages}")
        self.issues = list(issues)


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_hashes(
    root: Path,
    expected: Dict[str, str],
    label: str,
    issues: List[VerificationIssue],
) -> None:
    for rel_path, expected_hash in expected.items():
        full_path = root / rel_path
        if not full_path.exists():
            issues.append(VerificationIssue(f"Missing {label} file: {rel_path}"))
            continue
        actual_hash = _hash_file(full_path)
        if actual_hash != expected_hash:
            issues.append(
                VerificationIssue(
                    f"{label} hash mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}"
                )
            )


def verify_manifest(root: Path, manifest_path: Path) -> None:
    manifest = load_manifest(manifest_path)

    issues: List[VerificationIssue] = []

    _verify_hashes(root, manifest.get("data_hashes", {}), "data", issues)
    _verify_hashes(root, manifest.get("code_hashes", {}), "code", issues)
    _verify_hashes(root, manifest.get("artifact_hashes", {}), "artifact", issues)

    if issues:
        raise ManifestVerificationError(issues)


def verify_manifest_default() -> None:
    root = find_project_root()
    manifest_path = root / "artifacts" / "manifest.json"
    verify_manifest(root, manifest_path)
