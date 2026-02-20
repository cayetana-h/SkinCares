from pathlib import Path

import pytest

from miguellib.ml_system.manifest import build_manifest, write_manifest
from miguellib.ml_system.manifest_verify import ManifestVerificationError, verify_manifest


def _write_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_verify_manifest_success(tmp_path: Path):
    root = tmp_path
    data_file = root / "data.csv"
    code_file = root / "model.py"
    artifact_file = root / "artifact.bin"

    data_file.write_text("a,b\n1,2\n", encoding="utf-8")
    code_file.write_text("print('x')\n", encoding="utf-8")
    artifact_file.write_bytes(b"artifact")

    manifest = build_manifest(
        root=root,
        data_paths=[data_file],
        code_paths=[code_file],
        artifact_paths=[artifact_file],
        schema_version="v1",
    )
    manifest_path = root / "manifest.json"
    write_manifest(manifest, manifest_path)

    verify_manifest(root, manifest_path)


def test_verify_manifest_detects_change(tmp_path: Path):
    root = tmp_path
    data_file = root / "data.csv"
    code_file = root / "model.py"
    artifact_file = root / "artifact.bin"

    data_file.write_text("a,b\n1,2\n", encoding="utf-8")
    code_file.write_text("print('x')\n", encoding="utf-8")
    artifact_file.write_bytes(b"artifact")

    manifest = build_manifest(
        root=root,
        data_paths=[data_file],
        code_paths=[code_file],
        artifact_paths=[artifact_file],
        schema_version="v1",
    )
    manifest_path = root / "manifest.json"
    write_manifest(manifest, manifest_path)

    _write_file(artifact_file, b"artifact-changed")

    with pytest.raises(ManifestVerificationError) as exc:
        verify_manifest(root, manifest_path)

    assert "hash mismatch" in str(exc.value)
