from pathlib import Path

from miguellib.ml_system.manifest import build_manifest


def test_build_manifest_hashes(tmp_path: Path):
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

    assert manifest.schema_version == "v1"
    assert "data.csv" in manifest.data_hashes
    assert "model.py" in manifest.code_hashes
    assert "artifact.bin" in manifest.artifact_hashes
    assert manifest.data_hashes["data.csv"]
    assert manifest.code_hashes["model.py"]
    assert manifest.artifact_hashes["artifact.bin"]
