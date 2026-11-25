from __future__ import annotations

from hashlib import sha256

import pytest

from parciver.blueprint_artifact import file_sha256, require_artifacts, require_sha256


def test_sha256_verification_accepts_exact_bytes_and_rejects_mismatch(tmp_path):
    artifact = tmp_path / "blueprint.bin"
    payload = b"parciver-saved-blueprint\x00\x01"
    artifact.write_bytes(payload)
    expected = sha256(payload).hexdigest().upper()

    assert file_sha256(artifact, chunk_bytes=3) == expected
    assert require_sha256(artifact, expected.lower()) == expected
    assert require_artifacts({artifact: expected}) == {str(artifact): expected}

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        require_sha256(artifact, "0" * 64)


@pytest.mark.parametrize("expected", ["", "abc", "G" * 64, "0" * 63])
def test_sha256_verification_rejects_invalid_expected_digest(tmp_path, expected):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"x")
    with pytest.raises(ValueError, match="64 hexadecimal"):
        require_sha256(artifact, expected)


def test_sha256_verification_rejects_empty_set():
    with pytest.raises(ValueError, match="at least one"):
        require_artifacts({})
