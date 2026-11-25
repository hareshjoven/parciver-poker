"""Read-only integrity checks for durable blueprint/abstraction artifacts.

The Rust loader validates the binary strategy structure. These helpers validate
the externally pre-registered SHA-256 identity before the artifact is opened by
an experiment. Expected hashes are supplied by the caller so protocol constants
remain next to their protocol rather than hidden in generic infrastructure.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Mapping


def file_sha256(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return the uppercase SHA-256 digest of *path* using bounded memory."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    target = Path(path)
    digest = sha256()
    with target.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require_sha256(path: str | Path, expected: str) -> str:
    """Verify one artifact and return its normalized digest; mismatch is fatal."""
    expected = expected.strip().upper()
    if len(expected) != 64 or any(c not in "0123456789ABCDEF" for c in expected):
        raise ValueError("expected SHA-256 must be exactly 64 hexadecimal characters")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {Path(path)}: expected {expected}, got {actual}")
    return actual


def require_artifacts(expected_by_path: Mapping[str | Path, str]) -> dict[str, str]:
    """Verify a non-empty artifact set in caller iteration order."""
    if not expected_by_path:
        raise ValueError("at least one artifact is required")
    return {str(Path(path)): require_sha256(path, expected)
            for path, expected in expected_by_path.items()}
