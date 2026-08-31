"""Pure utilities for the external Autoresearch checkout adapter."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CUDA_VISIBLE_DEVICE_PATTERN = re.compile(
    r"(?:[0-9]+|GPU-[0-9A-Fa-f-]+|MIG-[0-9A-Fa-f-]+(?:/[0-9]+/[0-9]+)?)"
)
_OPTIONAL_SUMMARY_FIELDS = {
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "mfu_percent",
    "total_tokens_M",
    "num_steps",
    "num_params_M",
    "depth",
}
_SUMMARY_FIELD_PATTERN = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\s*:\s*(\S+)\s*")
_OFFICIAL_TRAINING_BUDGET_SECONDS = 300.0
_TRAINING_OVERSHOOT_TOLERANCE_SECONDS = 30.0


class AdapterError(ValueError):
    """Raised when an external Autoresearch checkout is not usable."""


def require_single_cuda_device(environ: Mapping[str, str]) -> str:
    """Require one operator-assigned CUDA ordinal, GPU UUID, or MIG UUID."""
    cuda_visible_devices = environ.get("CUDA_VISIBLE_DEVICES", "")
    if (
        not cuda_visible_devices
        or cuda_visible_devices != cuda_visible_devices.strip()
        or _CUDA_VISIBLE_DEVICE_PATTERN.fullmatch(cuda_visible_devices) is None
    ):
        raise AdapterError(
            "CUDA_VISIBLE_DEVICES must identify exactly one non-negative ordinal "
            "or GPU/MIG UUID set by the operator or scheduler."
        )
    return cuda_visible_devices


def _validated_manifest(manifest: object) -> dict[str, object]:
    if not isinstance(manifest, dict):
        raise AdapterError("Manifest must be a JSON object.")

    repository = manifest.get("repository")
    commit = manifest.get("commit")
    protected_files = manifest.get("protected_files")
    if not isinstance(repository, str) or not repository:
        raise AdapterError("Manifest repository must be a non-empty string.")
    if not isinstance(commit, str) or not _COMMIT_PATTERN.fullmatch(commit):
        raise AdapterError("Manifest commit must be a 40-character lowercase SHA.")
    if not isinstance(protected_files, dict) or not protected_files:
        raise AdapterError("Manifest protected_files must be a non-empty object.")

    for relative_path, digest in protected_files.items():
        if not isinstance(relative_path, str) or not _is_safe_relative_path(
            relative_path
        ):
            raise AdapterError(
                "Manifest protected file paths must stay under the root."
            )
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
            raise AdapterError(
                "Manifest protected file hashes must be SHA-256 digests."
            )
    return manifest


def _is_safe_relative_path(value: str) -> bool:
    path = Path(value)
    return (
        not path.is_absolute()
        and not path.drive
        and bool(path.parts)
        and ".." not in path.parts
    )


def load_manifest(path: str | os.PathLike[str]) -> dict[str, object]:
    """Load and validate the pinned upstream manifest."""
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdapterError(f"Could not load upstream manifest: {error}") from error
    return _validated_manifest(manifest)


def snapshot_protected_files(
    root: str | os.PathLike[str], manifest: Mapping[str, object]
) -> dict[str, str]:
    """Return SHA-256 digests for every manifest-protected upstream file."""
    validated_manifest = _validated_manifest(dict(manifest))
    protected_files = validated_manifest["protected_files"]
    assert isinstance(protected_files, dict)
    upstream_root = Path(root).resolve()

    snapshots: dict[str, str] = {}
    for relative_path in protected_files:
        assert isinstance(relative_path, str)
        protected_path = _resolve_protected_path(upstream_root, relative_path)
        try:
            snapshots[relative_path] = hashlib.sha256(
                protected_path.read_bytes()
            ).hexdigest()
        except OSError as error:
            raise AdapterError(
                f"Could not hash protected upstream file {relative_path}: {error}"
            ) from error
    return snapshots


def _resolve_protected_path(upstream_root: Path, relative_path: str) -> Path:
    try:
        protected_path = (upstream_root / relative_path).resolve(strict=True)
    except OSError as error:
        raise AdapterError(
            f"Could not resolve protected upstream file {relative_path}: {error}"
        ) from error
    try:
        protected_path.relative_to(upstream_root)
    except ValueError as error:
        raise AdapterError(
            f"Protected upstream file escapes the upstream root: {relative_path}"
        ) from error
    return protected_path


def validate_upstream(
    root: str | os.PathLike[str], manifest: Mapping[str, object]
) -> None:
    """Require an exact checked-out commit and unmodified protected files."""
    validated_manifest = _validated_manifest(dict(manifest))
    upstream_root = Path(root).resolve()
    if not upstream_root.is_dir():
        raise AdapterError(f"Upstream root does not exist: {upstream_root}")

    git_environment = os.environ.copy()
    for variable in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR"):
        git_environment.pop(variable, None)
    try:
        completed = subprocess.run(
            ["git", "-C", str(upstream_root), "rev-parse", "--verify", "HEAD^{commit}"],
            check=True,
            text=True,
            capture_output=True,
            env=git_environment,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise AdapterError(f"Could not read upstream commit: {error}") from error

    actual_commit = completed.stdout.strip()
    expected_commit = validated_manifest["commit"]
    assert isinstance(expected_commit, str)
    if actual_commit != expected_commit:
        raise AdapterError(
            f"upstream commit mismatch: expected {expected_commit}, got {actual_commit}"
        )

    expected_hashes = validated_manifest["protected_files"]
    assert isinstance(expected_hashes, dict)
    actual_hashes = snapshot_protected_files(upstream_root, validated_manifest)
    for relative_path, expected_hash in expected_hashes.items():
        assert isinstance(relative_path, str)
        assert isinstance(expected_hash, str)
        if actual_hashes[relative_path] != expected_hash:
            raise AdapterError(f"Protected upstream file changed: {relative_path}")


def build_training_command(
    root: str | os.PathLike[str], candidate: str | os.PathLike[str]
) -> list[str]:
    """Build the official uv invocation for one absolute candidate file."""
    del root
    return [
        "uv",
        "run",
        "--frozen",
        "--no-sync",
        "python",
        str(Path(candidate).resolve()),
    ]


def build_training_environment(
    root: str | os.PathLike[str], environ: Mapping[str, str]
) -> dict[str, str]:
    """Prepend the upstream checkout to PYTHONPATH without dropping variables."""
    environment = dict(environ)
    upstream_root = str(Path(root).resolve())
    previous_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{upstream_root}{os.pathsep}{previous_pythonpath}"
        if previous_pythonpath
        else upstream_root
    )
    environment["HF_HUB_OFFLINE"] = "1"
    return environment


def parse_official_summary(stdout: str) -> dict[str, float | int]:
    """Parse the final, delimiter-anchored official training summary."""
    lines = stdout.splitlines()
    separator_indexes = [index for index, line in enumerate(lines) if line == "---"]
    if not separator_indexes:
        raise AdapterError("Official final summary delimiter is missing.")

    parsed: dict[str, float | int] = {}
    for line in lines[separator_indexes[-1] + 1 :]:
        if not line.strip():
            continue
        match = _SUMMARY_FIELD_PATTERN.fullmatch(line)
        if not match:
            raise AdapterError("Official final summary contains a malformed line.")
        field, value = match.groups()
        if field != "val_bpb" and field not in _OPTIONAL_SUMMARY_FIELDS:
            raise AdapterError(f"Official final summary has an unknown field: {field}")
        if field in parsed:
            raise AdapterError(f"Official final summary repeats field: {field}")
        if field in {"num_steps", "depth"}:
            parsed[field] = _nonnegative_integer(value, field)
        else:
            parsed[field] = _finite_nonnegative_float(value, field)
    if "val_bpb" not in parsed:
        raise AdapterError("Official final summary must contain exactly one val_bpb.")
    training_seconds = parsed.get("training_seconds")
    if training_seconds is None:
        raise AdapterError("Official final summary must contain training_seconds.")
    if training_seconds < _OFFICIAL_TRAINING_BUDGET_SECONDS:
        raise AdapterError("Candidate did not complete the official 300-second budget.")
    if training_seconds > (
        _OFFICIAL_TRAINING_BUDGET_SECONDS + _TRAINING_OVERSHOOT_TOLERANCE_SECONDS
    ):
        raise AdapterError(
            "Official training_seconds exceeds the 300-second budget "
            "plus the final-step tolerance."
        )
    return parsed


def _finite_nonnegative_float(value: str, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise AdapterError(f"Official {field} must be a number.") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise AdapterError(f"Official {field} must be finite and non-negative.")
    return parsed


def _nonnegative_integer(value: str, field: str) -> int:
    if not re.fullmatch(r"[0-9]+", value):
        raise AdapterError(f"Official {field} must be a non-negative integer.")
    return int(value)


def score_from_bpb(val_bpb: float) -> float:
    """Convert raw validation BPB to ShinkaEvolve's maximize-oriented score."""
    return 1.0 / (1.0 + _finite_nonnegative_float(val_bpb, "val_bpb"))
