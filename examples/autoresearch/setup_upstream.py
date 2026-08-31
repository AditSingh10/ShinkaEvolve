"""Explicit setup commands for the external pinned Autoresearch checkout."""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import adapter

MANIFEST_PATH = Path(__file__).with_name("upstream_manifest.json")
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "autoresearch"
_PINNED_VALIDATION_FILENAME = "shard_06542.parquet"
_DEPENDENCY_PROBE = (
    "import kernels, matplotlib, numpy, pandas, pyarrow, requests, rustbpe, "
    "tiktoken, torch"
)
_KERNEL_PROBE = (
    "import torch; from kernels import get_kernel; "
    "cap = torch.cuda.get_device_capability(); "
    "repo = 'varunneal/flash-attention-3' if cap == (9, 0) "
    "else 'kernels-community/flash-attn3'; get_kernel(repo)"
)


class SetupError(ValueError):
    """Raised when an explicit upstream setup operation is unsafe or invalid."""


class ReadinessError(SetupError):
    """Raised when an upstream checkout is valid but not ready to evaluate."""


class _CloneReservation:
    __slots__ = ("created", "device", "inode", "path")

    def __init__(self, *, path: Path, device: int, inode: int, created: bool) -> None:
        self.path = path
        self.device = device
        self.inode = inode
        self.created = created


def build_clone_command(root: str | Path, manifest: Mapping[str, object]) -> list[str]:
    """Build the only command that obtains the external upstream source."""
    repository = manifest["repository"]
    assert isinstance(repository, str)
    return ["git", "clone", repository, str(Path(root))]


def build_checkout_command(
    root: str | Path, manifest: Mapping[str, object]
) -> list[str]:
    """Build the pinned detached-checkout command for a cloned upstream root."""
    commit = manifest["commit"]
    assert isinstance(commit, str)
    return ["git", "-C", str(Path(root)), "checkout", "--detach", commit]


def build_upstream_command(operation: str) -> list[str]:
    """Build an upstream-owned uv command for one explicit setup operation."""
    commands = {
        "sync": ["uv", "sync", "--frozen"],
        "prepare": ["uv", "run", "--frozen", "--no-sync", "prepare.py"],
        "kernel": [
            "uv",
            "run",
            "--frozen",
            "--no-sync",
            "python",
            "-B",
            "-c",
            _KERNEL_PROBE,
        ],
        "baseline": ["uv", "run", "--frozen", "--no-sync", "train.py"],
    }
    try:
        return commands[operation]
    except KeyError as error:
        raise SetupError(f"Unsupported upstream operation: {operation}") from error


def canonicalize_setup_root(root: str | Path) -> Path:
    """Expand a setup target and reject symlinks in its existing path components."""
    expanded = Path(root).expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    candidate = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        candidate /= component
        if candidate.is_symlink():
            raise SetupError(f"Setup root must not contain a symlink: {candidate}")
        if not candidate.exists():
            break
    return absolute.resolve(strict=False)


def _validate_clone_target(target: Path) -> None:
    """Allow cloning only into an absent or empty canonical target directory."""
    if target.is_symlink():
        raise SetupError(f"Clone target must not be a symlink: {target}")
    if target.exists() and not target.is_dir():
        raise SetupError(f"Clone target is not a directory: {target}")
    if target.is_dir() and any(target.iterdir()):
        raise SetupError(f"Clone target is not empty: {target}")
    if target.parent.exists() and not target.parent.is_dir():
        raise SetupError(f"Clone target parent is not a directory: {target.parent}")


def validate_clone_target(root: str | Path) -> None:
    """Validate a clone target after canonicalizing it once for external Git."""
    _validate_clone_target(canonicalize_setup_root(root))


def _clone_target_identity(target: Path) -> tuple[int, int] | None:
    try:
        metadata = target.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(metadata.st_mode):
        raise SetupError(f"Clone target must not be a symlink: {target}")
    return metadata.st_dev, metadata.st_ino


def _revalidate_clone_reservation(reservation: _CloneReservation) -> None:
    canonical_target = canonicalize_setup_root(reservation.path)
    if canonical_target != reservation.path:
        raise SetupError(f"Clone target identity changed: {reservation.path}")
    identity = _clone_target_identity(reservation.path)
    if identity != (reservation.device, reservation.inode):
        raise SetupError(f"Clone target identity changed: {reservation.path}")
    metadata = reservation.path.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise SetupError(f"Clone target is not a directory: {reservation.path}")
    if any(reservation.path.iterdir()):
        raise SetupError(
            f"Clone target changed and is no longer empty: {reservation.path}"
        )


def _reserve_clone_target(
    target: Path, initial_identity: tuple[int, int] | None
) -> _CloneReservation:
    created = initial_identity is None
    if created:
        try:
            target.mkdir(mode=0o700)
        except FileExistsError as error:
            raise SetupError(
                f"Clone target changed after validation: {target}"
            ) from error
        except OSError as error:
            raise SetupError(
                f"Could not reserve clone target {target}: {error}"
            ) from error
    elif _clone_target_identity(target) != initial_identity:
        raise SetupError(f"Clone target identity changed after validation: {target}")

    canonical_target = canonicalize_setup_root(target)
    metadata = canonical_target.lstat()
    reservation = _CloneReservation(
        path=canonical_target,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        created=created,
    )
    _revalidate_clone_reservation(reservation)
    return reservation


def _cleanup_clone_reservation(reservation: _CloneReservation) -> None:
    if not reservation.created:
        return
    try:
        _revalidate_clone_reservation(reservation)
        reservation.path.rmdir()
    except (OSError, SetupError):
        return


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    subprocess.run(list(command), cwd=cwd, check=True, env=env)


def _git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for variable in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR"):
        environment.pop(variable, None)
    return environment


def _validated_root(root: str | Path, manifest: Mapping[str, object]) -> Path:
    upstream_root = Path(root)
    adapter.validate_upstream(upstream_root, manifest)
    try:
        symbolic_ref = subprocess.run(
            ["git", "-C", str(upstream_root), "symbolic-ref", "--quiet", "HEAD"],
            check=False,
            text=True,
            capture_output=True,
            env=_git_environment(),
        )
    except OSError as error:
        raise SetupError(f"Could not verify detached upstream HEAD: {error}") from error
    if symbolic_ref.returncode == 0:
        raise SetupError(
            "Upstream HEAD must be detached at the pinned commit; "
            f"currently on {symbolic_ref.stdout.strip()}."
        )
    if symbolic_ref.returncode != 1:
        raise SetupError(
            "Could not verify detached upstream HEAD: "
            f"{symbolic_ref.stderr.strip() or symbolic_ref.returncode}"
        )
    return upstream_root


def _venv_python_path(root: Path) -> Path:
    relative_path = (
        Path(".venv/Scripts/python.exe")
        if os.name == "nt"
        else Path(".venv/bin/python")
    )
    return root / relative_path


def _dependency_probe_error(root: Path, environ: Mapping[str, str]) -> str | None:
    try:
        adapter.require_single_cuda_device(environ)
    except adapter.AdapterError as error:
        return str(error)
    command = [
        "uv",
        "run",
        "--frozen",
        "--no-sync",
        "python",
        "-B",
        "-c",
        f"{_DEPENDENCY_PROBE}; {_KERNEL_PROBE}",
    ]
    environment = dict(environ)
    environment["HF_HUB_OFFLINE"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            text=True,
            capture_output=True,
            env=environment,
        )
    except OSError as error:
        return f"upstream dependency smoke probe or GPU kernel check could not run: {error}"
    if completed.returncode == 0:
        return None
    detail = completed.stderr.strip() or completed.stdout.strip()
    suffix = f" ({detail})" if detail else ""
    return (
        "upstream dependency smoke probe or cached GPU kernel check failed under "
        "frozen/no-sync offline mode; run setup_upstream.py sync, then "
        f"setup_upstream.py kernel.{suffix}"
    )


def readiness_errors(
    root: str | Path,
    *,
    cache_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Return all non-mutating operator prerequisites missing for evaluation."""
    upstream_root = Path(root)
    cache_path = (
        DEFAULT_CACHE_ROOT
        if cache_root is None
        else Path(cache_root).expanduser().resolve()
    )
    errors: list[str] = []
    environment = os.environ.copy()
    if environ is not None:
        environment.update(environ)
    uv_available = shutil.which("uv") is not None
    if not uv_available:
        errors.append("uv executable is unavailable; install uv and rerun setup sync.")
    venv_directory = upstream_root / ".venv"
    if not venv_directory.is_dir():
        errors.append("upstream .venv is missing; run setup_upstream.py sync.")
    venv_python = _venv_python_path(upstream_root)
    venv_python_available = venv_python.is_file() and os.access(venv_python, os.X_OK)
    if venv_directory.is_dir() and not venv_python_available:
        errors.append(
            "upstream virtual environment interpreter is missing or not executable at "
            f"{venv_python}; run setup_upstream.py sync."
        )
    if uv_available and venv_python_available:
        probe_error = _dependency_probe_error(upstream_root, environment)
        if probe_error is not None:
            errors.append(probe_error)

    data_dir = cache_path / "data"
    training_files = [
        path
        for path in data_dir.glob("shard_*.parquet")
        if path.name != _PINNED_VALIDATION_FILENAME and path.is_file()
    ]
    if not training_files:
        errors.append(
            "training parquet is missing from the Autoresearch cache; "
            "run setup_upstream.py prepare."
        )
    if not (data_dir / _PINNED_VALIDATION_FILENAME).is_file():
        errors.append(
            "pinned validation parquet shard_06542.parquet is missing; "
            "run setup_upstream.py prepare."
        )

    tokenizer_dir = cache_path / "tokenizer"
    if not (tokenizer_dir / "tokenizer.pkl").is_file():
        errors.append(
            "tokenizer.pkl is missing from the Autoresearch cache; "
            "run setup_upstream.py prepare."
        )
    if not (tokenizer_dir / "token_bytes.pt").is_file():
        errors.append(
            "token_bytes.pt is missing from the Autoresearch cache; "
            "run setup_upstream.py prepare."
        )
    return errors


def require_readiness(
    root: str | Path,
    *,
    cache_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Raise one actionable error if the validated checkout cannot evaluate yet."""
    errors = readiness_errors(root, cache_root=cache_root, environ=environ)
    if errors:
        raise ReadinessError(
            "Upstream checkout is not ready:\n- " + "\n- ".join(errors)
        )


def run_operation(
    operation: str,
    root: str | Path,
    *,
    manifest_path: str | Path = MANIFEST_PATH,
) -> None:
    """Run one deliberate setup action; evaluation never calls this function."""
    manifest = adapter.load_manifest(manifest_path)
    target = canonicalize_setup_root(root)
    if operation == "clone":
        initial_identity = _clone_target_identity(target)
        _validate_clone_target(target)
        reservation = _reserve_clone_target(target, initial_identity)
        try:
            _revalidate_clone_reservation(reservation)
            git_environment = _git_environment()
            _run(build_clone_command(reservation.path, manifest), env=git_environment)
            _run(
                build_checkout_command(reservation.path, manifest),
                env=git_environment,
            )
            _validated_root(reservation.path, manifest)
        except Exception:
            _cleanup_clone_reservation(reservation)
            raise
        return

    upstream_root = _validated_root(target, manifest)
    if operation == "check":
        require_readiness(upstream_root)
        return
    if operation not in {"sync", "prepare", "kernel", "baseline"}:
        raise SetupError(f"Unsupported setup operation: {operation}")
    if operation == "baseline":
        require_readiness(upstream_root)
    if operation == "kernel":
        adapter.require_single_cuda_device(os.environ)
    _run(build_upstream_command(operation), cwd=upstream_root)
    _validated_root(upstream_root, manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explicit setup commands for the external Autoresearch checkout."
    )
    parser.add_argument(
        "operation",
        choices=("clone", "sync", "prepare", "kernel", "check", "baseline"),
    )
    parser.add_argument(
        "--root", required=True, type=Path, help="External Autoresearch checkout"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one setup subcommand and return a shell-compatible exit status."""
    arguments = _parser().parse_args(argv)
    try:
        run_operation(arguments.operation, arguments.root)
    except (
        adapter.AdapterError,
        OSError,
        SetupError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"setup failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
