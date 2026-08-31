"""Evaluate an external Autoresearch candidate checkout for ShinkaEvolve."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import adapter
import psutil

MANIFEST_PATH = Path(__file__).with_name("upstream_manifest.json")
_TERMINATION_GRACE_SECONDS = 1.0
_RUNTIME_REPORTING_TOLERANCE_SECONDS = 0.5


def _write_artifact(path: Path, content: str) -> None:
    """Atomically replace one result artifact with UTF-8 content."""
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _write_results(
    results_dir: Path,
    metrics: dict[str, Any],
    correct: bool,
    error: str,
    stdout: str,
    stderr: str,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_artifact(results_dir / "candidate_stdout.log", stdout)
    _write_artifact(results_dir / "candidate_stderr.log", stderr)
    _write_artifact(
        results_dir / "metrics.json",
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
    )
    _write_artifact(
        results_dir / "correct.json",
        json.dumps({"correct": correct, "error": error}, indent=2, allow_nan=False)
        + "\n",
    )


def _refresh_process_tree(root: psutil.Process, targets: set[psutil.Process]) -> None:
    """Add currently observable descendants to a stable process set."""
    targets.add(root)
    try:
        targets.update(root.children(recursive=True))
    except (
        psutil.AccessDenied,
        psutil.NoSuchProcess,
        psutil.ZombieProcess,
        PermissionError,
    ):
        pass


def _signal_processes(
    root: psutil.Process,
    targets: set[psutil.Process],
    method: str,
    *,
    include_root: bool,
) -> None:
    """Signal observed descendants before their root process."""
    ordered_targets = [target for target in targets if target != root]
    if include_root:
        ordered_targets.append(root)
    for target in ordered_targets:
        try:
            getattr(target, method)()
        except (
            psutil.AccessDenied,
            psutil.NoSuchProcess,
            psutil.ZombieProcess,
            PermissionError,
        ):
            pass


def _signal_process_group(process_id: int, requested_signal: int) -> None:
    """Signal the original POSIX group even if its leader has exited."""
    try:
        os.killpg(process_id, requested_signal)
    except (PermissionError, ProcessLookupError):
        pass


def _stop_process_group(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Terminate and reap the candidate group and observed descendants."""
    root: psutil.Process | None
    targets: set[psutil.Process] = set()
    try:
        root = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        root = None

    if root is not None:
        _refresh_process_tree(root, targets)
    if os.name == "posix":
        _signal_process_group(process.pid, signal.SIGTERM)
    if root is not None:
        _signal_processes(
            root,
            targets,
            "terminate",
            include_root=os.name != "posix",
        )
    elif os.name != "posix":
        process.terminate()

    output: tuple[str, str] | None = None
    try:
        output = process.communicate(timeout=_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass

    if root is not None:
        _refresh_process_tree(root, targets)
    if os.name == "posix":
        _signal_process_group(process.pid, signal.SIGKILL)
    if root is not None:
        _signal_processes(
            root,
            targets,
            "kill",
            include_root=os.name != "posix",
        )
        descendants = [target for target in targets if target != root]
        _, alive = psutil.wait_procs(descendants, timeout=_TERMINATION_GRACE_SECONDS)
        for target in alive:
            try:
                target.kill()
            except (
                psutil.AccessDenied,
                psutil.NoSuchProcess,
                psutil.ZombieProcess,
                PermissionError,
            ):
                pass
        if alive:
            psutil.wait_procs(alive, timeout=_TERMINATION_GRACE_SECONDS)
    elif os.name != "posix":
        process.kill()
    if output is None:
        output = process.communicate()
    return output


def _run_candidate(
    command: list[str], upstream_root: Path, environment: dict[str, str], timeout: float
) -> tuple[str, str, float, str | None]:
    """Run a candidate, decoding invalid UTF-8 as the replacement character."""
    started_at = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=upstream_root,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        stdout, stderr = _stop_process_group(process)
        return stdout, stderr, time.perf_counter() - started_at, "Candidate timed out."

    runtime_seconds = time.perf_counter() - started_at
    if process.returncode != 0:
        return (
            stdout,
            stderr,
            runtime_seconds,
            f"Candidate exited with status {process.returncode}.",
        )
    return stdout, stderr, runtime_seconds, None


def _failure_metrics(
    error: str,
    runtime_seconds: float,
    upstream_commit: str | None,
    candidate_sha256: str | None,
) -> dict[str, Any]:
    return {
        "combined_score": 0.0,
        "public": {
            "valid": False,
            "runtime_seconds": runtime_seconds,
            "upstream_commit": upstream_commit,
            "candidate_sha256": candidate_sha256,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "error": error,
        },
        "private": {},
    }


def _candidate_sha256(program_path: str | os.PathLike[str]) -> str:
    return hashlib.sha256(Path(program_path).read_bytes()).hexdigest()


def evaluate_candidate(
    program_path: str | os.PathLike[str],
    results_dir: str | os.PathLike[str],
    upstream_root: str | os.PathLike[str],
    timeout_seconds: float = 600.0,
    *,
    manifest_path: str | os.PathLike[str] = MANIFEST_PATH,
) -> tuple[dict[str, Any], bool, str]:
    """Run a candidate against a pinned checkout and write Shinka artifacts."""
    results_path = Path(results_dir)
    stdout = ""
    stderr = ""
    runtime_seconds = 0.0
    upstream_commit: str | None = None
    candidate_sha256: str | None = None
    error = ""
    correct = False

    try:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive and finite.")
        adapter.require_single_cuda_device(os.environ)
        candidate_sha256 = _candidate_sha256(program_path)
        manifest = adapter.load_manifest(manifest_path)
        expected_commit = manifest["commit"]
        assert isinstance(expected_commit, str)
        upstream_commit = expected_commit
        upstream_path = Path(upstream_root).resolve()
        adapter.validate_upstream(upstream_path, manifest)
        before_snapshot = adapter.snapshot_protected_files(upstream_path, manifest)

        try:
            stdout, stderr, runtime_seconds, run_error = _run_candidate(
                adapter.build_training_command(upstream_path, program_path),
                upstream_path,
                adapter.build_training_environment(upstream_path, os.environ),
                timeout_seconds,
            )
        finally:
            try:
                after_snapshot = adapter.snapshot_protected_files(
                    upstream_path, manifest
                )
                if after_snapshot != before_snapshot:
                    raise adapter.AdapterError(
                        "Protected upstream files changed during run."
                    )
            except adapter.AdapterError as snapshot_error:
                run_error = str(snapshot_error)

        if run_error is not None:
            raise RuntimeError(run_error)
        summary = adapter.parse_official_summary(stdout)
        training_seconds = summary["training_seconds"]
        assert isinstance(training_seconds, float)
        if training_seconds > (runtime_seconds + _RUNTIME_REPORTING_TOLERANCE_SECONDS):
            raise adapter.AdapterError(
                "Reported training_seconds exceeds evaluator wall-clock runtime."
            )
        val_bpb = summary.pop("val_bpb")
        assert isinstance(val_bpb, float)
        metrics = {
            "combined_score": adapter.score_from_bpb(val_bpb),
            "public": {
                "valid": True,
                "val_bpb": val_bpb,
                "runtime_seconds": runtime_seconds,
                **summary,
                "upstream_commit": upstream_commit,
                "candidate_sha256": candidate_sha256,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "private": {},
        }
        correct = True
    except (
        adapter.AdapterError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exception:
        error = f"{type(exception).__name__}: {exception}"
        metrics = _failure_metrics(
            error, runtime_seconds, upstream_commit, candidate_sha256
        )

    _write_results(results_path, metrics, correct, error, stdout, stderr)
    return metrics, correct, error


def main(
    program_path: str,
    results_dir: str,
    upstream_root: str,
    timeout_seconds: float = 600.0,
) -> None:
    """Evaluate one external candidate and emit Shinka-compatible artifacts."""
    metrics, correct, error = evaluate_candidate(
        program_path, results_dir, upstream_root, timeout_seconds
    )
    print(f"Evaluated program: {program_path}")
    print(f"Results saved to: {results_dir}")
    print(f"Correct: {correct}")
    if error:
        print(f"Error: {error}")
    print(f"Combined score: {metrics['combined_score']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an Autoresearch candidate")
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    arguments = parser.parse_args()
    main(
        arguments.program_path,
        arguments.results_dir,
        arguments.upstream_root,
        arguments.timeout_seconds,
    )
