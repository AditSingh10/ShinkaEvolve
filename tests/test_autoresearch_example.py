from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "autoresearch"
ADAPTER_PATH = EXAMPLE_DIR / "adapter.py"
EVALUATOR_PATH = EXAMPLE_DIR / "evaluate.py"
SETUP_PATH = EXAMPLE_DIR / "setup_upstream.py"
RUNNER_PATH = EXAMPLE_DIR / "run_evo.py"
MANIFEST_PATH = EXAMPLE_DIR / "upstream_manifest.json"
UPSTREAM_REPOSITORY = "https://github.com/karpathy/autoresearch"
UPSTREAM_COMMIT = "228791fb499afffb54b46200aca536f79142f117"
UPSTREAM_TRAIN_SHA256 = (
    "2954175f4ac42ad65164aef40910ef953789abcd05a5cc886ac9ba5a00814414"
)


def _load_adapter():
    spec = importlib.util.spec_from_file_location("autoresearch_adapter", ADAPTER_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_evaluator():
    example_directory = str(EXAMPLE_DIR)
    sys.path.insert(0, example_directory)
    try:
        spec = importlib.util.spec_from_file_location(
            "autoresearch_external_evaluator", EVALUATOR_PATH
        )
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(example_directory)


def _load_example_module(name: str, path: Path):
    example_directory = str(EXAMPLE_DIR)
    sys.path.insert(0, example_directory)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(example_directory)


@pytest.fixture
def adapter():
    return _load_adapter()


@pytest.fixture
def evaluator():
    return _load_evaluator()


@pytest.fixture
def setup_upstream():
    return _load_example_module("autoresearch_setup_upstream", SETUP_PATH)


@pytest.fixture
def runner_module():
    return _load_example_module("autoresearch_runner", RUNNER_PATH)


def _sha256(contents: str) -> str:
    return hashlib.sha256(contents.encode("utf-8")).hexdigest()


def _manifest(protected_files: Mapping[str, object]) -> dict[str, object]:
    return {
        "repository": UPSTREAM_REPOSITORY,
        "commit": UPSTREAM_COMMIT,
        "protected_files": protected_files,
    }


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _git(*args: str, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _upstream_tree(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "upstream"
    root.mkdir()
    protected_contents = {
        ".python-version": "3.12\n",
        "prepare.py": "print('prepare')\n",
        "pyproject.toml": "[project]\nname = 'fake'\n",
        "uv.lock": "version = 1\n",
    }
    for relative_path, contents in protected_contents.items():
        (root / relative_path).write_text(contents, encoding="utf-8")

    _git("init", cwd=root)
    _git("config", "user.email", "tests@example.invalid", cwd=root)
    _git("config", "user.name", "Autoresearch tests", cwd=root)
    _git("add", ".", cwd=root)
    _git("commit", "-m", "fake upstream", cwd=root)

    protected_hashes = {
        relative_path: _sha256(contents)
        for relative_path, contents in protected_contents.items()
    }
    return root, _manifest(protected_hashes)


def _write_candidate(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")


def _write_uv_launcher(directory: Path) -> None:
    launcher = directory / "uv"
    launcher.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "run" ]; then\n'
        "    shift\n"
        '    if [ "$1" = "--frozen" ]; then shift; fi\n'
        '    if [ "$1" = "--no-sync" ]; then shift; fi\n'
        '    if [ "$1" = "python" ]; then\n'
        "        shift\n"
        '        if [ "$1" = "-B" ] && [ "$2" = "-c" ]; then exit 0; fi\n'
        '        exec "${PYTHON_EXECUTABLE}" "$@"\n'
        "    fi\n"
        "fi\n"
        "exit 97\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)


def _write_probe_uv_launcher(directory: Path) -> None:
    launcher = directory / "uv"
    launcher.write_text(
        "#!/bin/sh\n"
        'printf \'HF_HUB_OFFLINE=%s\\n\' "${HF_HUB_OFFLINE:-}" > "${UV_PROBE_LOG}"\n'
        'printf \'%s\\n\' "$@" >> "${UV_PROBE_LOG}"\n'
        'exit "${UV_PROBE_EXIT_CODE:-0}"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)


def _external_evaluation_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    upstream_root, manifest = _upstream_tree(tmp_path)
    manifest["commit"] = _git("rev-parse", "HEAD", cwd=upstream_root)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, manifest)
    launcher_directory = tmp_path / "bin"
    launcher_directory.mkdir()
    _write_uv_launcher(launcher_directory)
    monkeypatch.setenv(
        "PATH", f"{launcher_directory}{os.pathsep}{os.environ.get('PATH', '')}"
    )
    monkeypatch.setenv("PYTHON_EXECUTABLE", sys.executable)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    return upstream_root, manifest_path, tmp_path / "results"


def _pinned_upstream_tree(tmp_path: Path) -> tuple[Path, Path]:
    root, manifest = _upstream_tree(tmp_path)
    (root / "train.py").write_text("print('train')\n", encoding="utf-8")
    _git("add", "train.py", cwd=root)
    _git("commit", "-m", "add train", cwd=root)
    manifest["commit"] = _git("rev-parse", "HEAD", cwd=root)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, manifest)
    return root, manifest_path


def _write_venv_python(root: Path) -> Path:
    relative_path = (
        Path(".venv/Scripts/python.exe")
        if os.name == "nt"
        else Path(".venv/bin/python")
    )
    interpreter = root / relative_path
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    interpreter.chmod(0o755)
    return interpreter


def _write_ready_cache_state(cache_root: Path) -> None:
    data_dir = cache_root / "data"
    tokenizer_dir = cache_root / "tokenizer"
    data_dir.mkdir(parents=True)
    tokenizer_dir.mkdir()
    (data_dir / "shard_00000.parquet").write_bytes(b"train")
    (data_dir / "shard_06542.parquet").write_bytes(b"validation")
    (tokenizer_dir / "tokenizer.pkl").write_bytes(b"tokenizer")
    (tokenizer_dir / "token_bytes.pt").write_bytes(b"tokens")


def _write_ready_upstream_state(root: Path, cache_root: Path) -> None:
    _write_venv_python(root)
    _write_ready_cache_state(cache_root)


def _read_artifacts(results_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    correct = json.loads((results_dir / "correct.json").read_text(encoding="utf-8"))
    return metrics, correct


def test_setup_builds_frozen_upstream_uv_commands(setup_upstream, tmp_path):
    root = tmp_path / "external-autoresearch"
    manifest = _manifest({"prepare.py": "a" * 64})

    assert setup_upstream.build_clone_command(root, manifest) == [
        "git",
        "clone",
        UPSTREAM_REPOSITORY,
        str(root),
    ]
    assert setup_upstream.build_checkout_command(root, manifest) == [
        "git",
        "-C",
        str(root),
        "checkout",
        "--detach",
        UPSTREAM_COMMIT,
    ]
    assert setup_upstream.build_upstream_command("sync") == ["uv", "sync", "--frozen"]
    assert setup_upstream.build_upstream_command("prepare") == [
        "uv",
        "run",
        "--frozen",
        "--no-sync",
        "prepare.py",
    ]
    assert setup_upstream.build_upstream_command("kernel")[:6] == [
        "uv",
        "run",
        "--frozen",
        "--no-sync",
        "python",
        "-B",
    ]
    assert setup_upstream.build_upstream_command("baseline") == [
        "uv",
        "run",
        "--frozen",
        "--no-sync",
        "train.py",
    ]


def test_setup_cli_does_not_offer_an_unusable_cache_override(setup_upstream):
    assert "--cache-root" not in setup_upstream._parser().format_help()


def test_pinned_manifest_protects_the_official_train_seed(adapter):
    manifest = adapter.load_manifest(MANIFEST_PATH)

    assert manifest["protected_files"]["train.py"] == UPSTREAM_TRAIN_SHA256


def test_setup_rejects_a_conflicting_or_nonempty_clone_target(setup_upstream, tmp_path):
    missing_root = tmp_path / "missing"
    populated_root = tmp_path / "populated"
    populated_root.mkdir()
    (populated_root / "unrelated.txt").write_text("keep me", encoding="utf-8")

    assert setup_upstream.validate_clone_target(missing_root) is None
    with pytest.raises(setup_upstream.SetupError, match="not empty"):
        setup_upstream.validate_clone_target(populated_root)
    assert (populated_root / "unrelated.txt").read_text(encoding="utf-8") == "keep me"


def test_setup_rejects_symlinked_clone_paths_and_expands_the_root(
    setup_upstream, monkeypatch, tmp_path
):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(setup_upstream.SetupError, match="symlink"):
        setup_upstream.validate_clone_target(linked_parent / "checkout")

    monkeypatch.setenv("HOME", str(tmp_path))
    assert setup_upstream.canonicalize_setup_root("~/checkout") == (
        tmp_path / "checkout"
    )


def test_clone_reserves_an_absent_destination_before_invoking_git(
    setup_upstream, monkeypatch, tmp_path
):
    target = tmp_path / "checkout"
    manifest = _manifest({"prepare.py": "a" * 64})
    commands: list[list[str]] = []
    monkeypatch.setattr(setup_upstream.adapter, "load_manifest", lambda _: manifest)
    monkeypatch.setattr(
        setup_upstream, "_validated_root", lambda path, supplied_manifest: target
    )

    def run(command, *, cwd=None, env=None):
        assert target.is_dir()
        assert target.is_symlink() is False
        assert env is not None
        commands.append(list(command))

    monkeypatch.setattr(setup_upstream, "_run", run)

    setup_upstream.run_operation("clone", target)

    assert commands == [
        setup_upstream.build_clone_command(target, manifest),
        setup_upstream.build_checkout_command(target, manifest),
    ]


def test_setup_git_commands_scrub_repository_routing_environment(
    setup_upstream, monkeypatch, tmp_path
):
    target = tmp_path / "checkout"
    manifest = _manifest({"prepare.py": "a" * 64})
    monkeypatch.setattr(setup_upstream.adapter, "load_manifest", lambda _: manifest)
    monkeypatch.setattr(
        setup_upstream, "_validated_root", lambda path, supplied_manifest: target
    )
    for variable in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR"):
        monkeypatch.setenv(variable, f"ambient-{variable.lower()}")

    observed_environments: list[dict[str, str] | None] = []

    def run(command, *, cwd=None, env=None):
        observed_environments.append(env)

    monkeypatch.setattr(setup_upstream, "_run", run)

    setup_upstream.run_operation("clone", target)

    assert len(observed_environments) == 2
    for environment in observed_environments:
        assert environment is not None
        assert "GIT_DIR" not in environment
        assert "GIT_WORK_TREE" not in environment
        assert "GIT_COMMON_DIR" not in environment


@pytest.mark.parametrize("value", ["", "   ", "-1", "0,1", "GPU-a,GPU-b"])
def test_adapter_rejects_cuda_visibility_that_is_not_one_device(adapter, value):
    with pytest.raises(adapter.AdapterError, match="CUDA_VISIBLE_DEVICES"):
        adapter.require_single_cuda_device({"CUDA_VISIBLE_DEVICES": value})


@pytest.mark.parametrize("value", ["0", "17", "GPU-1234abcd", "MIG-1234abcd"])
def test_adapter_accepts_one_cuda_ordinal_or_uuid(adapter, value):
    assert adapter.require_single_cuda_device({"CUDA_VISIBLE_DEVICES": value}) == value


def test_clone_refuses_a_destination_replaced_after_initial_validation(
    setup_upstream, monkeypatch, tmp_path
):
    target = tmp_path / "checkout"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    original_validate = setup_upstream._validate_clone_target
    manifest = _manifest({"prepare.py": "a" * 64})
    monkeypatch.setattr(setup_upstream.adapter, "load_manifest", lambda _: manifest)

    def validate_then_replace(path):
        original_validate(path)
        path.symlink_to(replacement, target_is_directory=True)

    git_called = False

    def reject_git_call(command, *, cwd=None):
        nonlocal git_called
        git_called = True
        raise AssertionError("Git was invoked after clone destination replacement")

    monkeypatch.setattr(setup_upstream, "_validate_clone_target", validate_then_replace)
    monkeypatch.setattr(setup_upstream, "_run", reject_git_call)

    with pytest.raises(setup_upstream.SetupError, match="symlink|identity|changed"):
        setup_upstream.run_operation("clone", target)

    assert git_called is False
    assert target.is_symlink()
    assert replacement.is_dir()


@pytest.mark.parametrize("value", ["", "   ", "-1", "0,1", "GPU-a,GPU-b"])
def test_runner_rejects_cuda_visibility_that_is_not_one_device(runner_module, value):
    with pytest.raises(
        runner_module.RunnerPrerequisiteError, match="CUDA_VISIBLE_DEVICES"
    ):
        runner_module.require_cuda_visible_devices({"CUDA_VISIBLE_DEVICES": value})


@pytest.mark.parametrize("value", ["0", "17", "GPU-1234abcd", "MIG-1234abcd"])
def test_runner_accepts_one_cuda_ordinal_or_uuid(runner_module, value):
    assert (
        runner_module.require_cuda_visible_devices({"CUDA_VISIBLE_DEVICES": value})
        == value
    )


def test_runner_reports_missing_root_and_cuda_assignment(runner_module):
    with pytest.raises(
        runner_module.RunnerPrerequisiteError, match="AUTORESEARCH_ROOT"
    ):
        runner_module.resolve_upstream_root({})
    with pytest.raises(
        runner_module.RunnerPrerequisiteError, match="CUDA_VISIBLE_DEVICES"
    ):
        runner_module.require_cuda_visible_devices({})


def test_setup_check_rejects_an_attached_pinned_commit(
    setup_upstream, monkeypatch, tmp_path
):
    root, manifest_path = _pinned_upstream_tree(tmp_path)
    cache_root = tmp_path / "cache"
    _write_ready_upstream_state(root, cache_root)
    launcher_directory = tmp_path / "bin"
    launcher_directory.mkdir()
    _write_uv_launcher(launcher_directory)
    monkeypatch.setattr(setup_upstream, "DEFAULT_CACHE_ROOT", cache_root)
    monkeypatch.setenv(
        "PATH", f"{launcher_directory}{os.pathsep}{os.environ.get('PATH', '')}"
    )

    with pytest.raises(setup_upstream.SetupError, match="detached"):
        setup_upstream.run_operation("check", root, manifest_path=manifest_path)


def test_readiness_reports_each_missing_operator_prerequisite(
    setup_upstream, monkeypatch, tmp_path
):
    monkeypatch.setenv("PATH", "")
    root = tmp_path / "upstream"
    root.mkdir()

    with pytest.raises(setup_upstream.ReadinessError) as error:
        setup_upstream.require_readiness(root, cache_root=tmp_path / "cache")

    message = str(error.value)
    for expected in (
        "uv executable",
        ".venv",
        "training parquet",
        "validation parquet",
        "tokenizer.pkl",
        "token_bytes.pt",
    ):
        assert expected in message


def test_readiness_rejects_an_empty_virtual_environment_before_probe(
    setup_upstream, monkeypatch, tmp_path
):
    root = tmp_path / "upstream"
    root.mkdir()
    (root / ".venv").mkdir()
    cache_root = tmp_path / "cache"
    _write_ready_cache_state(cache_root)
    launcher_directory = tmp_path / "bin"
    launcher_directory.mkdir()
    _write_probe_uv_launcher(launcher_directory)
    probe_log = tmp_path / "probe.log"
    monkeypatch.setenv("PATH", str(launcher_directory))
    monkeypatch.setenv("UV_PROBE_LOG", str(probe_log))

    with pytest.raises(setup_upstream.ReadinessError, match="interpreter"):
        setup_upstream.require_readiness(root, cache_root=cache_root)

    assert probe_log.exists() is False


def test_readiness_requires_the_platform_virtual_environment_interpreter(
    setup_upstream, monkeypatch, tmp_path
):
    root = tmp_path / "upstream"
    root.mkdir()
    interpreter = _write_venv_python(root)
    interpreter.unlink()
    cache_root = tmp_path / "cache"
    _write_ready_cache_state(cache_root)
    monkeypatch.setattr(setup_upstream.shutil, "which", lambda executable: "/bin/uv")

    with pytest.raises(setup_upstream.ReadinessError) as error:
        setup_upstream.require_readiness(root, cache_root=cache_root)

    assert str(interpreter) in str(error.value)


def test_readiness_reports_a_failed_frozen_dependency_probe(
    setup_upstream, monkeypatch, tmp_path
):
    root = tmp_path / "upstream"
    root.mkdir()
    _write_ready_upstream_state(root, tmp_path / "cache")
    launcher_directory = tmp_path / "bin"
    launcher_directory.mkdir()
    _write_probe_uv_launcher(launcher_directory)
    probe_log = tmp_path / "probe.log"
    monkeypatch.setenv("PATH", str(launcher_directory))
    monkeypatch.setenv("UV_PROBE_LOG", str(probe_log))
    monkeypatch.setenv("UV_PROBE_EXIT_CODE", "23")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    with pytest.raises(setup_upstream.ReadinessError, match="dependency smoke probe"):
        setup_upstream.require_readiness(root, cache_root=tmp_path / "cache")

    assert probe_log.read_text(encoding="utf-8").splitlines()[:5] == [
        "HF_HUB_OFFLINE=1",
        "run",
        "--frozen",
        "--no-sync",
        "python",
    ]


def test_readiness_uses_a_non_syncing_dependency_probe_for_a_healthy_environment(
    setup_upstream, monkeypatch, tmp_path
):
    root = tmp_path / "upstream"
    root.mkdir()
    cache_root = tmp_path / "cache"
    _write_ready_upstream_state(root, cache_root)
    launcher_directory = tmp_path / "bin"
    launcher_directory.mkdir()
    _write_probe_uv_launcher(launcher_directory)
    probe_log = tmp_path / "probe.log"
    monkeypatch.setenv("PATH", str(launcher_directory))
    monkeypatch.setenv("UV_PROBE_LOG", str(probe_log))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    assert setup_upstream.require_readiness(root, cache_root=cache_root) is None
    assert probe_log.is_file()
    arguments = probe_log.read_text(encoding="utf-8").splitlines()
    assert arguments == [
        "HF_HUB_OFFLINE=1",
        "run",
        "--frozen",
        "--no-sync",
        "python",
        "-B",
        "-c",
        (
            "import kernels, matplotlib, numpy, pandas, pyarrow, requests, "
            "rustbpe, tiktoken, torch; import torch; from kernels import "
            "get_kernel; cap = torch.cuda.get_device_capability(); repo = "
            "'varunneal/flash-attention-3' if cap == (9, 0) else "
            "'kernels-community/flash-attn3'; get_kernel(repo)"
        ),
    ]
    assert not {"sync", "install", "prepare.py", "train.py"}.intersection(arguments)


def test_readiness_loads_the_gpu_kernel_with_hub_offline(
    setup_upstream, monkeypatch, tmp_path
):
    root = tmp_path / "upstream"
    root.mkdir()
    cache_root = tmp_path / "cache"
    _write_ready_upstream_state(root, cache_root)
    launcher_directory = tmp_path / "bin"
    launcher_directory.mkdir()
    _write_probe_uv_launcher(launcher_directory)
    probe_log = tmp_path / "probe.log"
    monkeypatch.setenv("PATH", str(launcher_directory))
    monkeypatch.setenv("UV_PROBE_LOG", str(probe_log))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    setup_upstream.require_readiness(root, cache_root=cache_root)

    probe_lines = probe_log.read_text(encoding="utf-8").splitlines()
    assert probe_lines[0] == "HF_HUB_OFFLINE=1"
    assert "get_kernel" in probe_lines[-1]


def test_readiness_requires_a_training_parquet_file(
    setup_upstream, monkeypatch, tmp_path
):
    root = tmp_path / "upstream"
    cache_root = tmp_path / "cache"
    root.mkdir()
    _write_ready_upstream_state(root, cache_root)
    training_shard = cache_root / "data" / "shard_00000.parquet"
    training_shard.unlink()
    training_shard.mkdir()
    monkeypatch.setattr(setup_upstream.shutil, "which", lambda executable: "/bin/uv")

    with pytest.raises(setup_upstream.ReadinessError, match="training parquet"):
        setup_upstream.require_readiness(root, cache_root=cache_root)


def test_runner_preflight_validates_a_ready_detached_pinned_checkout(
    runner_module, monkeypatch, tmp_path
):
    root, manifest_path = _pinned_upstream_tree(tmp_path)
    _git("checkout", "--detach", cwd=root)
    cache_root = tmp_path / "cache"
    _write_ready_upstream_state(root, cache_root)
    launcher_directory = tmp_path / "bin"
    launcher_directory.mkdir()
    _write_uv_launcher(launcher_directory)
    monkeypatch.setenv(
        "PATH", f"{launcher_directory}{os.pathsep}{os.environ.get('PATH', '')}"
    )
    monkeypatch.setattr(runner_module.setup_upstream, "DEFAULT_CACHE_ROOT", cache_root)

    assert (
        runner_module.preflight_evolution(
            root,
            environ={"CUDA_VISIBLE_DEVICES": "GPU-1234abcd"},
            manifest_path=manifest_path,
        )
        == root / "train.py"
    )


def test_setup_operations_validate_before_and_after_execution(
    setup_upstream, monkeypatch, tmp_path
):
    events: list[tuple[str, object]] = []
    root = tmp_path / "upstream"
    manifest = _manifest({"prepare.py": "a" * 64})
    monkeypatch.setattr(setup_upstream.adapter, "load_manifest", lambda _: manifest)

    def validated_root(path, supplied_manifest):
        events.append(("validate", (Path(path), supplied_manifest)))
        return root

    def run(command, *, cwd=None):
        events.append(("run", (list(command), cwd)))

    monkeypatch.setattr(setup_upstream, "_validated_root", validated_root)
    monkeypatch.setattr(setup_upstream, "_run", run)

    setup_upstream.run_operation("sync", root)

    assert events == [
        ("validate", (root, manifest)),
        ("run", (["uv", "sync", "--frozen"], root)),
        ("validate", (root, manifest)),
    ]


def test_kernel_setup_prefetches_for_the_assigned_gpu(
    setup_upstream, monkeypatch, tmp_path
):
    root = tmp_path / "upstream"
    manifest = _manifest({"prepare.py": "a" * 64})
    events: list[tuple[list[str], Path | None]] = []
    monkeypatch.setattr(setup_upstream.adapter, "load_manifest", lambda _: manifest)
    monkeypatch.setattr(
        setup_upstream, "_validated_root", lambda path, supplied_manifest: root
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        setup_upstream,
        "_run",
        lambda command, *, cwd=None, env=None: events.append((list(command), cwd)),
    )

    setup_upstream.run_operation("kernel", root)

    assert events == [(setup_upstream.build_upstream_command("kernel"), root)]


def test_setup_propagates_command_failure_without_post_validation(
    setup_upstream, monkeypatch, tmp_path
):
    events: list[str] = []
    root = tmp_path / "upstream"
    manifest = _manifest({"prepare.py": "a" * 64})
    monkeypatch.setattr(setup_upstream.adapter, "load_manifest", lambda _: manifest)
    monkeypatch.setattr(
        setup_upstream,
        "_validated_root",
        lambda path, supplied_manifest: events.append("validate") or root,
    )

    def fail(command, *, cwd=None):
        events.append("run")
        raise subprocess.CalledProcessError(9, command)

    monkeypatch.setattr(setup_upstream, "_run", fail)

    with pytest.raises(subprocess.CalledProcessError):
        setup_upstream.run_operation("sync", root)
    assert events == ["validate", "run"]


def test_runner_builds_an_absolute_scheduler_command_from_any_working_directory(
    runner_module, monkeypatch, tmp_path
):
    from shinka.launch import JobScheduler

    monkeypatch.chdir(tmp_path)
    _, job_config, _ = runner_module.build_configs(tmp_path / "upstream")
    command = JobScheduler(job_type="local", config=job_config)._build_command(
        "candidate.py", "results"
    )

    assert command[1] == str(EXAMPLE_DIR / "evaluate.py")
    assert command[-4:] == [
        "--upstream-root",
        str(tmp_path / "upstream"),
        "--timeout-seconds",
        "600.0",
    ]


def test_committed_manifest_pins_the_required_upstream_files(adapter):
    manifest = adapter.load_manifest(MANIFEST_PATH)

    assert manifest == {
        "repository": UPSTREAM_REPOSITORY,
        "commit": UPSTREAM_COMMIT,
        "protected_files": {
            ".python-version": "7a41a41354ab8049091ef1a00253ca00567f5a1d3f9aef5502c0b4b5ce9ae707",
            "prepare.py": "4f2ba9cbb8ba8c4a3d35be405a913e2f3be3af9aea103ed52ef7b2a662058150",
            "pyproject.toml": "675c150a9e0769f0e39a43eb7d836934266fa348d6e2403521f1ff99f9b9f1af",
            "train.py": UPSTREAM_TRAIN_SHA256,
            "uv.lock": "03174c5cce6387418c5b6cc9bbe8f71ad0ae1e1d6fedeaecae5cdcf7321da0a3",
        },
    }


def test_load_manifest_accepts_a_complete_manifest(adapter, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    expected = _manifest({"prepare.py": "a" * 64})
    _write_manifest(manifest_path, expected)

    assert adapter.load_manifest(manifest_path) == expected


@pytest.mark.parametrize(
    "manifest",
    [
        {},
        {"repository": UPSTREAM_REPOSITORY, "protected_files": {}},
        _manifest({}),
        _manifest({"prepare.py": "not-a-sha256"}),
        _manifest({"../prepare.py": "a" * 64}),
        _manifest({"prepare.py": 7}),
    ],
)
def test_load_manifest_rejects_malformed_data(
    adapter, tmp_path, manifest: dict[str, object]
):
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, manifest)

    with pytest.raises(adapter.AdapterError):
        adapter.load_manifest(manifest_path)


def test_validate_upstream_requires_the_exact_head_and_protected_hashes(
    adapter, tmp_path
):
    root, manifest = _upstream_tree(tmp_path)
    manifest["commit"] = _git("rev-parse", "HEAD", cwd=root)

    assert adapter.validate_upstream(root, manifest) is None
    assert (
        adapter.snapshot_protected_files(root, manifest) == manifest["protected_files"]
    )

    (root / "prepare.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(adapter.AdapterError, match="prepare.py"):
        adapter.validate_upstream(root, manifest)

    manifest["commit"] = "0" * 40
    with pytest.raises(adapter.AdapterError, match="commit"):
        adapter.validate_upstream(root, manifest)


def test_validate_upstream_ignores_inherited_git_checkout_variables(
    adapter, monkeypatch, tmp_path
):
    approved_parent = tmp_path / "approved"
    approved_parent.mkdir()
    approved_root, manifest = _upstream_tree(approved_parent)
    (approved_root / "train.py").write_text("approved\n", encoding="utf-8")
    _git("add", "train.py", cwd=approved_root)
    _git("commit", "-m", "approved train", cwd=approved_root)
    manifest["commit"] = _git("rev-parse", "HEAD", cwd=approved_root)

    supplied_parent = tmp_path / "supplied"
    supplied_parent.mkdir()
    supplied_root, _ = _upstream_tree(supplied_parent)
    (supplied_root / "train.py").write_text("different train\n", encoding="utf-8")
    _git("add", "train.py", cwd=supplied_root)
    _git("commit", "-m", "different train", cwd=supplied_root)

    monkeypatch.setenv("GIT_DIR", str(approved_root / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(approved_root))
    monkeypatch.setenv("GIT_COMMON_DIR", str(approved_root / ".git"))

    with pytest.raises(adapter.AdapterError, match="commit mismatch"):
        adapter.validate_upstream(supplied_root, manifest)


def test_snapshot_protected_files_rejects_a_symlink_outside_upstream_root(
    adapter, tmp_path
):
    root, manifest = _upstream_tree(tmp_path)
    outside_file = tmp_path / "outside_prepare.py"
    outside_file.write_text("print('prepare')\n", encoding="utf-8")
    (root / "prepare.py").unlink()
    (root / "prepare.py").symlink_to(outside_file)

    with pytest.raises(adapter.AdapterError, match="prepare.py"):
        adapter.snapshot_protected_files(root, manifest)


def test_build_training_command_uses_an_absolute_candidate_path(adapter, tmp_path):
    root = tmp_path / "upstream"
    root.mkdir()
    candidate = tmp_path / "candidate.py"
    candidate.write_text("print('candidate')\n", encoding="utf-8")

    assert adapter.build_training_command(root, candidate) == [
        "uv",
        "run",
        "--frozen",
        "--no-sync",
        "python",
        str(candidate.resolve()),
    ]


def test_build_training_environment_preserves_inherited_values(adapter, tmp_path):
    root = tmp_path / "upstream"
    root.mkdir()
    inherited = {"PYTHONPATH": "/existing/path", "KEEP_ME": "yes"}

    environment = adapter.build_training_environment(root, inherited)

    assert environment == {
        "PYTHONPATH": f"{root.resolve()}{os.pathsep}/existing/path",
        "KEEP_ME": "yes",
        "HF_HUB_OFFLINE": "1",
    }
    assert inherited == {"PYTHONPATH": "/existing/path", "KEEP_ME": "yes"}


def test_parse_official_summary_reads_the_single_anchored_bpb(adapter):
    summary = adapter.parse_official_summary(
        "training output\n---\nval_bpb:          1.250000\ntraining_seconds: 300.0\n"
    )

    assert summary["val_bpb"] == 1.25
    assert summary["training_seconds"] == 300.0


def test_parse_official_summary_rejects_unknown_nonempty_suffix(adapter):
    with pytest.raises(adapter.AdapterError):
        adapter.parse_official_summary(
            "---\nval_bpb: 1.250000\nnot an official summary line\n"
        )


def test_parse_official_summary_rejects_malformed_optional_field(adapter):
    with pytest.raises(adapter.AdapterError):
        adapter.parse_official_summary(
            "---\nval_bpb: 1.250000\ntraining_seconds: 300 seconds\n"
        )


def test_parse_official_summary_rejects_duplicate_optional_field(adapter):
    with pytest.raises(adapter.AdapterError):
        adapter.parse_official_summary(
            "---\nval_bpb: 1.250000\ntraining_seconds: 300.0\ntraining_seconds: 301.0\n"
        )


@pytest.mark.parametrize(
    "stdout",
    [
        "---\nval_bpb: 1.250000\n",
        "---\nval_bpb: 1.250000\ntraining_seconds: 299.9\n",
    ],
)
def test_parse_official_summary_requires_the_full_training_budget(adapter, stdout):
    with pytest.raises(adapter.AdapterError, match="training_seconds|300-second"):
        adapter.parse_official_summary(stdout)


@pytest.mark.parametrize("training_seconds", ["330.1", "599.0"])
def test_parse_official_summary_rejects_excessive_training_time(
    adapter, training_seconds
):
    with pytest.raises(adapter.AdapterError, match="training_seconds|budget"):
        adapter.parse_official_summary(
            f"---\nval_bpb: 1.250000\ntraining_seconds: {training_seconds}\n"
        )


@pytest.mark.parametrize(
    "stdout",
    [
        "val_bpb: 1.250000\n",
        "---\nval_bpb: 1.250000\nval_bpb: 1.500000\n",
        "---\nval_bpb: nan\n",
        "---\nval_bpb: -0.100000\n",
        "---\nval_bpb: 1.250000 trailing\n",
    ],
)
def test_parse_official_summary_rejects_malformed_or_ambiguous_bpb(adapter, stdout):
    with pytest.raises(adapter.AdapterError):
        adapter.parse_official_summary(stdout)


def test_score_from_bpb_rewards_lower_finite_validation_loss(adapter):
    assert adapter.score_from_bpb(1.0) == pytest.approx(0.5)
    assert adapter.score_from_bpb(1.0) > adapter.score_from_bpb(2.0)

    with pytest.raises(adapter.AdapterError):
        adapter.score_from_bpb(float("inf"))


def test_external_candidate_writes_valid_summary_metrics_and_logs(
    evaluator, monkeypatch, tmp_path
):
    upstream_root, manifest_path, results_dir = _external_evaluation_inputs(
        tmp_path, monkeypatch
    )
    candidate = tmp_path / "candidate.py"
    _write_candidate(
        candidate,
        "print('candidate source is hashed')\n",
    )
    stdout = "candidate started\n---\nval_bpb: 1.250000\ntraining_seconds: 300.0\n"
    monkeypatch.setattr(
        evaluator,
        "_run_candidate",
        lambda *_args, **_kwargs: (stdout, "", 301.0, None),
    )

    metrics, correct, error = evaluator.evaluate_candidate(
        candidate,
        results_dir,
        upstream_root,
        timeout_seconds=1.0,
        manifest_path=manifest_path,
    )

    artifact_metrics, artifact_correct = _read_artifacts(results_dir)
    assert correct is True
    assert error == ""
    assert metrics == artifact_metrics
    assert artifact_correct == {"correct": True, "error": ""}
    assert metrics["combined_score"] == pytest.approx(1.0 / 2.25)
    assert metrics["public"] == {
        "valid": True,
        "val_bpb": 1.25,
        "runtime_seconds": pytest.approx(metrics["public"]["runtime_seconds"]),
        "training_seconds": 300.0,
        "upstream_commit": _git("rev-parse", "HEAD", cwd=upstream_root),
        "candidate_sha256": _sha256(candidate.read_text(encoding="utf-8")),
        "cuda_visible_devices": "3",
    }
    assert (results_dir / "candidate_stdout.log").read_text(encoding="utf-8") == (
        "candidate started\n---\nval_bpb: 1.250000\ntraining_seconds: 300.0\n"
    )
    assert (results_dir / "candidate_stderr.log").read_text(encoding="utf-8") == ""


def test_evaluator_rejects_training_time_longer_than_measured_wall_runtime(
    evaluator, monkeypatch, tmp_path
):
    upstream_root, manifest_path, results_dir = _external_evaluation_inputs(
        tmp_path, monkeypatch
    )
    candidate = tmp_path / "fabricated_summary.py"
    _write_candidate(
        candidate,
        'print("---")\nprint("val_bpb: 0.0")\nprint("training_seconds: 300.0")\n',
    )

    metrics, correct, error = evaluator.evaluate_candidate(
        candidate,
        results_dir,
        upstream_root,
        timeout_seconds=1.0,
        manifest_path=manifest_path,
    )

    assert correct is False
    assert "wall-clock runtime" in error
    assert metrics["combined_score"] == 0.0


@pytest.mark.parametrize("cuda_visible_devices", [None, "0,1"])
def test_evaluator_rejects_invalid_gpu_assignment_before_launch(
    evaluator, monkeypatch, tmp_path, cuda_visible_devices
):
    upstream_root, manifest_path, results_dir = _external_evaluation_inputs(
        tmp_path, monkeypatch
    )
    marker = tmp_path / "candidate-ran"
    candidate = tmp_path / "candidate.py"
    _write_candidate(
        candidate,
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
    )
    if cuda_visible_devices is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda_visible_devices)

    metrics, correct, error = evaluator.evaluate_candidate(
        candidate,
        results_dir,
        upstream_root,
        timeout_seconds=1.0,
        manifest_path=manifest_path,
    )

    assert correct is False
    assert "CUDA_VISIBLE_DEVICES" in error
    assert metrics["combined_score"] == 0.0
    assert marker.exists() is False


def test_nonzero_exit_is_incorrect_and_preserves_both_logs(
    evaluator, monkeypatch, tmp_path
):
    upstream_root, manifest_path, results_dir = _external_evaluation_inputs(
        tmp_path, monkeypatch
    )
    candidate = tmp_path / "nonzero.py"
    _write_candidate(
        candidate,
        """
import sys
print("before failure")
print("candidate failure", file=sys.stderr)
raise SystemExit(23)
""",
    )

    metrics, correct, error = evaluator.evaluate_candidate(
        candidate,
        results_dir,
        upstream_root,
        timeout_seconds=1.0,
        manifest_path=manifest_path,
    )

    artifact_metrics, artifact_correct = _read_artifacts(results_dir)
    assert correct is False
    assert "status 23" in error
    assert metrics == artifact_metrics
    assert metrics["combined_score"] == 0.0
    assert artifact_correct["correct"] is False
    assert "before failure" in (results_dir / "candidate_stdout.log").read_text(
        encoding="utf-8"
    )
    assert "candidate failure" in (results_dir / "candidate_stderr.log").read_text(
        encoding="utf-8"
    )


def test_malformed_bpb_is_incorrect_after_a_zero_exit(evaluator, monkeypatch, tmp_path):
    upstream_root, manifest_path, results_dir = _external_evaluation_inputs(
        tmp_path, monkeypatch
    )
    candidate = tmp_path / "malformed.py"
    _write_candidate(candidate, 'print("---\\nval_bpb: nan")\n')

    metrics, correct, error = evaluator.evaluate_candidate(
        candidate,
        results_dir,
        upstream_root,
        timeout_seconds=1.0,
        manifest_path=manifest_path,
    )

    assert correct is False
    assert "val_bpb" in error
    assert metrics["combined_score"] == 0.0
    assert _read_artifacts(results_dir)[1]["correct"] is False
    assert "val_bpb: nan" in (results_dir / "candidate_stdout.log").read_text(
        encoding="utf-8"
    )


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_timeout_kills_and_reaps_the_entire_candidate_process_group(
    evaluator, monkeypatch, tmp_path
):
    upstream_root, manifest_path, results_dir = _external_evaluation_inputs(
        tmp_path, monkeypatch
    )
    child_pid_path = tmp_path / "child.pid"
    candidate = tmp_path / "timeout.py"
    _write_candidate(
        candidate,
        f"""
import signal
import subprocess
import sys
import time
from pathlib import Path

child = subprocess.Popen([
    sys.executable,
    "-c",
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
])
Path({str(child_pid_path)!r}).write_text(str(child.pid), encoding="utf-8")
print("child started", flush=True)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
while True:
    time.sleep(0.01)
""",
    )

    metrics, correct, error = evaluator.evaluate_candidate(
        candidate,
        results_dir,
        upstream_root,
        timeout_seconds=0.5,
        manifest_path=manifest_path,
    )

    assert child_pid_path.is_file()
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
    assert correct is False
    assert "timed out" in error
    assert metrics["combined_score"] == 0.0
    assert _read_artifacts(results_dir)[1]["correct"] is False
    assert "child started" in (results_dir / "candidate_stdout.log").read_text(
        encoding="utf-8"
    )


@pytest.mark.skipif(os.name != "posix", reason="detached sessions require POSIX")
def test_timeout_kills_detached_candidate_descendant(evaluator, monkeypatch, tmp_path):
    upstream_root, manifest_path, results_dir = _external_evaluation_inputs(
        tmp_path, monkeypatch
    )
    child_pid_path = tmp_path / "detached-child.pid"
    candidate = tmp_path / "detached_child.py"
    _write_candidate(
        candidate,
        f"""
import signal
import subprocess
import sys
import time
from pathlib import Path

child = subprocess.Popen([
    sys.executable,
    "-c",
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
], start_new_session=True)
Path({str(child_pid_path)!r}).write_text(str(child.pid), encoding="utf-8")
while True:
    time.sleep(0.01)
""",
    )

    try:
        metrics, correct, error = evaluator.evaluate_candidate(
            candidate,
            results_dir,
            upstream_root,
            timeout_seconds=0.5,
            manifest_path=manifest_path,
        )

        assert child_pid_path.is_file()
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        assert correct is False
        assert "timed out" in error
        assert metrics["combined_score"] == 0.0
    finally:
        if child_pid_path.is_file():
            try:
                os.kill(int(child_pid_path.read_text(encoding="utf-8")), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_timeout_kills_descendant_after_its_leader_exits(
    evaluator, monkeypatch, tmp_path
):
    upstream_root, manifest_path, results_dir = _external_evaluation_inputs(
        tmp_path, monkeypatch
    )
    leader_pid_path = tmp_path / "leader.pid"
    child_pid_path = tmp_path / "child.pid"
    survived_marker = tmp_path / "descendant-survived"
    candidate = tmp_path / "leader_exits.py"
    _write_candidate(
        candidate,
        f"""
import os
import subprocess
import sys
from pathlib import Path

child = subprocess.Popen([
    sys.executable,
    "-c",
    "import signal, time; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(3.0); Path({str(survived_marker)!r}).write_text('survived', encoding='utf-8')",
])
Path({str(leader_pid_path)!r}).write_text(str(os.getpid()), encoding="utf-8")
Path({str(child_pid_path)!r}).write_text(str(child.pid), encoding="utf-8")
print("leader exited", flush=True)
""",
    )

    started_at = time.monotonic()
    metrics, correct, error = evaluator.evaluate_candidate(
        candidate,
        results_dir,
        upstream_root,
        timeout_seconds=0.7,
        manifest_path=manifest_path,
    )
    elapsed = time.monotonic() - started_at

    assert elapsed < 2.25
    assert leader_pid_path.is_file()
    assert child_pid_path.is_file()
    with pytest.raises(ProcessLookupError):
        os.kill(int(leader_pid_path.read_text(encoding="utf-8")), 0)
    with pytest.raises(ProcessLookupError):
        os.kill(int(child_pid_path.read_text(encoding="utf-8")), 0)
    assert survived_marker.exists() is False
    assert correct is False
    assert "timed out" in error
    assert metrics["combined_score"] == 0.0
    assert _read_artifacts(results_dir)[1]["correct"] is False
    assert "leader exited" in (results_dir / "candidate_stdout.log").read_text(
        encoding="utf-8"
    )


def test_non_utf8_output_is_preserved_and_does_not_mask_protected_mutation(
    evaluator, monkeypatch, tmp_path
):
    upstream_root, manifest_path, results_dir = _external_evaluation_inputs(
        tmp_path, monkeypatch
    )
    candidate = tmp_path / "non_utf8.py"
    _write_candidate(
        candidate,
        """
import sys
from pathlib import Path

sys.stdout.buffer.write(b"prefix\\xffsuffix\\n")
sys.stdout.flush()
print("stderr context", file=sys.stderr)
Path("prepare.py").write_text("changed\\n", encoding="utf-8")
""",
    )

    metrics, correct, error = evaluator.evaluate_candidate(
        candidate,
        results_dir,
        upstream_root,
        timeout_seconds=1.0,
        manifest_path=manifest_path,
    )

    assert correct is False
    assert "protected" in error.lower()
    assert "unicodedecodeerror" not in error.lower()
    assert metrics["combined_score"] == 0.0
    assert _read_artifacts(results_dir)[1]["correct"] is False
    assert (results_dir / "candidate_stdout.log").read_text(encoding="utf-8") == (
        "prefix\ufffdsuffix\n"
    )
    assert "stderr context" in (results_dir / "candidate_stderr.log").read_text(
        encoding="utf-8"
    )


def test_protected_file_mutation_is_incorrect_even_when_candidate_exits_zero(
    evaluator, monkeypatch, tmp_path
):
    upstream_root, manifest_path, results_dir = _external_evaluation_inputs(
        tmp_path, monkeypatch
    )
    candidate = tmp_path / "mutates_protected.py"
    _write_candidate(
        candidate,
        """
from pathlib import Path
import sys
Path("prepare.py").write_text("changed\\n", encoding="utf-8")
print("---")
print("val_bpb: 0.75")
raise SystemExit(7)
""",
    )

    metrics, correct, error = evaluator.evaluate_candidate(
        candidate,
        results_dir,
        upstream_root,
        timeout_seconds=1.0,
        manifest_path=manifest_path,
    )

    assert correct is False
    assert "protected" in error.lower()
    assert metrics["combined_score"] == 0.0
    assert _read_artifacts(results_dir)[1]["correct"] is False
    assert "val_bpb: 0.75" in (results_dir / "candidate_stdout.log").read_text(
        encoding="utf-8"
    )


def test_missing_checkout_writes_incorrect_json_artifacts_and_empty_logs(
    evaluator, monkeypatch, tmp_path
):
    results_dir = tmp_path / "results"
    missing_root = tmp_path / "missing"
    manifest_path = tmp_path / "manifest.json"
    candidate = tmp_path / "candidate.py"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    _write_candidate(candidate, 'raise AssertionError("must not execute")\n')
    _write_manifest(manifest_path, _manifest({"prepare.py": "a" * 64}))

    metrics, correct, error = evaluator.evaluate_candidate(
        candidate,
        results_dir,
        missing_root,
        timeout_seconds=1.0,
        manifest_path=manifest_path,
    )

    artifact_metrics, artifact_correct = _read_artifacts(results_dir)
    assert correct is False
    assert "does not exist" in error.lower()
    assert metrics == artifact_metrics
    assert metrics["combined_score"] == 0.0
    assert artifact_correct["correct"] is False
    assert (results_dir / "candidate_stdout.log").read_text(encoding="utf-8") == ""
    assert (results_dir / "candidate_stderr.log").read_text(encoding="utf-8") == ""
