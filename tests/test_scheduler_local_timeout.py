from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from shinka.launch import JobScheduler, LocalJobConfig


def test_local_scheduler_applies_configured_timeout_to_direct_run(
    tmp_path: Path,
) -> None:
    evaluator = tmp_path / "slow_evaluator.py"
    completion_marker = tmp_path / "completed"
    evaluator.write_text(
        "import time\n"
        "from pathlib import Path\n"
        "time.sleep(2.5)\n"
        f"Path({str(completion_marker)!r}).write_text('completed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    scheduler = JobScheduler(
        job_type="local",
        config=LocalJobConfig(
            eval_program_path=str(evaluator),
            python_executable=sys.executable,
            time="00:00:01",
        ),
    )

    started_at = time.monotonic()
    results, runtime_seconds = scheduler.run(
        str(tmp_path / "candidate.py"), str(tmp_path / "results")
    )
    elapsed = time.monotonic() - started_at

    assert elapsed < 2.25
    assert runtime_seconds < 2.25
    assert completion_marker.exists() is False
    assert results["correct"] == {"correct": False}


def test_local_scheduler_timeout_kills_descendant_in_a_new_session(
    tmp_path: Path,
) -> None:
    evaluator = tmp_path / "evaluator_with_descendant.py"
    child_pid_path = tmp_path / "child.pid"
    child_survived = tmp_path / "child-survived"
    evaluator.write_text(
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([\n"
        "    sys.executable, '-c',\n"
        f"    \"import signal, time; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(3); Path({str(child_survived)!r}).write_text('survived')\",\n"
        "], start_new_session=True)\n"
        f"Path({str(child_pid_path)!r}).write_text(str(child.pid))\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    scheduler = JobScheduler(
        job_type="local",
        config=LocalJobConfig(
            eval_program_path=str(evaluator),
            python_executable=sys.executable,
            time="00:00:01",
        ),
    )

    scheduler.run(str(tmp_path / "candidate.py"), str(tmp_path / "results"))

    assert child_pid_path.is_file()
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    try:
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        time.sleep(2.2)
        assert child_survived.exists() is False
    finally:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_local_scheduler_timeout_kills_descendant_spawned_during_termination(
    tmp_path: Path,
) -> None:
    evaluator = tmp_path / "evaluator_spawns_on_sigterm.py"
    child_pid_path = tmp_path / "signal-child.pid"
    evaluator.write_text(
        "import signal\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n"
        "def handle_sigterm(_signum, _frame):\n"
        "    child = subprocess.Popen([\n"
        "        sys.executable, '-c',\n"
        "        'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)',\n"
        "    ], start_new_session=True)\n"
        f"    Path({str(child_pid_path)!r}).write_text(str(child.pid))\n"
        "signal.signal(signal.SIGTERM, handle_sigterm)\n"
        "while True:\n"
        "    time.sleep(0.01)\n",
        encoding="utf-8",
    )
    scheduler = JobScheduler(
        job_type="local",
        config=LocalJobConfig(
            eval_program_path=str(evaluator),
            python_executable=sys.executable,
            time="00:00:01",
        ),
    )

    scheduler.run(str(tmp_path / "candidate.py"), str(tmp_path / "results"))

    assert child_pid_path.is_file()
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    try:
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_async_local_timeout_uses_the_same_process_tree_cleanup(
    tmp_path: Path,
) -> None:
    evaluator = tmp_path / "async_evaluator_with_descendant.py"
    child_pid_path = tmp_path / "async-child.pid"
    evaluator.write_text(
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([\n"
        "    sys.executable, '-c',\n"
        "    'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)',\n"
        "], start_new_session=True)\n"
        f"Path({str(child_pid_path)!r}).write_text(str(child.pid))\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    scheduler = JobScheduler(
        job_type="local",
        config=LocalJobConfig(
            eval_program_path=str(evaluator),
            python_executable=sys.executable,
            time="00:00:01",
        ),
    )
    process = scheduler.submit_async(
        str(tmp_path / "candidate.py"), str(tmp_path / "results")
    )
    deadline = time.monotonic() + 2.0
    while not child_pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert child_pid_path.is_file()
    job = SimpleNamespace(
        job_id=process,
        start_time=time.time() - 2.0,
        generation=1,
    )

    assert scheduler.check_job_status(job) is False

    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
    process.cleanup_logging()


def test_async_local_cancellation_kills_descendants(tmp_path: Path) -> None:
    evaluator = tmp_path / "cancel_evaluator_with_descendant.py"
    child_pid_path = tmp_path / "cancel-child.pid"
    evaluator.write_text(
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([\n"
        "    sys.executable, '-c',\n"
        "    'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)',\n"
        "], start_new_session=True)\n"
        f"Path({str(child_pid_path)!r}).write_text(str(child.pid))\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    scheduler = JobScheduler(
        job_type="local",
        config=LocalJobConfig(
            eval_program_path=str(evaluator),
            python_executable=sys.executable,
        ),
    )
    process = scheduler.submit_async(
        str(tmp_path / "candidate.py"), str(tmp_path / "results")
    )
    deadline = time.monotonic() + 2.0
    while not child_pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert child_pid_path.is_file()

    assert asyncio.run(scheduler.cancel_job_async(process)) is True

    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    try:
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.cleanup_logging()
