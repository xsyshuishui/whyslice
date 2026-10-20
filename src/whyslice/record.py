"""``python -m whyslice.record`` -- run one pytest test or a script under the recorder.

    python -m whyslice.record --out run.pkl [--root DIR] [--limit N] [--stdin FILE] -- pytest tests/t.py::test_x
    python -m whyslice.record --out run.pkl -- prog.py arg1 arg2
    python -m whyslice.record --out run.pkl -- -m package.module arg1

Only files under ``--root`` (default: the working directory) are traced, never
site-packages.  The trace is pickled to ``--out``; the exit status is the
target's.
"""

from __future__ import annotations

import argparse
import os
import pickle
import runpy
import sys
import time
from typing import Any, List, Optional, Tuple

from .recorder import Recorder


class _PytestProbe:
    """pytest plugin that keeps the exception object of every failing phase."""

    def __init__(self) -> None:
        self.failures: List[Tuple[str, str, BaseException, str]] = []

    def pytest_runtest_makereport(self, item: Any, call: Any) -> None:
        excinfo = call.excinfo
        if excinfo is None or excinfo.typename in ("Skipped", "XFailed"):
            return
        self.failures.append((item.nodeid, call.when, excinfo.value, excinfo.exconly()))


def _exc_text(exc: BaseException) -> str:
    msg = str(exc)
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m whyslice.record")
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--limit", type=int, default=300_000)
    ap.add_argument("--stdin", help="file to feed to the program's standard input")
    ap.add_argument("target", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    target = args.target[1:] if args.target[:1] == ["--"] else args.target
    if not target:
        ap.error("nothing to run")

    rec = Recorder(args.root, args.limit)
    failures: List[dict] = []
    exit_code: Any = 0
    started = time.time()
    if args.stdin:
        sys.stdin = open(args.stdin, encoding="utf-8")

    pytest_args: Optional[List[str]] = None
    if target[0] in ("pytest", "py.test"):
        pytest_args = target[1:]
    elif target[:2] == ["-m", "pytest"]:
        pytest_args = target[2:]

    if pytest_args is not None:
        import pytest

        probe = _PytestProbe()
        rec.start()
        try:
            exit_code = pytest.main(["-p", "no:cacheprovider", *pytest_args], plugins=[probe])
        finally:
            rec.stop()
        for nodeid, when, value, text in probe.failures:
            failures.append({"test": nodeid, "phase": when, "exc": text,
                             "origin": rec.origin_of(value), "raised_at": rec.raised_at_of(value)})
        exit_code = int(exit_code)
    else:
        if target[0] == "-m":
            module, rest = target[1], target[2:]
            sys.argv = [module, *rest]
            runner = lambda: runpy.run_module(module, run_name="__main__", alter_sys=True)  # noqa: E731
        else:
            script = os.path.abspath(target[0])
            sys.argv = [script, *target[1:]]
            sys.path.insert(0, os.path.dirname(script))
            runner = lambda: runpy.run_path(script, run_name="__main__")  # noqa: E731
        rec.start()
        try:
            runner()
        except SystemExit as exc:
            exit_code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        except BaseException as exc:  # noqa: BLE001 - we report whatever the program raised
            rec.stop()
            failures.append({"test": "<program>", "phase": "call", "exc": _exc_text(exc),
                             "origin": rec.origin_of(exc), "raised_at": rec.raised_at_of(exc)})
            exit_code = 1
        finally:
            rec.stop()
        sys.stdout.flush()

    data = rec.snapshot()
    data.update({
        "command": target,
        "cwd": os.getcwd(),
        "elapsed": round(time.time() - started, 3),
        "exit_code": exit_code,
        "failures": failures,
    })
    with open(args.out, "wb") as fh:
        pickle.dump(data, fh, protocol=4)
    return exit_code if isinstance(exit_code, int) else 1


if __name__ == "__main__":
    sys.exit(main())
