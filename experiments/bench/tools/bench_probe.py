"""pytest plugin used only while generating the benchmark (never shipped in a task).

Load with ``PYTHONPATH=<bench>/tools python -m pytest -p bench_probe ...``.
Everything is driven by environment variables:

BENCH_REPORT        path of a JSON-lines file; failures, collection errors, a
                    session record and probe records are appended to it
BENCH_TEST_TIMEOUT  per-test timeout in seconds for the call phase (0 = off)
BENCH_LIB           module name whose ``__file__`` is recorded at session end
BENCH_PROBE_FILE    absolute path of the mutated source file   } depth probe,
BENCH_PROBE_LINES   "lo-hi" line range of the mutated expression } see below

Depth probe: while a test's call phase runs, a ``sys.settrace`` tracer watches
'line' events in BENCH_PROBE_FILE.  Whenever a line in lo..hi executes, it walks
``f_back`` from that frame to the test function's own frame and records the
number of named function frames walked (1 = the test function itself called the
function that holds the mutated line; 2 = one intermediate frame; ...;
<genexpr>/<lambda>/comprehension frames are not counted).  Executions that do
not have the test function on the stack (fixtures, subprocesses) are ignored.
This is a deliberately small, independent probe: it shares no code with any
slicing tool.
"""

import inspect
import json
import os
import signal
import sys

import pytest

_REPORT = os.environ.get("BENCH_REPORT")
_TIMEOUT = float(os.environ.get("BENCH_TEST_TIMEOUT", "0") or 0)
_LIB = os.environ.get("BENCH_LIB")
_PFILE = os.environ.get("BENCH_PROBE_FILE")
_PLINES = os.environ.get("BENCH_PROBE_LINES")


def _emit(rec):
    if _REPORT:
        with open(_REPORT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")


class BenchTimeout(BaseException):
    """BaseException so that library code doing `except Exception` cannot swallow it."""


def _on_alarm(signum, frame):
    raise BenchTimeout(f"bench per-test timeout ({_TIMEOUT:g}s)")


def _test_code(item):
    func = getattr(item, "obj", None)
    func = getattr(func, "__func__", func)
    try:
        func = inspect.unwrap(func)
    except Exception:
        pass
    return getattr(func, "__code__", None)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    use_alarm = _TIMEOUT > 0 and hasattr(signal, "setitimer")
    if use_alarm:
        old_handler = signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, _TIMEOUT)
    probe = None
    if _PFILE and _PLINES:
        probe = _Probe(_test_code(item))
        probe.start()
    try:
        yield
    finally:
        if probe is not None:
            probe.stop()
            _emit(probe.record(item.nodeid))
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


class _Probe:
    def __init__(self, test_code):
        lo, hi = _PLINES.split("-")
        self.lo, self.hi = int(lo), int(hi)
        self.target = os.path.realpath(_PFILE)
        self.test_code = test_code
        self.depths = []
        self.raw_depths = []
        self.best_chain = None
        self.outside = 0  # executions without the test frame on the stack
        self._cache = {}
        self._old = None

    def _local(self, frame, event, arg):
        if event == "line" and self.lo <= frame.f_lineno <= self.hi:
            codes = [frame.f_code]
            f = frame.f_back
            while f is not None and f.f_code is not self.test_code:
                codes.append(f.f_code)
                f = f.f_back
            if f is None:
                self.outside += 1
            else:
                # depth counts named function frames only: <genexpr>/<lambda>/comprehension
                # frames are not separate functions in the source (minimum 1).
                depth = max(1, sum(1 for c in codes if not c.co_name.startswith("<")))
                if not self.depths or depth < min(self.depths):
                    self.best_chain = [c.co_qualname for c in reversed(codes)]
                self.depths.append(depth)
                self.raw_depths.append(len(codes))
        return self._local

    def _global(self, frame, event, arg):
        fn = frame.f_code.co_filename
        hit = self._cache.get(fn)
        if hit is None:
            hit = self._cache[fn] = os.path.realpath(fn) == self.target
        return self._local if hit else None

    def start(self):
        self._old = sys.gettrace()
        sys.settrace(self._global)

    def stop(self):
        sys.settrace(self._old)

    def record(self, nodeid):
        return {
            "kind": "probe",
            "nodeid": nodeid,
            "test_code_found": self.test_code is not None,
            "hits": len(self.depths),
            "outside_hits": self.outside,
            "min_depth": min(self.depths) if self.depths else None,
            "max_depth": max(self.depths) if self.depths else None,
            "min_raw_depth": min(self.raw_depths) if self.raw_depths else None,
            "chain": self.best_chain,
        }


def pytest_runtest_logreport(report):
    if report.failed:
        _emit({"kind": "fail", "nodeid": report.nodeid, "when": report.when})


def pytest_collectreport(report):
    if report.failed:
        _emit({"kind": "collect_error", "nodeid": report.nodeid})


def pytest_sessionfinish(session, exitstatus):
    rec = {
        "kind": "session",
        "exitstatus": int(exitstatus),
        "collected": session.testscollected,
        "failed": session.testsfailed,
    }
    if _LIB:
        mod = sys.modules.get(_LIB)
        rec["lib_file"] = getattr(mod, "__file__", None)
    _emit(rec)
