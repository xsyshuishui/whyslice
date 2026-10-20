"""``python -m whyslice.spectrum`` -- per-test line coverage of one pytest module.

Lighter than the dependence recorder: it only notes which project lines each
test executes.  The runner uses it to rank the lines of a failing run by how
specific they are to that run (Ochiai), which brings deep faults to the top
when the causal chain is too long to read.

    python -m whyslice.spectrum --out cov.json --root DIR -- tests/test_x.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Set, Tuple

_PKG_DIR = os.path.dirname(os.path.realpath(__file__))


class _Coverage:
    def __init__(self, root: str) -> None:
        self.root = os.path.realpath(root)
        self.ok: Dict[Any, Optional[str]] = {}
        self.current: Optional[str] = None
        self.lines: Dict[str, Set[Tuple[str, int]]] = {}
        self.outcomes: Dict[str, str] = {}

    def _want(self, code: Any) -> Optional[str]:
        hit = self.ok.get(code, False)
        if hit is not False:
            return hit
        path = os.path.realpath(code.co_filename) if code.co_filename else ""
        keep = (path.startswith(self.root + os.sep) and path.endswith(".py") and "site-packages" not in path
                and not path.startswith(_PKG_DIR + os.sep))
        self.ok[code] = path if keep else None
        return self.ok[code]

    def tracer(self, frame: Any, event: str, arg: Any) -> Any:
        if event != "call" or self.current is None:
            return None
        path = self._want(frame.f_code)
        if path is None:
            return None
        bucket = self.lines.setdefault(self.current, set())

        def local(frame: Any, event: str, arg: Any) -> Any:
            if event == "line":
                bucket.add((path, frame.f_lineno))
            return local

        return local

    # pytest hooks
    def pytest_runtest_setup(self, item: Any) -> None:
        self.current = item.nodeid

    def pytest_runtest_logreport(self, report: Any) -> None:
        if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
            self.outcomes[report.nodeid] = report.outcome

    def pytest_runtest_teardown(self, item: Any) -> None:
        self.current = None


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m whyslice.spectrum")
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("target", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    target = args.target[1:] if args.target[:1] == ["--"] else args.target
    import pytest

    cov = _Coverage(args.root)
    sys.settrace(cov.tracer)
    try:
        code = pytest.main(["-q", "-p", "no:cacheprovider", "--no-header", "-o", "console_output_style=classic",
                            *target], plugins=[cov])
    finally:
        sys.settrace(None)
    data = {"outcomes": cov.outcomes,
            "lines": {t: sorted([p, ln] for p, ln in s) for t, s in cov.lines.items()}}
    with open(args.out, "w") as fh:
        json.dump(data, fh)
    return int(code)


if __name__ == "__main__":
    sys.exit(main())
