"""Command line front end (used by the Agent Skill and for evaluation).

    whyslice run [--python PY] [--timeout S] -- pytest tests/test_x.py::test_y
    whyslice why FILE:LINE [VARIABLE] [-n OCCURRENCE]
    whyslice lines FILE:LINE [VARIABLE]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .runner import project_root, record, report, trace_path
from .trace import QueryError, Trace


def _load(cwd: str) -> Trace:
    path = trace_path(project_root(cwd))
    if not os.path.exists(path):
        raise QueryError("no recording yet: run `whyslice run -- <failing test command>` first")
    return Trace.load(path)


def _loc(text: str) -> tuple:
    file, _, line = text.rpartition(":")
    if not file or not line.isdigit():
        raise QueryError(f"expected FILE:LINE, got {text!r}")
    return file, int(line)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="whyslice", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="record one failing test/program and explain the failure")
    run.add_argument("--python", default="")
    run.add_argument("--timeout", type=int, default=120)
    run.add_argument("command", nargs=argparse.REMAINDER)
    for name in ("why", "lines"):
        q = sub.add_parser(name)
        q.add_argument("location", help="FILE:LINE")
        q.add_argument("variable", nargs="?", default="")
        q.add_argument("-n", "--occurrence", type=int, default=0,
                       help="which execution of the line (1 = first, default = last)")
    args = ap.parse_args(argv)
    cwd = os.getcwd()
    try:
        if args.cmd == "run":
            cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
            if not cmd:
                ap.error("give the test command after --")
            import shlex
            print(report(record(shlex.join(cmd), cwd, args.python, args.timeout)))
            return 0
        trace = _load(cwd)
        file, line = _loc(args.location)
        iid = trace.find(file, line, args.occurrence if args.occurrence > 0 else -1)
        if args.cmd == "why":
            print(trace.explain(iid, args.variable or None))
        else:
            roots = trace.seeds(iid, args.variable or None)
            for f, ln in sorted(trace.lines_of(trace.backward(roots) | {iid})):
                print(f"{trace.short(f)}:{ln}")
        return 0
    except QueryError as exc:
        print(f"whyslice: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
