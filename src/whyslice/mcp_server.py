"""MCP server: record once, ask *why* many times.

The dynamic dependence graph of the last recorded run is kept in memory, so
follow-up ``why`` questions do not re-run the program.
"""

from __future__ import annotations

import os
from typing import Optional

from mcp.server import MCPServer

from .runner import project_root, record, report, trace_path
from .trace import QueryError, Trace

INSTRUCTIONS = """\
WhySlice answers "why does this variable have this value?" for a failing Python run.
It records one execution of a failing test (or program) and builds its dynamic
dependence graph: which executed statements, with which runtime values, fed the
failing value -- across function calls -- and which branch decisions kept a
variable from being re-assigned. Use it when a test fails and the traceback does
not show where the wrong value came from: call record_failure once, read the causal
tree, then call why on any suspicious line/variable it shows. Recording is cheap
for a single test; never record a whole test suite."""

server = MCPServer("whyslice", instructions=INSTRUCTIONS)
_state: dict = {"trace": None, "path": None, "mtime": 0.0}


def _latest(cwd: str = "") -> Optional[Trace]:
    """The newest recording for this project: ours, or one made by the CLI or the hook."""
    path = trace_path(project_root(cwd or os.getcwd()))
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return _state["trace"]
    if _state["trace"] is None or _state["path"] != path or mtime > _state["mtime"]:
        _state.update(trace=Trace.load(path), path=path, mtime=mtime)
    return _state["trace"]


@server.tool()
def record_failure(command: str, cwd: str = "") -> str:
    """Run ONE failing test or program under WhySlice and explain why it failed.

    Returns a causal tree rooted at the failing assertion/exception: each indented
    line is a statement execution that caused the line above it, with the values it
    read and produced (e.g. `rate=0.1 from #37 pricing.py:8 ... rate = r`), including
    "`x` not re-assigned because (not taken) #39 ... if a > b:" lines that expose
    wrong branch decisions. Ends with the slice: the few executed lines that can
    influence the failure.

    command: the shell command that reproduces the failure, e.g.
        "pytest tests/test_cart.py::test_total"  or  "python prog.py < input.txt".
        Select a single test; WhySlice traces only files of this repository.
    cwd: directory to run in (default: the server's working directory).
    """
    rec = record(command, cwd or os.getcwd())
    if rec.trace is not None:
        _state.update(trace=rec.trace, path=rec.path, mtime=os.path.getmtime(rec.path))
    return report(rec)


@server.tool()
def why(location: str, variable: str = "", occurrence: int = 0) -> str:
    """Explain why `variable` had its value at `location` in the last recorded run.

    location: "path/to/file.py:LINE" (a line that executed in the recording).
    variable: a name or attribute read or assigned on that line, e.g. "total" or
        "self.items"; leave empty to explain the whole line.
    occurrence: which execution of the line, 1 = first; 0 = the last one.
    Returns the causal tree for that value with runtime values, plus its slice.
    Does not re-run the program; call record_failure again after editing code.
    """
    trace = _latest()
    if trace is None:
        return "No recording yet: call record_failure with the failing test command first."
    file, _, line = location.rpartition(":")
    if not file or not line.strip().isdigit():
        return f"location must look like path/to/file.py:123, got {location!r}"
    try:
        iid = trace.find(file, int(line), occurrence if occurrence > 0 else -1)
        return trace.explain(iid, variable or None)
    except QueryError as exc:
        return f"WhySlice: {exc}"


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
