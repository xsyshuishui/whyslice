"""Run a command under the recorder in the *target project's* interpreter.

Shared by the CLI, the MCP server and the push hook.  The recorder is
stdlib-only, so it is injected through ``PYTHONPATH`` instead of being
installed into the project.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .trace import Trace

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Recording:
    trace: Optional[Trace]
    path: str
    log: str            # tail of the program's own output
    error: str = ""     # why there is no trace, if there is none
    spectrum: Optional[dict] = None   # per-test coverage of the failing test's module


def project_root(cwd: str) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=cwd, capture_output=True,
                             text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return os.path.realpath(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return os.path.realpath(cwd)


def trace_path(root: str) -> str:
    digest = hashlib.sha1(root.encode()).hexdigest()[:12]
    d = os.path.join(tempfile.gettempdir(), "whyslice", digest)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "last.pkl")


def _shebang_python(executable: Optional[str]) -> Optional[str]:
    if not executable:
        return None
    try:
        with open(executable, "rb") as fh:
            first = fh.readline().decode(errors="replace").strip()
    except OSError:
        return None
    if first.startswith("#!"):
        cand = first[2:].strip().split()[0]
        if os.path.basename(cand).startswith("python") and os.path.exists(cand):
            return cand
    return None


def pick_python(cwd: str, interp: Optional[str], pytest_mode: bool) -> str:
    if interp and (os.sep in interp):
        path = interp if os.path.isabs(interp) else os.path.join(cwd, interp)
        if os.path.exists(path):
            return path
    d = os.path.realpath(cwd)
    for _ in range(4):
        for venv in (".venv", "venv", "env"):
            cand = os.path.join(d, venv, "bin", "python")
            if os.path.exists(cand):
                return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    if pytest_mode:
        cand = _shebang_python(shutil.which("pytest"))
        if cand:
            return cand
    if interp:
        found = shutil.which(interp)
        if found:
            return found
    return shutil.which("python3") or sys.executable


def parse_command(command: str) -> Tuple[Optional[str], List[str], Optional[str]]:
    """``"python -m pytest t.py::x < in.txt"`` -> (``python``, [``-m``, ``pytest``, ``t.py::x``], ``in.txt``)."""
    parts = shlex.split(command)
    stdin_file = None
    if "<" in parts:
        i = parts.index("<")
        if i + 1 < len(parts):
            stdin_file = parts[i + 1]
        parts = parts[:i] + parts[i + 2:]
    interp = None
    if parts and os.path.basename(parts[0]).startswith("python"):
        interp, parts = parts[0], parts[1:]
    if parts[:1] == ["uv"] and parts[1:2] == ["run"]:   # "uv run pytest ..." -> "pytest ..."
        parts = [p for p in parts[2:] if not p.startswith("--")]
    return interp, parts, stdin_file


def record(command: str, cwd: str = "", python: str = "", timeout: int = 120,
           limit: int = 300_000, out: str = "", root: str = "", spectrum_timeout: int = 90) -> Recording:
    cwd = os.path.realpath(cwd or os.getcwd())
    root = os.path.realpath(root) if root else project_root(cwd)
    interp, target, stdin_file = parse_command(command)
    if not target:
        return Recording(None, "", "", "empty command")
    pytest_mode = target[0] in ("pytest", "py.test") or target[:2] == ["-m", "pytest"]
    py = python or pick_python(cwd, interp, pytest_mode)
    out = out or trace_path(root)
    if os.path.exists(out):
        os.remove(out)
    env = os.environ.copy()
    env["PYTHONPATH"] = SRC_DIR + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    argv = [py, "-m", "whyslice.record", "--out", out, "--root", root, "--limit", str(limit)]
    if stdin_file:
        argv += ["--stdin", stdin_file if os.path.isabs(stdin_file) else os.path.join(cwd, stdin_file)]
    argv += ["--", *target]
    try:
        proc = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        log = (proc.stdout + proc.stderr)
    except subprocess.TimeoutExpired as exc:
        log = ((exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""))
        return Recording(None, out, _tail(log), f"timed out after {timeout} s (the program may hang)")
    if not os.path.exists(out):
        return Recording(None, out, _tail(log),
                         f"no trace was written (interpreter {py}); program output follows")
    spectrum = None
    node = next((a for a in target if "::" in a), None) if pytest_mode else None
    if node and spectrum_timeout > 0:
        module = node.split("::")[0]
        cov = out + ".spectrum.json"
        try:
            subprocess.run([py, "-m", "whyslice.spectrum", "--out", cov, "--root", root, "--", module],
                           cwd=cwd, env=env, capture_output=True, text=True, timeout=spectrum_timeout)
            if os.path.exists(cov):
                import json
                with open(cov) as fh:
                    spectrum = json.load(fh)
                os.remove(cov)
        except (OSError, subprocess.SubprocessError, ValueError):
            spectrum = None
    return Recording(Trace.load(out), out, _tail(log), spectrum=spectrum)


def _tail(text: str, n: int = 25) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-n:])


def report(rec: Recording, max_nodes: int = 30, sections: str = "all") -> str:
    """Overview, the failing line, suspicious lines, then the causal tree.

    ``sections`` keeps only part of the failure report for ablations: "suspects" drops the
    causal tree, "tree" drops the suspicious lines.
    """
    if rec.trace is None:
        return f"WhySlice could not record: {rec.error}\n{rec.log}"
    t = rec.trace
    parts = [t.overview()]
    if t.failure() is None and not t.data.get("outputs"):
        parts.append("The run raised no exception and printed nothing, so there is no failure to explain. "
                     "Use `why` on a specific file:line and variable.")
        parts.append("Program output (tail):\n" + rec.log)
    else:
        report_text = t.failure_report(max_nodes=max_nodes)
        sus = t.suspects(rec.spectrum) if rec.spectrum and sections != "tree" else ""
        head, _, rest = report_text.partition("\n")
        if sections == "suspects":
            report_text = head + ("\n\n" + sus if sus else "")
        elif sus:
            report_text = head + "\n\n" + sus + "\n\n" + rest
        parts.append(report_text)
    return "\n\n".join(parts)
