"""PostToolUse / PostToolUseFailure hook: when a Bash call runs pytest and a test fails, record that
test with WhySlice and hand the causal tree to the agent as extra context.

Claude Code passes the hook event as JSON on stdin; printing
``{"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": ...}}``
adds the text to the conversation.  At most two pushes per test and four per
session, so a stubborn failure does not flood the context.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from whyslice.runner import parse_command, record, report  # noqa: E402

FAILED = re.compile(r"^(?:FAILED|ERROR) (\S+?::\S+?)(?: - |\s*$)", re.M)
PER_TEST, PER_SESSION = 2, 4


def _state_path(session: str) -> str:
    d = os.path.join(tempfile.gettempdir(), "whyslice-hook")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, hashlib.sha1(session.encode()).hexdigest()[:16] + ".json")


def _output(event: dict) -> str:
    """Tool output: PostToolUse carries ``tool_response``; PostToolUseFailure
    (a command that exited non-zero, e.g. pytest with a failing test) carries ``error``."""
    parts = [str(event.get("error") or "")]
    resp = event.get("tool_response") or {}
    if isinstance(resp, str):
        parts.append(resp)
    elif isinstance(resp, dict):
        parts += [str(resp.get(k) or "") for k in ("stdout", "stderr", "output")]
    return "\n".join(parts)


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except ValueError:
        return 0
    if event.get("tool_name") != "Bash":
        return 0
    command = (event.get("tool_input") or {}).get("command", "")
    if "pytest" not in command:
        return 0
    failed = FAILED.findall(_output(event))
    if not failed:
        return 0
    nodeid = failed[0]
    session = event.get("session_id") or "default"
    state_file = _state_path(session)
    try:
        state = json.load(open(state_file))
    except (OSError, ValueError):
        state = {"total": 0, "tests": {}}
    if state["total"] >= PER_SESSION or state["tests"].get(nodeid, 0) >= PER_TEST:
        return 0
    cwd = event.get("cwd") or os.getcwd()
    interp, _target, _stdin = parse_command(command.split("&&")[-1].strip())
    python = ""
    if interp and os.sep in interp:
        python = interp if os.path.isabs(interp) else os.path.join(cwd, interp)
    rec = record(f"pytest -q -p no:cacheprovider {nodeid}", cwd=cwd, python=python, timeout=90)
    if rec.trace is None:
        return 0
    state["total"] += 1
    state["tests"][nodeid] = state["tests"].get(nodeid, 0) + 1
    json.dump(state, open(state_file, "w"))
    # WHYSLICE_PUSH_SECTIONS=suspects|tree pushes only part of the report (experiment ablations)
    sections = os.environ.get("WHYSLICE_PUSH_SECTIONS", "all")
    intro = ("Suspicious lines for the failure, ranked by how rarely passing tests execute them"
             if sections == "suspects" else
             "Causal chain of the failure (each indented line caused the line above it; "
             "values are from this run)")
    text = ("WhySlice recorded the failing test automatically. " + intro + ":\n\n"
            + report(rec, max_nodes=20, sections=sections)
            + "\n\nYou can ask follow-up questions with the whyslice `why` tool.")
    log = os.environ.get("WHYSLICE_HOOK_LOG")
    if log:  # experiment bookkeeping, written outside the agent's working tree
        with open(log, "a") as fh:
            fh.write(json.dumps({"test": nodeid, "chars": len(text), "text": text}) + "\n")
    event_name = event.get("hook_event_name") or "PostToolUse"
    print(json.dumps({"hookSpecificOutput": {"hookEventName": event_name, "additionalContext": text[:9500]}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
