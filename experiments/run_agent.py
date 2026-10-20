"""E2: controlled experiment -- does WhySlice help a coding agent fix seeded bugs?

Arms (nested, same prompt, same model):
  A0  baseline      Claude Code with its built-in tools only
  A1  pull          + WhySlice MCP server and skill; the agent decides whether to call it
  A2  push          + A1, and a PostToolUse hook that records a failing pytest run
                    automatically and injects the causal tree

Each run copies the task into a fresh directory with its own venv, runs
``claude -p`` there, then the harness (not the agent) judges the result:
the originally failing test passes, the whole suite passes, and no test file
was modified.

    python experiments/run_agent.py --tasks experiments/bench/tasks --arms A0 A1 A2 --seeds 1 --jobs 4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
VENV_PY = os.path.join(REPO, ".venv", "bin", "python")
MCP_BIN = os.path.join(REPO, ".venv", "bin", "whyslice-mcp")
SKILL_DIR = os.path.join(REPO, "plugin", "skills", "whyslice")
HOOK = os.path.join(REPO, "plugin", "hooks", "push_failure.py")
CLAUDE = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
# agent working copies live outside the developer's project tree so no ancestor CLAUDE.md can reach the session
WORK_ROOT = "/private/tmp/whyslice-e2"

PROMPT = """\
The test `{test}` fails in this repository. Find the root cause in the library's source \
code and fix it with a minimal change. Do not modify any test files.
Run tests with the project's virtualenv, e.g. `.venv/bin/python -m pytest {test}`.
When you are done, that test must pass and the full suite (`.venv/bin/python -m pytest -q`) \
must still pass."""

# A1E: the on-demand arm, told to call the tool right after the first failure
EARLY_CALL = ("\nWhen a test fails, first call the whyslice MCP tool `record_failure` on that test and read "
              "its report before you open any source files.")

# A2S / A2T: the push arm with only part of the report (ablation)
PUSH_SECTIONS = {"A2": "all", "A2S": "suspects", "A2T": "tree"}


def load_task(task_dir: str) -> dict:
    """task.json sits next to the task directory (``<id>.task.json``) so the agent cannot read it."""
    for path in (task_dir.rstrip("/") + ".task.json", os.path.join(task_dir, "task.json")):
        if os.path.exists(path):
            return json.load(open(path))
    raise FileNotFoundError(f"no task.json for {task_dir}")


def arm_config(arm: str, run_dir: str) -> List[str]:
    """Extra claude CLI arguments (and files in run_dir) for one arm."""
    mcp: Dict[str, dict] = {}
    # Bash runs inside the macOS sandbox (writes confined to the run directory), so it
    # needs no approval; file edits are accepted by --permission-mode acceptEdits.
    settings: Dict[str, object] = {"sandbox": {"enabled": True, "autoAllowBashIfSandboxed": True,
                                               "allowUnsandboxedCommands": False, "failIfUnavailable": True}}
    if arm[:2] in ("A1", "A2"):
        mcp["whyslice"] = {"command": MCP_BIN, "args": []}
        dst = os.path.join(run_dir, ".claude", "skills", "whyslice")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(SKILL_DIR, dst)
    if arm in PUSH_SECTIONS:
        # a failing pytest exits non-zero, which Claude Code reports as PostToolUseFailure
        command = f"WHYSLICE_PUSH_SECTIONS={PUSH_SECTIONS[arm]} {VENV_PY} {HOOK}"
        hook = [{"matcher": "Bash", "hooks": [{"type": "command", "command": command, "timeout": 120}]}]
        settings["hooks"] = {"PostToolUse": hook, "PostToolUseFailure": hook}
    exclude = os.path.join(run_dir, ".git", "info", "exclude")
    if os.path.isdir(os.path.dirname(exclude)):
        with open(exclude, "a") as fh:
            fh.write("\n.claude/\n.venv/\n")
    mcp_file = os.path.join(run_dir, "..", "mcp.json")
    settings_file = os.path.join(run_dir, "..", "settings.json")
    json.dump({"mcpServers": mcp}, open(mcp_file, "w"))
    json.dump(settings, open(settings_file, "w"))
    allowed = ["mcp__whyslice__record_failure", "mcp__whyslice__why"] if mcp else []
    return ["--strict-mcp-config", "--mcp-config", os.path.abspath(mcp_file),
            "--settings", os.path.abspath(settings_file), "--setting-sources", "project,local",
            "--tools", "Bash,Read,Edit,Write,Glob,Grep", "--max-turns", "60",
            "--permission-mode", "acceptEdits", *(["--allowedTools", *allowed] if allowed else [])]


def prepare(task_dir: str, run_root: str) -> str:
    work = os.path.join(run_root, "work")
    if os.path.exists(run_root):
        shutil.rmtree(run_root)
    os.makedirs(run_root)
    shutil.copytree(task_dir, work, symlinks=True, ignore=shutil.ignore_patterns(".venv", "__pycache__"))
    task = load_task(task_dir)
    setup = task["setup"].replace("{bench}", os.path.join(HERE, "bench"))
    subprocess.run(setup, shell=True, cwd=work, check=True, capture_output=True, timeout=600)
    os.remove(os.path.join(work, "task.json")) if os.path.exists(os.path.join(work, "task.json")) else None
    return work


def judge(work: str, task: dict) -> dict:
    py = os.path.join(work, ".venv", "bin", "python")
    def run(args: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run([py, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args], cwd=work,
                              capture_output=True, text=True, timeout=600)
    target = run([task["failing_test"]])
    suite = run([])
    diff = subprocess.run(["git", "diff", "--name-only"], cwd=work, capture_output=True, text=True).stdout.split()
    touched_tests = [f for f in diff if "test" in os.path.basename(f) or f.startswith("tests/")]
    fault_fixed = False
    patch = subprocess.run(["git", "diff", "-U0", "--", task["fault_file"]], cwd=work, capture_output=True,
                           text=True).stdout
    for line in patch.splitlines():
        if line.startswith("@@"):
            old = line.split()[1]  # -start,count
            start, _, count = old[1:].partition(",")
            lo = int(start)
            hi = lo + (int(count) if count else 1) - 1
            if lo <= task["fault_line"] <= max(lo, hi):
                fault_fixed = True
    return {"target_pass": target.returncode == 0, "suite_pass": suite.returncode == 0,
            "touched_tests": touched_tests, "edited_files": diff, "edited_fault_line": fault_fixed,
            "success": target.returncode == 0 and suite.returncode == 0 and not touched_tests,
            "suite_tail": suite.stdout.strip().splitlines()[-1:] if suite.stdout else []}


def parse_transcript(path: str) -> dict:
    tools: Dict[str, int] = {}
    turns = 0
    result: Optional[dict] = None
    first_whyslice = None
    hook_pushes = 0
    with open(path) as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("type") == "assistant":
                turns += 1
                for block in ev.get("message", {}).get("content", []):
                    if block.get("type") == "tool_use":
                        name = block.get("name", "?")
                        tools[name] = tools.get(name, 0) + 1
                        if "whyslice" in name and first_whyslice is None:
                            first_whyslice = turns
            elif ev.get("type") == "user":
                text = json.dumps(ev.get("message", {}))
                if "WhySlice recorded the failing test automatically" in text:
                    hook_pushes += 1
            elif ev.get("type") == "system" and "WhySlice recorded the failing test automatically" in json.dumps(ev):
                hook_pushes += 1
            elif ev.get("type") == "result":
                result = ev
    usage = (result or {}).get("usage", {})
    return {"turns": turns, "tools": tools, "first_whyslice_turn": first_whyslice, "hook_pushes": hook_pushes,
            "cost_usd": (result or {}).get("total_cost_usd"), "num_turns": (result or {}).get("num_turns"),
            "duration_ms": (result or {}).get("duration_ms"), "is_error": (result or {}).get("is_error"),
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
            "cache_read": usage.get("cache_read_input_tokens"), "cache_write": usage.get("cache_creation_input_tokens")}


def run_one(task_dir: str, arm: str, seed: int, out_root: str, model: str, budget: float,
            timeout: int, extra: List[str], provider_env: Optional[Dict[str, str]] = None) -> dict:
    task_dir, out_root = os.path.abspath(task_dir), os.path.abspath(out_root)
    task = load_task(task_dir)
    run_root = os.path.join(out_root, task["id"], f"{arm}-s{seed}")
    t0 = time.time()
    # namespaced by the output directory so concurrent batches (e.g. two models) never share a copy
    scratch = os.path.join(WORK_ROOT, os.path.basename(out_root), task["id"], f"{arm}-s{seed}")
    work = prepare(task_dir, scratch)
    os.makedirs(run_root, exist_ok=True)
    prompt = PROMPT.format(test=task["failing_test"]) + (EARLY_CALL if arm == "A1E" else "")
    args = [CLAUDE, "-p", prompt, "--model", model,
            "--output-format", "stream-json", "--verbose", "--max-budget-usd", str(budget),
            "--no-session-persistence", "--include-hook-events", *arm_config(arm, work), *extra]
    transcript = os.path.join(run_root, "transcript.jsonl")
    hook_log = os.path.join(run_root, "hook_log.jsonl")
    if os.path.exists(hook_log):
        os.remove(hook_log)
    env = {k: v for k, v in os.environ.items()
           if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")}
    env.update(WHYSLICE_HOOK_LOG=hook_log, CLAUDE_CODE_DISABLE_CLAUDE_MDS="1",
               CLAUDE_CODE_DISABLE_AUTO_MEMORY="1", CLAUDE_CODE_DISABLE_BUNDLED_SKILLS="1",
               CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS="0")
    env.update(provider_env or {})
    with open(transcript, "w") as fh:
        try:
            proc = subprocess.run(args, cwd=work, stdout=fh, stderr=subprocess.PIPE, text=True,
                                  timeout=timeout, env=env)
            status = f"exit {proc.returncode}"
            err = proc.stderr[-2000:]
        except subprocess.TimeoutExpired:
            status, err = "timeout", ""
    metrics = parse_transcript(transcript)
    if os.path.exists(hook_log):
        metrics["hook_pushes"] = sum(1 for _ in open(hook_log))
    verdict = judge(work, task)
    row = {"task": task["id"], "library": task.get("library"), "arm": arm, "seed": seed, "model": model,
           "status": status, "stderr_tail": err[-500:], "wall_s": round(time.time() - t0, 1),
           **metrics, **verdict}
    patch = subprocess.run(["git", "diff"], cwd=work, capture_output=True, text=True).stdout
    open(os.path.join(run_root, "agent.patch"), "w").write(patch)
    json.dump(row, open(os.path.join(run_root, "result.json"), "w"), indent=1)
    shutil.rmtree(scratch, ignore_errors=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.path.join(HERE, "bench", "tasks"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--arms", nargs="+", default=["A0", "A1", "A2"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--out", default=os.path.join(HERE, "runs"))
    ap.add_argument("--skip-existing", action="store_true", help="skip runs that already have a result.json")
    ap.add_argument("--base-url", help="Anthropic-compatible endpoint for a non-Claude model")
    ap.add_argument("--key-env", help="environment variable holding that endpoint's key (e.g. via `ak run`)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()
    provider_env: Dict[str, str] = {}
    if args.base_url:
        # every model slot Claude Code may use points at the same model
        provider_env = {"ANTHROPIC_BASE_URL": args.base_url, "ANTHROPIC_AUTH_TOKEN": os.environ[args.key_env],
                        "ANTHROPIC_DEFAULT_HAIKU_MODEL": args.model, "ANTHROPIC_DEFAULT_SONNET_MODEL": args.model,
                        "ANTHROPIC_DEFAULT_OPUS_MODEL": args.model, "ANTHROPIC_SMALL_FAST_MODEL": args.model,
                        "API_TIMEOUT_MS": "600000", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
    task_dirs = sorted(os.path.join(args.tasks, d) for d in os.listdir(args.tasks)
                       if os.path.isdir(os.path.join(args.tasks, d))
                       and (os.path.isfile(os.path.join(args.tasks, d + ".task.json"))
                            or os.path.isfile(os.path.join(args.tasks, d, "task.json"))))
    if args.only:
        task_dirs = [d for d in task_dirs if os.path.basename(d) in args.only]
    jobs = [(d, arm, s) for s in range(args.seed_start, args.seed_start + args.seeds) for d in task_dirs
            for arm in args.arms]
    if args.skip_existing:
        jobs = [(d, arm, s) for d, arm, s in jobs
                if not os.path.exists(os.path.join(args.out, load_task(d)["id"], f"{arm}-s{s}", "result.json"))]
    print(f"{len(jobs)} runs", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, d, arm, s, args.out, args.model, args.budget, args.timeout, args.extra,
                          provider_env):
                (os.path.basename(d), arm, s) for d, arm, s in jobs}
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                row = fut.result()
            except Exception as exc:
                row = {"task": key[0], "arm": key[1], "seed": key[2], "error": repr(exc)}
            rows.append(row)
            print(json.dumps({k: row.get(k) for k in ("task", "arm", "seed", "success", "turns", "cost_usd",
                                                         "tools", "hook_pushes", "status", "error")}), flush=True)
    with open(os.path.join(args.out, f"results-{int(time.time())}.jsonl"), "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
