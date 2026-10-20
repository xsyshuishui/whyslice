#!/usr/bin/env python3
"""Select surviving mutants, build task directories, verify them.

    python build_tasks.py select  [--results DIR] [--exclude ID ...]   # -> mutants/selection.json
    python build_tasks.py build      # selection.json + mutants/<lib>/results.jsonl -> tasks/<id>/, <id>.task.json, <id>.gt.diff
    python build_tasks.py verify  [ID ...]               # -> tasks/verify.json

Selection (per library, 12 tasks, greedy and deterministic): only mutants that
passed filters a-e and whose failing node id has no blanks/quotes/backslashes; at
most 2 per source function and 1 per source line; at each step take the best
candidate by (depth >= 3, function not used yet, faulty function absent from the
failing test's traceback, operator used least so far, failing test not used yet,
seeded random tie-break).

A task directory is the upstream clone minus .git/.github/.tox/build/dist, the
changelog and the agent-instruction files (AGENTS.md/CLAUDE.md), with the
mutation applied, committed as ONE initial git commit. Ground truth lives next
to it (tasks/<id>.task.json, tasks/<id>.gt.diff), never inside it. No venv is
created in the task: task.json's `setup` builds one from the pinned
upstream/<lib>.requirements.txt, and the library is imported from the task root
(cwd), so a copied task tests its own copy of the source.
"""

import argparse
import difflib
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_mutants as mm  # noqa: E402

BENCH = mm.BENCH
TASKS = BENCH / "tasks"
PER_LIB = 12
SAFE_NODE = re.compile(r"[\w./:\[\]-]+")  # failing node ids without blanks, quotes or backslashes
EXCLUDE_NAMES = {".git", ".github", ".tox", "build", "dist", ".venv", "__pycache__", ".pytest_cache",
                 "CHANGELOG", "CHANGELOG.md", "CHANGELOG.rst", "CHANGES", "CHANGES.rst", "HISTORY",
                 "HISTORY.md", "HISTORY.rst", "AGENTS.md", "CLAUDE.md"}
GITIGNORE_EXTRA = "\n# benchmark task environment\n.venv/\n__pycache__/\n.pytest_cache/\n"
GIT_ENV = {"GIT_AUTHOR_NAME": "bench", "GIT_AUTHOR_EMAIL": "bench@example.invalid",
           "GIT_COMMITTER_NAME": "bench", "GIT_COMMITTER_EMAIL": "bench@example.invalid",
           "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+0000", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+0000"}
FIXED_MTIME = 1767225600  # 2026-01-01T00:00:00Z
COMMIT_MSG = "Initial commit\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"


def _git_head(lib):
    return subprocess.run(["git", "-C", str(mm.UPSTREAM / lib), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def _requirements(lib):
    return mm.UPSTREAM / f"{lib}.requirements.txt"


def setup_command(lib):
    """`{bench}` stands for this benchmark directory; runners substitute it (see expand_setup)."""
    return ("uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r "
            + shlex.quote("{bench}/" + str(_requirements(lib).relative_to(mm.UPSTREAM.parent))))


def expand_setup(cmd):
    return cmd.replace("{bench}", str(mm.UPSTREAM.parent))


def _node_cmd(node):
    return ".venv/bin/python -m pytest " + shlex.quote(node)


# --------------------------------------------------------------------------- select

def survivors(lib, rdir=mm.MUTANTS):
    out = []
    for line in open(Path(rdir) / lib / "results.jsonl", encoding="utf-8"):
        m = json.loads(line)
        if m["result"]["stage"] == "e":
            out.append(m)
    return out


def select(lib, exclude=(), rdir=mm.MUTANTS):
    pool = [m for m in survivors(lib, rdir) if m["id"] not in exclude]
    rng = random.Random(f"{mm.SEED}:select:{lib}")
    tiebreak = {m["id"]: rng.random() for m in pool}
    chosen, per_func, per_op, lines, tests = [], Counter(), Counter(), set(), set()
    while len(chosen) < PER_LIB:
        ok = [m for m in pool if m not in chosen and per_func[m["function"]] < 2
              and (m["file"], m["line"]) not in lines and SAFE_NODE.fullmatch(m["result"]["failing_test"])]
        if not ok:
            break

        def key(m):
            r = m["result"]
            return (r["depth"] >= 3, per_func[m["function"]] == 0, not r["func_in_tb"],
                    -per_op[m["operator"]], r["failing_test"] not in tests, tiebreak[m["id"]])

        best = max(ok, key=key)
        chosen.append(best)
        per_func[best["function"]] += 1
        per_op[best["operator"]] += 1
        lines.add((best["file"], best["line"]))
        tests.add(best["result"]["failing_test"])
    chosen.sort(key=lambda m: (m["file"], m["line"], m["col"]))
    return chosen


def cmd_select(args):
    sel = {}
    for lib in mm.LIBS:
        chosen = select(lib, set(args.exclude), args.results)
        sel[lib] = [m["id"] for m in chosen]
        print(lib, len(survivors(lib, args.results)), "survivors ->", len(chosen), "selected")
        for m in chosen:
            r = m["result"]
            print(f"  {m['id']} {m['file']}:{m['line']} {m['function']} {m['operator']} "
                  f"{m['original']!r}->{m['mutated']!r} fail={r['n_failing']} depth={r['depth']} "
                  f"func_in_tb={r['func_in_tb']} {r['failing_test']}")
    (mm.MUTANTS / "selection.json").write_text(json.dumps(
        {"results": str(Path(args.results).resolve().relative_to(BENCH)), "exclude": args.exclude,
         "selected": sel}, indent=1) + "\n")


# --------------------------------------------------------------------------- build

def _copy_repo(src, dst):
    def ignore(d, names):
        return {n for n in names if n in EXCLUDE_NAMES or n.endswith(".egg-info")}
    shutil.copytree(src, dst, symlinks=True, ignore=ignore)


def _git(cwd, *args):
    env = dict(os.environ, **GIT_ENV)
    return subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *args],
                          cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout


def build_task(lib, m, task_id):
    cfg = mm.LIBS[lib]
    src = mm.UPSTREAM / lib
    tdir = TASKS / task_id
    shutil.rmtree(tdir, ignore_errors=True)
    _copy_repo(src, tdir)
    fault = tdir / m["file"]
    original = (src / m["file"]).read_text(encoding="utf-8")
    mutated = mm._apply(original, m)
    fault.write_text(mutated, encoding="utf-8")
    gi = tdir / ".gitignore"
    gi.write_text((gi.read_text(encoding="utf-8") if gi.exists() else "") + GITIGNORE_EXTRA, encoding="utf-8")
    # one fixed mtime for every file and directory, so `ls -t` / `find -newer` cannot
    # single out the mutated file in a copy that preserves timestamps
    for p in [tdir, *tdir.rglob("*")]:
        os.utime(p, (FIXED_MTIME, FIXED_MTIME), follow_symlinks=False)
    _git(tdir, "init", "-q", "--template=", "-b", "main")
    _git(tdir, "add", "-A", "-f")  # -f: tabulate's own .gitignore ignores ".*" (e.g. .gitignore itself)
    _git(tdir, "commit", "-q", "-m", COMMIT_MSG)
    diff = "".join(difflib.unified_diff(mutated.splitlines(keepends=True), original.splitlines(keepends=True),
                                        fromfile="a/" + m["file"], tofile="b/" + m["file"]))
    (TASKS / f"{task_id}.gt.diff").write_text(diff, encoding="utf-8")
    r = m["result"]
    task = {
        "id": task_id,
        "library": lib,
        "upstream_tag": cfg["tag"],
        "upstream_commit": _git_head(lib),
        "failing_test": r["failing_test"],
        "all_failing_tests": r["failing"],
        "test_command": _node_cmd(r["failing_test"]),
        "full_suite_command": ".venv/bin/python -m pytest " + cfg["tests"],
        "setup": setup_command(lib),
        "fault_file": m["file"],
        "fault_line": m["line"],
        "fault_function": m["function"],
        "original_token": m["original"],
        "mutated_token": m["mutated"],
        "operator": m["operator"],
        "depth": r["depth"],
        "fault_func_in_traceback": r["func_in_tb"],
        "mutant_id": m["id"],
    }
    (TASKS / f"{task_id}.task.json").write_text(json.dumps(task, indent=1) + "\n", encoding="utf-8")
    return task


def cmd_build(args):
    sel = json.loads((mm.MUTANTS / "selection.json").read_text())["selected"]
    TASKS.mkdir(exist_ok=True)
    for old in TASKS.iterdir():  # rebuild from scratch
        shutil.rmtree(old) if old.is_dir() else old.unlink()
    index = []
    for lib, ids in sel.items():
        res = {json.loads(l)["id"]: json.loads(l) for l in open(mm.MUTANTS / lib / "results.jsonl")}
        for n, mid in enumerate(ids, 1):
            task = build_task(lib, res[mid], f"{lib}-{n:02d}")
            index.append(task)
            print(task["id"], mid, task["fault_file"], task["fault_line"], task["operator"])
    (TASKS / "index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- verify

def _sh(cmd, cwd, timeout=600, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.update(env_extra or {})
    p = subprocess.run(["bash", "-c", cmd], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def verify_task(task):
    tid, lib = task["id"], task["library"]
    pkg = mm.LIBS[lib]["pkg"]
    scratch = mm.WORK / "verify" / tid
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TASKS / tid, scratch, symlinks=True)
    v = {"id": tid}
    rc, out = _sh(expand_setup(task["setup"]), scratch)
    v["setup_ok"] = rc == 0
    if rc:
        v["setup_out"] = out[-2000:]
        return v
    site = list((scratch / ".venv" / "lib").glob("python3.12/site-packages"))[0]
    v["lib_in_site_packages"] = sorted(p.name for p in site.iterdir() if pkg in p.name.lower())
    rc, out = _sh(f".venv/bin/python -c 'import {pkg}; print({pkg}.__file__)'", scratch)
    v["import_file"] = out.strip()
    v["import_ok"] = rc == 0 and out.strip().startswith(str(scratch) + "/")
    rc, out = _sh("git status --porcelain", scratch)
    v["git_clean"] = rc == 0 and out.strip() == ""
    rc, out = _sh(task["test_command"], scratch)
    v["failing_test_exit"] = rc
    v["failing_test_fails"] = rc == 1 and re.search(r"\b1 failed\b", out) is not None
    mut_line = (scratch / task["fault_file"]).read_text(encoding="utf-8").splitlines()[task["fault_line"] - 1]
    v["mutated_line_in_output"] = mut_line.strip() in out  # default --tb=auto, as the agent sees it
    # same node again with the probe plugin, only to read the module path seen inside pytest
    report = mm.WORK / "verify" / f"{tid}.report.jsonl"
    report.unlink(missing_ok=True)
    _sh(task["test_command"] + " -p bench_probe -q", scratch,
        env_extra={"PYTHONPATH": str(mm.TOOLS), "BENCH_REPORT": str(report), "BENCH_LIB": pkg})
    recs = mm._read_report(report)
    v["pytest_lib_file"] = next((r.get("lib_file") for r in recs if r["kind"] == "session"), None)
    v["pytest_lib_ok"] = str(v["pytest_lib_file"]).startswith(str(scratch) + "/")
    rc, out = _sh(task["full_suite_command"] + " -p bench_probe -q --tb=no", scratch,
                  env_extra={"PYTHONPATH": str(mm.TOOLS), "BENCH_REPORT": str(report)})
    recs = mm._read_report(report)
    fails = []
    for r in recs:
        if r["kind"] in ("fail", "collect_error") and r["nodeid"] not in fails:
            fails.append(r["nodeid"])
    v["mutant_suite_failing"] = fails
    v["mutant_failing_matches"] = fails == task["all_failing_tests"]
    rc, out = _sh(f"git apply {shlex.quote(str(TASKS / (tid + '.gt.diff')))}", scratch)
    v["gt_apply_ok"] = rc == 0
    restored = (scratch / task["fault_file"]).read_bytes()
    v["gt_restores_upstream"] = restored == (mm.UPSTREAM / lib / task["fault_file"]).read_bytes()
    for cache in scratch.rglob("__pycache__"):  # no chance of a same-second stale .pyc
        if ".venv" not in cache.parts:
            shutil.rmtree(cache, ignore_errors=True)
    rc, out = _sh(task["full_suite_command"], scratch)
    v["fixed_suite_exit"] = rc
    v["fixed_suite_tail"] = out.strip().splitlines()[-1] if out.strip() else ""
    v["ok"] = all([v["setup_ok"], v["import_ok"], v["pytest_lib_ok"], v["git_clean"], v["failing_test_fails"],
                   v["mutant_failing_matches"], v["gt_apply_ok"], v["gt_restores_upstream"], rc == 0,
                   not v["lib_in_site_packages"]])
    shutil.rmtree(scratch, ignore_errors=True)
    return v


def cmd_verify(args):
    index = json.loads((TASKS / "index.json").read_text())
    if args.ids:
        index = [t for t in index if t["id"] in args.ids]
    out_path = TASKS / "verify.json"
    prev = json.loads(out_path.read_text()) if out_path.exists() else {}
    for task in index:
        v = verify_task(task)
        prev[task["id"]] = v
        print(task["id"], "OK" if v.get("ok") else "FAIL",
              {k: v[k] for k in v if k not in ("id", "ok") and not k.endswith("_tail")}
              if not v.get("ok") else v.get("fixed_suite_tail"), flush=True)
    out_path.write_text(json.dumps(prev, indent=1) + "\n", encoding="utf-8")
    shutil.rmtree(mm.WORK / "verify", ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--exclude", nargs="*", default=[])
    s.add_argument("--results", default=str(mm.MUTANTS), help="directory holding <lib>/results.jsonl")
    s.set_defaults(func=cmd_select)
    b = sub.add_parser("build")
    b.set_defaults(func=cmd_build)
    v = sub.add_parser("verify")
    v.add_argument("ids", nargs="*")
    v.set_defaults(func=cmd_verify)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
