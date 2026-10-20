#!/usr/bin/env python3
"""Build quixbugs_tasks.json: ground-truth fault lines + per-test outcomes for QuixBugs (Python).

Run from whyslice/eval with the dedicated venv (system python3 is 3.9: no sys.monitoring, no pytest):
    .venv-quixbugs/bin/python build_quixbugs_tasks.py [--verify-isolated]
~13 min, mostly re-running the 17 timed-out tests with a 30 s budget; --verify-isolated also runs
all 278 node ids one at a time through test_cmd_template and compares outcomes (~3 more min).
Rules and caveats: QUIXBUGS_NOTES.md.
"""
import argparse
import ast
import difflib
import io
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import time
import tokenize
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIMEOUT = 5            # per-test budget for buggy runs and for test_cmd_template
CONFIRM_TIMEOUT = 30   # every timed-out test is re-run with this budget: slow vs. non-terminating
CORRECT_TIMEOUT = 60   # generous, so correct-version durations are measured, not truncated
VENV_PY_REL = "../../.venv-quixbugs/bin/python"   # as seen from the QuixBugs root
# -c /dev/null --rootdir=. --confcutdir=. : ignore pytest config and conftests of enclosing repos
# (whyslice/pyproject.toml otherwise becomes the rootdir and prefixes every node id with eval/data/...).
ISOLATE = ["-c", "/dev/null", "--rootdir=.", "--confcutdir=.", "-p", "no:cacheprovider"]
TIMEOUT_FLAGS = ["--timeout=%d" % TIMEOUT, "--timeout-disable-debugger-detection"]
TEMPLATE = " ".join([VENV_PY_REL, "-m", "pytest"] + ISOLATE + ["-q"] + TIMEOUT_FLAGS + ["'{node_id}'"])

# Failure behaviour of each buggy program, checked against the test runs (the fix itself is
# added automatically from the diff).
NOTES = {
    "bitcount": "`n ^= n - 1` never reaches 0: every test loops forever (timeout-only program).",
    "breadth_first_search": "Only the unreachable-goal case (test3) fails: IndexError, pop from an empty deque.",
    "bucketsort": "Wrong output (iterates the input instead of the counts).",
    "depth_first_search": "Only the cyclic graph (test5) fails: unbounded recursion -> RecursionError "
                          "(~1000 frames). Anchor line 9 is an `else:` header with no bytecode of its own.",
    "detect_cycle": "Only test4 (2-node acyclic list) fails: AttributeError on `None.successor`.",
    "find_first_in_sorted": "x above the maximum -> IndexError (1 test); any other absent x -> infinite loop (2 timeouts).",
    "find_in_sorted": "Absent x -> unbounded recursion -> RecursionError.",
    "flatten": "Yields generator objects; assertion messages embed memory addresses (vary per run).",
    "gcd": "Unbounded recursion -> RecursionError (5/6 tests).",
    "get_factors": "Drops the last prime factor.",
    "hanoi": "Wrong move list.",
    "is_valid_parenthesization": "Accepts unclosed '('.",
    "kheapsort": "Pushes the first k elements twice.",
    "knapsack": "Wrong optimum. The 10th case (README: ~4 min) is skipped unless --runslow, which the template does not pass.",
    "kth": "Recurses with the wrong k until the list is empty -> IndexError.",
    "lcs_length": "Wrong length.",
    "levenshtein": "Wrong distance. Exponential recursion: input_data2 takes ~1 s untraced; the test file always skips the 4th case.",
    "lis": "Wrong length.",
    "longest_common_subsequence": "Wrong subsequence; exponential recursion (slowest case ~0.2 s untraced).",
    "max_sublist_sum": "Wrong sum (running sum never reset).",
    "mergesort": "1-element lists recurse forever -> RecursionError on every non-empty input (13/14 tests).",
    "minimum_spanning_tree": "Mutates the set being iterated -> RuntimeError on all 3 tests. Input is a dict of weighted edges.",
    "next_palindrome": "Only the all-9s case fails (one zero too many).",
    "next_permutation": "Wrong permutation on all 8 tests.",
    "pascal": "Rows one element short -> IndexError (3 tests) or wrong rows (1).",
    "possible_change": "No empty-coins guard -> ValueError unpacking [] (9/10 tests).",
    "powerset": "Loses the subsets without `first`.",
    "quicksort": "Drops elements equal to the pivot: only the one input with duplicates fails.",
    "reverse_linked_list": "Returns None for non-empty lists.",
    "rpn_eval": "Swapped operands: exactly the inputs with '-' or '/' fail. The correct file's alternative "
                "swaps them inside `op` (line 9) instead, but relies on an undefined `Stack` class.",
    "shortest_path_length": "Wrong distances.",
    "shortest_path_lengths": "Wrong distances on all 4 tests. Input is (n, dict of edge lengths).",
    "shortest_paths": "Writes into the input dict instead of weight_by_node (mutates the test's input). Input is a dict of edge weights.",
    "shunting_yard": "Operators are dropped. Anchor line 17 is the body of the inner `while`, which never runs "
                     "in the buggy version (opstack stays empty).",
    "sieve": "Returns [].",
    "sqrt": "Loop exits only if the first Newton step lands within epsilon ([2, 0.5] passes): 6 timeouts, "
            "no assertion failure (timeout-only program).",
    "subsequences": "Base case returns [] so everything is [].",
    "to_base": "Digits come out reversed.",
    "topological_ordering": "Wrong order on all 3 tests.",
    "wrap": "Drops the final chunk on all 5 tests.",
}

# ----------------------------------------------------------------------------- normalization
_SKIP = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
         tokenize.ENDMARKER, tokenize.ENCODING}


def _is_str_stmt(n):
    return isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)


def normalized_lines(src):
    """[(physical line, key)] for code lines only. Dropped: comments, blank lines, and every bare
    string-expression statement (docstrings, and the module-level triple-quoted blocks in which the
    correct files park alternative implementations). key = (indent width, token strings), so
    intra-line whitespace changes do not count."""
    drop = set()
    for n in ast.walk(ast.parse(src)):
        if _is_str_stmt(n):
            drop.update(range(n.lineno, n.end_lineno + 1))
    toks = {}
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type not in _SKIP:
            toks.setdefault(t.start[0], []).append(t.string)
    lines = src.splitlines()
    return [(ln, (len(lines[ln - 1]) - len(lines[ln - 1].lstrip(" \t")), tuple(toks[ln])))
            for ln in sorted(toks) if ln not in drop]


def diff_fault(b, c, bsrc=None, csrc=None):
    """Convention: buggy lines replaced/deleted by the fix -> 'modified'; a fix that only inserts
    -> 'missing', with the buggy code line just before and just after the insertion point."""
    sm = difflib.SequenceMatcher(None, [k for _, k in b], [k for _, k in c], autojunk=False)
    modified, anchors, hunks = set(), set(), []
    bl = bsrc.splitlines() if bsrc else None
    cl = csrc.splitlines() if csrc else None
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        new = [cl[c[j][0] - 1].strip() for j in range(j1, j2)] if cl else []
        if tag in ("replace", "delete"):
            old = [b[i][0] for i in range(i1, i2)]
            modified.update(old)
            if bl:
                hunks.append("L%s `%s` -> `%s`" % (",".join(map(str, old)),
                                                   " / ".join(bl[x - 1].strip() for x in old),
                                                   " / ".join(new) or "(deleted)"))
        else:
            before = b[i1 - 1][0] if i1 > 0 else None
            after = b[i1][0] if i1 < len(b) else None
            anchors.update(x for x in (before, after) if x is not None)
            hunks.append("inserts `%s` between L%s and L%s" % (" / ".join(new), before, after))
    if modified:
        return "modified", sorted(modified), hunks
    return "missing", sorted(anchors), hunks


def stmt_first_lines(src, lines):
    """First line of the innermost AST statement containing each line."""
    stmts = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.stmt)]
    return sorted({min((s for s in stmts if s.lineno <= ln <= s.end_lineno),
                       key=lambda s: s.end_lineno - s.lineno).lineno for ln in lines})


def _def_lines(src, name):
    d = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name][-1]
    return [(ln, k) for ln, k in normalized_lines(src) if d.lineno <= ln <= d.end_lineno]


def alternatives(bsrc, csrc, name):
    """Alternative `def <name>` kept in module-level strings of the correct file, each diffed
    against the buggy function. Yields (correct-file line of the def, code to append, kind, lines)."""
    b = _def_lines(bsrc, name)
    for n in ast.parse(csrc).body:
        if not _is_str_stmt(n):
            continue
        text = textwrap.dedent(n.value.value)
        try:
            st = ast.parse(text)
        except SyntaxError:
            continue
        imports = "\n".join(ast.get_source_segment(text, x) for x in st.body
                            if isinstance(x, (ast.Import, ast.ImportFrom)))
        for d in st.body:
            if isinstance(d, ast.FunctionDef) and d.name == name:
                seg = "\n".join(text.splitlines()[d.lineno - 1:d.end_lineno]) + "\n"
                kind, lines, _ = diff_fault(b, _def_lines(seg, name))
                yield n.lineno + d.lineno - 1, imports + "\n" + seg, kind, lines


# ----------------------------------------------------------------------------- pytest plumbing
REPORT_PLUGIN = r'''
import json, os, time
_R, _T0 = [], [None]
def pytest_sessionstart(session):
    _T0[0] = time.perf_counter()
def _crash(rep):
    lr = rep.longrepr
    if lr is None: return ""
    if isinstance(lr, tuple): return str(lr[2])
    rc = getattr(lr, "reprcrash", None)
    return rc.message if rc is not None else str(lr)
def pytest_runtest_logreport(report):
    if not (report.when == "call" or (report.when == "setup" and not report.passed)
            or (report.when == "teardown" and report.failed)):
        return
    msg = _crash(report)
    _R.append({"nodeid": report.nodeid, "outcome": report.outcome, "duration": round(report.duration, 4),
               "timeout": bool(report.failed and "Timeout" in msg and "pytest-timeout" in msg),
               "msg": msg[:300]})
def pytest_sessionfinish(session, exitstatus):
    with open(os.environ["QB_REPORT"], "w") as f:
        json.dump({"session_s": round(time.perf_counter() - _T0[0], 3), "results": _R}, f)
'''

# Records, per test, which lines of $QB_TARGET emit a sys.monitoring LINE event (3.12+).
LINES_PLUGIN = r'''
import json, os, sys, pytest
_M = sys.monitoring; _TOOL = 4
_TARGET = os.path.realpath(os.environ["QB_TARGET"]); _HITS = {}; _CUR = [None]
def _on_line(code, line):
    if _CUR[0] is not None and os.path.realpath(code.co_filename) == _TARGET:
        _HITS.setdefault(_CUR[0], set()).add(line)
    return _M.DISABLE  # one event per location per test; restart_events() re-arms
def pytest_configure(config):
    _M.use_tool_id(_TOOL, "qb_lines"); _M.register_callback(_TOOL, _M.events.LINE, _on_line)
    _M.set_events(_TOOL, _M.events.LINE)
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    _CUR[0] = item.nodeid; _M.restart_events()
    try:
        yield
    finally:
        _CUR[0] = None
def pytest_sessionfinish(session, exitstatus):
    with open(os.environ["QB_LINES"], "w") as f:
        json.dump({k: sorted(v) for k, v in _HITS.items()}, f)
'''


class Pytest:
    def __init__(self, py, plugdir):
        self.py, self.plugdir = str(py), plugdir
        (plugdir / "qb_report_plugin.py").write_text(REPORT_PLUGIN)
        (plugdir / "qb_lines_plugin.py").write_text(LINES_PLUGIN)

    def run(self, root, args, plugins=("qb_report_plugin",), extra_env=None):
        rep = self.plugdir / "report.json"
        rep.unlink(missing_ok=True)
        env = dict(os.environ, PYTHONPATH=str(self.plugdir), QB_REPORT=str(rep), **(extra_env or {}))
        argv = [self.py, "-m", "pytest"] + ISOLATE + ["-q"]
        for p in plugins:
            argv += ["-p", p]
        t0 = time.perf_counter()
        subprocess.run(argv + args, cwd=root, env=env, capture_output=True, text=True, timeout=1800)
        out = json.loads(rep.read_text())
        out["wall_s"] = round(time.perf_counter() - t0, 2)
        return out

    def collect(self, root, test_file):
        cp = subprocess.run([self.py, "-m", "pytest"] + ISOLATE + ["--collect-only", "-q", test_file],
                            cwd=root, capture_output=True, text=True)
        return [ln.strip() for ln in cp.stdout.splitlines() if "::" in ln]


def classify(results):
    cls = {"passed": [], "failed": [], "timeout": [], "skipped": []}
    for r in results:
        cls["timeout" if r["timeout"] else r["outcome"]].append(r["nodeid"])
    return cls


# ----------------------------------------------------------------------------- main
def build_one(p, root, pt, tmp, problems):
    bpath, cpath, tfile = f"python_programs/{p}.py", f"correct_python_programs/{p}.py", f"python_testcases/test_{p}.py"
    bsrc, csrc = (root / bpath).read_text(), (root / cpath).read_text()
    kind, lines, hunks = diff_fault(normalized_lines(bsrc), normalized_lines(csrc), bsrc, csrc)
    stmt_lines = stmt_first_lines(bsrc, lines)

    collected = pt.collect(root, tfile)
    buggy = pt.run(root, TIMEOUT_FLAGS + [tfile])
    corr = pt.run(root, ["--correct", "--timeout=%d" % CORRECT_TIMEOUT, "--timeout-disable-debugger-detection", tfile])
    bc, cc = classify(buggy["results"]), classify(corr["results"])
    if {r["nodeid"] for r in buggy["results"]} != set(collected) or not collected:
        problems.append(f"{p}: buggy run reported {len(buggy['results'])} tests, collected {len(collected)}")
    bad = [n for n in bc["failed"] + bc["timeout"] + bc["passed"] if n not in cc["passed"]]
    if bad:
        problems.append(f"{p}: not passing on the correct version: {bad}")
    if set(bc["skipped"]) != set(cc["skipped"]):
        problems.append(f"{p}: skipped sets differ between buggy and correct")
    slow = [(r["nodeid"], r["duration"]) for r in corr["results"] if r["duration"] > TIMEOUT / 2]
    if slow:
        problems.append(f"{p}: correct version uses >{TIMEOUT / 2}s of the {TIMEOUT}s budget: {slow}")

    confirmed = []
    if bc["timeout"]:
        conf = pt.run(root, ["--timeout=%d" % CONFIRM_TIMEOUT, "--timeout-disable-debugger-detection"] + bc["timeout"])
        confirmed = classify(conf["results"])["timeout"]
        if set(confirmed) != set(bc["timeout"]):
            problems.append(f"{p}: only {len(confirmed)}/{len(bc['timeout'])} timeouts persist at {CONFIRM_TIMEOUT}s")

    fails = bc["failed"] + bc["timeout"]
    hits = {str(ln): 0 for ln in lines}
    if fails:
        lf = tmp / "lines.json"
        lf.unlink(missing_ok=True)
        pt.run(root, TIMEOUT_FLAGS + fails, plugins=("qb_report_plugin", "qb_lines_plugin"),
               extra_env={"QB_TARGET": str(root / bpath), "QB_LINES": str(lf)})
        lh = json.loads(lf.read_text())
        hits = {str(ln): sum(ln in lh.get(n, []) for n in fails) for ln in lines}
    else:
        problems.append(f"{p}: no failing test")

    alt_ok, alt_bad = {}, {}   # (kind, lines) -> [correct-file lines of the alternative defs]
    for where, code, akind, alines in alternatives(bsrc, csrc, p):
        if (akind, alines) == (kind, lines):
            continue  # same location as the primary fix: nothing new
        work = tmp / "alt"
        if not work.exists():
            shutil.copytree(root, work, ignore=shutil.ignore_patterns(".git", "*java*", "__pycache__"))
        (work / cpath).write_text(csrc + "\n\n# alternative under validation\n" + code)
        r = classify(pt.run(work, ["--correct", "--timeout=%d" % CORRECT_TIMEOUT, tfile])["results"])
        (work / cpath).write_text(csrc)
        nbad = len(r["failed"]) + len(r["timeout"])
        (alt_bad if nbad else alt_ok).setdefault((akind, tuple(alines)), []).append((where, nbad))

    note = ["Fix: " + "; ".join(hunks) + ".", NOTES[p]]
    test_src = (root / tfile).read_text()
    if "load_json_testcases" not in test_src:
        note.append("No JSON testcases: inputs are built inside the test functions"
                    + (" from python_testcases/node.py (Node objects)." if "from node import Node" in test_src else "."))
    for n in bc["skipped"]:
        note.append(f"Skipped by the test file: {n.split('::')[1]}.")
    if bc["timeout"]:
        note.append(f"{len(confirmed)}/{len(bc['timeout'])} timeouts persist at {CONFIRM_TIMEOUT} s (non-terminating, not slow).")
    if stmt_lines != lines:
        note.append(f"The innermost statements containing the fault line(s) start at {stmt_lines} (fault_stmt_lines).")
    dead = [ln for ln in lines if hits[str(ln)] == 0]
    if dead:
        note.append(f"Line(s) {dead} emit no LINE event in any failing test, so no dynamic slice can contain them.")
    for (akind, alines), lst in alt_ok.items():
        note.append(f"Alternative fix(es) kept in {cpath} at line(s) {[w for w, _ in lst]} pass all tests and "
                    + (f"change buggy line(s) {list(alines)}" if akind == "modified"
                       else f"insert code between buggy lines {list(alines)}") + " instead (alt_fault_lines).")
    for (akind, alines), lst in alt_bad.items():
        note.append(f"The alternative at {cpath}:{lst[0][0]} ({akind} {list(alines)}) fails {lst[0][1]} test(s) "
                    "as written; not used.")

    return {
        "program": p,
        "buggy_path": bpath,
        "correct_path": cpath,
        "test_file": tfile,
        "fault_lines": lines,
        "fault_kind": kind,
        "fault_stmt_lines": stmt_lines,
        "alt_fault_lines": [{"kind": k, "lines": list(ls), "correct_file_lines": [w for w, _ in lst]}
                            for (k, ls), lst in alt_ok.items()],
        "failing_tests": bc["failed"],
        "timeout_tests": bc["timeout"],
        "passing_tests": bc["passed"],
        "skipped_tests": bc["skipped"],
        "fault_line_hits": hits,
        "runtime_buggy_s": buggy["wall_s"],
        "runtime_correct_s": corr["wall_s"],
        "slowest_correct_test_s": max([r["duration"] for r in corr["results"]] or [0.0]),
        "correct_version_passes": not bad,
        "test_cmd_template": TEMPLATE,
        "notes": " ".join(note),
    }


def verify_isolated(tasks, root, tmp, problems):
    """Run every node id alone through the exact template (via /bin/sh) and compare outcomes."""
    n = mism = 0
    for t in tasks:
        expect = {**{x: "failed" for x in t["failing_tests"]}, **{x: "timeout" for x in t["timeout_tests"]},
                  **{x: "passed" for x in t["passing_tests"]}, **{x: "skipped" for x in t["skipped_tests"]}}
        for nid, want in expect.items():
            rep = tmp / "iso.json"
            rep.unlink(missing_ok=True)
            env = dict(os.environ, PYTHONPATH=str(tmp), QB_REPORT=str(rep), PYTEST_ADDOPTS="-p qb_report_plugin")
            subprocess.run(t["test_cmd_template"].format(node_id=nid), shell=True, cwd=root, env=env,
                           capture_output=True, text=True, timeout=600)
            res = json.loads(rep.read_text())["results"] if rep.exists() else []
            got = next((k for k, v in classify(res).items() if v), "none") if len(res) == 1 else f"{len(res)} results"
            n += 1
            if got != want:
                mism += 1
                problems.append(f"isolated run of {nid}: {got}; batch run: {want}")
    print(f"isolated verification: {n} node ids, {mism} mismatches", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quixbugs", default=str(HERE / "data" / "QuixBugs"))
    ap.add_argument("--python", default=str(HERE / ".venv-quixbugs" / "bin" / "python"))
    ap.add_argument("--out", default=str(HERE / "quixbugs_tasks.json"))
    ap.add_argument("--verify-isolated", action="store_true",
                    help="also run every node id alone through test_cmd_template and compare")
    a = ap.parse_args()
    root = Path(a.quixbugs).resolve()
    py = Path(a.python).absolute()          # not resolve(): the venv python is a symlink to the base interpreter
    if (root / VENV_PY_REL).resolve() != py.resolve():
        raise SystemExit(f"test_cmd_template assumes {VENV_PY_REL} relative to {root}")
    progs = sorted(x.stem for x in (root / "python_programs").glob("*.py")
                   if not x.stem.endswith("_test") and x.stem != "node")
    assert len(progs) == 40, progs
    tmp = Path(tempfile.mkdtemp(prefix="qb_build_"))
    pt = Pytest(py, tmp)
    tasks, problems = [], []
    try:
        for p in progs:
            t0 = time.perf_counter()
            t = build_one(p, root, pt, tmp, problems)
            tasks.append(t)
            print(f"{p:27s} {t['fault_kind']:8s} {t['fault_lines']} fail={len(t['failing_tests'])} "
                  f"timeout={len(t['timeout_tests'])} pass={len(t['passing_tests'])} skip={len(t['skipped_tests'])} "
                  f"hits={t['fault_line_hits']} ({time.perf_counter() - t0:.1f}s)", flush=True)
        if a.verify_isolated:
            verify_isolated(tasks, root, tmp, problems)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    tmp_out = Path(a.out + ".tmp")   # atomic replace: other sessions may be reading the file
    tmp_out.write_text(json.dumps(tasks, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp_out, a.out)
    print(f"wrote {a.out}: {len(tasks)} programs")
    print("PROBLEMS:" if problems else "no problems", *problems, sep="\n  ")


if __name__ == "__main__":
    main()
