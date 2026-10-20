#!/usr/bin/env python3
"""Seeded single-token mutants for the whyslice debugging benchmark.

    python make_mutants.py generate --lib tabulate          # -> mutants/<lib>/candidates.jsonl
    python make_mutants.py filter   --lib tabulate -j 4     # -> mutants/<lib>/results.jsonl, summary.json

`generate` edits exactly one token in place with `tokenize` (all other bytes are
untouched), only in non-test source files and only inside function bodies (never
module- or class-level code, never comments, docstrings or f-strings). Tokens are
sampled uniformly over all eligible tokens with a fixed seed.

`filter` runs every candidate against the library's full test suite in a private
copy of the clone (the library is imported from the copy's working directory, as
in the tasks) and records which filter stage each mutant passes:
  a  the library (and the mutated module) still import
  c  the full suite finishes within 60 s (per-test timeout 10 s)
  b  1..3 test nodes fail, no collection errors, and the run leaves the checkout
     unchanged (every worker copy is restored to the clone after each mutant)
  d  the first failing node, run alone with --tb=long, fails and none of the
     file:line pairs it reports falls on the mutated statement
  e  depth >= 2: whenever that test executes the mutated line there is at least
     one intermediate named function frame between the test function and the
     mutated frame (own sys.settrace probe, tools/bench_probe.py; <genexpr>,
     <lambda> and comprehension frames are not counted)
Stdlib only; run with a CPython 3.12 interpreter.
"""

import argparse
import ast
import hashlib
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import tokenize
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BENCH = Path(__file__).resolve().parent
UPSTREAM = BENCH / "upstream"
MUTANTS = BENCH / "mutants"
TOOLS = BENCH / "tools"
WORK = BENCH / "_work"

SEED = 20261020
N_CANDIDATES = 250
SUITE_TIMEOUT = 60.0
TEST_TIMEOUT = 10.0

LIBS = {
    "tabulate": {"tag": "v0.10.0", "pkg": "tabulate", "src": "tabulate", "tests": "test"},
    "sqlparse": {"tag": "0.6.0", "pkg": "sqlparse", "src": "sqlparse", "tests": "tests"},
}

COMPARE = {"<": "<=", "<=": "<", ">": ">=", ">=": ">", "==": "!=", "!=": "=="}
ARITH = {"+": "-", "-": "+"}
BOOLOP = {"and": "or", "or": "and"}
BOOLCONST = {"True": "False", "False": "True"}
COPY_IGNORE = shutil.ignore_patterns(".git", ".venv", "__pycache__", "*.egg-info", ".pytest_cache", ".tox")


# --------------------------------------------------------------------------- generate

def _span_contains(node, start, end):
    return (node.lineno, node.col_offset) <= start and end <= (node.end_lineno, node.end_col_offset)


def _functions(tree):
    """[(body_start, body_end, qualname, def_first_line, def_last_line)] for every def."""
    out = []

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = prefix + child.name
                first = min([child.lineno] + [d.lineno for d in child.decorator_list])
                out.append(((child.body[0].lineno, child.body[0].col_offset),
                            (child.body[-1].end_lineno, child.body[-1].end_col_offset),
                            qual, first, child.end_lineno))
                visit(child, qual + ".<locals>.")
            elif isinstance(child, ast.ClassDef):
                visit(child, prefix + child.name + ".")
            else:
                visit(child, prefix)

    visit(tree, "")
    return out


def _stmt_lines(tree, start, end):
    """Lines of the innermost statement holding the token (header only for compound stmts)."""
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.stmt, ast.ExceptHandler)) and _span_contains(node, start, end):
            if best is None or (node.lineno, node.col_offset) >= (best.lineno, best.col_offset):
                best = node
    if best is None:
        return [start[0], end[0]]
    body = getattr(best, "body", None)
    if isinstance(body, list) and body:
        return [best.lineno, max(best.lineno, body[0].lineno - 1)]
    return [best.lineno, best.end_lineno]


def _expr_lines(tree, start, end):
    """Lines of the innermost expression holding the token (what the depth probe watches)."""
    best = None
    for node in ast.walk(tree):
        if isinstance(node, ast.expr) and _span_contains(node, start, end):
            if best is None or ((node.lineno, node.col_offset) >= (best.lineno, best.col_offset)
                                and (node.end_lineno, node.end_col_offset) <= (best.end_lineno, best.end_col_offset)):
                best = node
    if best is None:
        return [start[0], end[0]]
    return [best.lineno, best.end_lineno]


def _options(tok, prev_sig, next_sig):
    """Mutation options [(operator, replacement)] for one token; [] if not eligible."""
    s = tok.string
    if tok.type == tokenize.OP:
        if s in COMPARE:
            return [("compare", COMPARE[s])]
        if s in ARITH:
            return [("arith", ARITH[s])]
    elif tok.type == tokenize.NAME:
        if s in BOOLOP:
            return [("boolop", BOOLOP[s])]
        if s in BOOLCONST:
            return [("bool-const", BOOLCONST[s])]
        if s == "not":
            # unary `not` only: not the binary operators `not in` / `is not`, and the
            # operand must start on the same line so that removing "not " is local.
            if prev_sig is not None and prev_sig.string == "is":
                return []
            if next_sig is None or next_sig.string == "in" or next_sig.start[0] != tok.end[0]:
                return []
            return [("remove-not", "")]
    elif tok.type == tokenize.NUMBER and re.fullmatch(r"[0-9]", s):
        n = int(s)
        opts = [("int+1", str(n + 1))]
        if n >= 1:  # 0 -> -1 would not be a single token
            opts.append(("int-1", str(n - 1)))
        return opts
    return []


def _apply(text, m):
    lines = text.splitlines(keepends=True)
    i = m["line"] - 1
    line = lines[i]
    assert line[m["col"]:m["end_col"]].rstrip() == m["original"], (m, line)
    lines[i] = line[:m["col"]] + m["mutated"] + line[m["end_col"]:]
    return "".join(lines)


def eligible(lib):
    cfg = LIBS[lib]
    root = UPSTREAM / lib
    out = []
    for path in sorted((root / cfg["src"]).rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        funcs = _functions(tree)
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
        sig_types = {tokenize.OP, tokenize.NAME, tokenize.NUMBER, tokenize.STRING,
                     tokenize.FSTRING_START, tokenize.FSTRING_END}
        sig = [k for k, t in enumerate(toks) if t.type in sig_types]
        sig_pos = {k: j for j, k in enumerate(sig)}
        fdepth = 0
        lines = text.splitlines(keepends=True)
        for k, tok in enumerate(toks):
            if tok.type == tokenize.FSTRING_START:
                fdepth += 1
                continue
            if tok.type == tokenize.FSTRING_END:
                fdepth -= 1
                continue
            if fdepth or k not in sig_pos:
                continue
            j = sig_pos[k]
            prev_sig = toks[sig[j - 1]] if j > 0 else None
            next_sig = toks[sig[j + 1]] if j + 1 < len(sig) else None
            opts = _options(tok, prev_sig, next_sig)
            if not opts:
                continue
            owners = [f for f in funcs if f[0] <= tok.start and tok.end <= f[1]]
            if not owners:
                continue  # module- or class-level code
            owner = max(owners, key=lambda f: f[0])
            end_col = tok.end[1]
            if opts[0][0] == "remove-not":  # also drop the blank(s) up to the operand
                end_col = next_sig.start[1]
            base = {
                "file": rel, "line": tok.start[0], "col": tok.start[1], "end_col": end_col,
                "original": tok.string, "function": owner[2], "func_lines": [owner[3], owner[4]],
                "stmt_lines": _stmt_lines(tree, tok.start, tok.end),
                "probe_lines": _expr_lines(tree, tok.start, tok.end),
                "line_text": lines[tok.start[0] - 1].strip(),
            }
            choices = []
            for op, rep in opts:
                m = dict(base, operator=op, mutated=rep)
                try:
                    compile(_apply(text, m), rel, "exec")
                except SyntaxError:
                    continue
                choices.append(m)
            if choices:
                out.append(choices)
    return out


def cmd_generate(args):
    lib = args.lib
    tokens = eligible(lib)
    rng = random.Random(f"{SEED}:{lib}")
    k = min(args.n, len(tokens))
    picked = sorted(rng.sample(range(len(tokens)), k))
    outdir = MUTANTS / lib
    outdir.mkdir(parents=True, exist_ok=True)
    by_op = {}
    with open(outdir / "candidates.jsonl", "w", encoding="utf-8") as fh:
        for n, idx in enumerate(picked):
            m = dict(rng.choice(tokens[idx]))
            m["id"] = f"{lib}-m{n:03d}"
            by_op[m["operator"]] = by_op.get(m["operator"], 0) + 1
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    elig_ops = {}
    for choices in tokens:
        for c in choices:
            elig_ops[c["operator"]] = elig_ops.get(c["operator"], 0) + 1
    meta = {"lib": lib, "seed": SEED, "eligible_tokens": len(tokens), "eligible_mutations": elig_ops,
            "candidates": k, "candidates_by_operator": by_op}
    (outdir / "generate.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))


# --------------------------------------------------------------------------- filter

def _env(report, extra=None):
    env = dict(os.environ)
    for key in [k for k in env if k.startswith("BENCH_") or k.startswith("PYTEST_")]:
        del env[key]
    env.update({"PYTHONPATH": str(TOOLS), "PYTHONDONTWRITEBYTECODE": "1", "BENCH_REPORT": str(report),
                "BENCH_TEST_TIMEOUT": str(TEST_TIMEOUT)})
    env.pop("VIRTUAL_ENV", None)
    env.update(extra or {})
    return env


def _run(cmd, cwd, env, timeout):
    """(returncode or None on timeout, combined output, seconds); kills the whole process group."""
    t0 = time.monotonic()
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out, time.monotonic() - t0
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, 9)
        except ProcessLookupError:
            pass
        out, _ = p.communicate()
        return None, out, time.monotonic() - t0


def _read_report(path):
    recs = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            recs.append(json.loads(line))
        path.unlink()
    return recs


SKIP_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".tox"}


def _snapshot(root):
    """{relative path: content digest} of every file in a checkout (symlinks by target)."""
    snap = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(part in SKIP_PARTS or part.endswith(".egg-info") for part in rel.parts):
            continue
        if p.is_symlink():
            snap[rel.as_posix()] = "link:" + os.readlink(p)
        elif p.is_file():
            snap[rel.as_posix()] = hashlib.sha1(p.read_bytes()).hexdigest()
    return snap


def _restore_tree(wdir, src_root, snap, skip=()):
    """Undo whatever a test run changed in the worker copy; return the changed paths."""
    now = _snapshot(wdir)
    changed = sorted(k for k in set(snap) | set(now) if k not in skip and snap.get(k) != now.get(k))
    for rel in changed:
        dst = wdir / rel
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        if rel in snap:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_root / rel, dst, follow_symlinks=False)
    return changed


TB_LOC = re.compile(r"^(?P<path>\S[^:\n]*?\.py):(?P<line>\d+):(?: |$)", re.M)


def check_one(lib, m, wdir, py, snap):
    """Run filters a-e for mutant m inside worker copy wdir. Returns a result dict."""
    cfg = LIBS[lib]
    res = {"id": m["id"], "stage": "generated"}
    target = wdir / m["file"]
    original = target.read_text(encoding="utf-8")
    report = wdir.parent / f"{wdir.name}.report.jsonl"
    report.unlink(missing_ok=True)
    try:
        target.write_text(_apply(original, m), encoding="utf-8")
        # (a) imports
        modname = m["file"][:-3].replace("/", ".")
        if modname.endswith(".__init__"):
            modname = modname[: -len(".__init__")]
        rc, out, _ = _run([py, "-c", f"import {cfg['pkg']}, {modname}"], wdir, _env(report), 30)
        if rc != 0:
            res["import_error"] = out.strip().splitlines()[-1:] if out else None
            return res
        res["stage"] = "a"
        # (c) + (b) full suite
        cmd = [py, "-m", "pytest", "-p", "no:cacheprovider", "-p", "bench_probe", "-q", "--tb=no",
               cfg["tests"]]
        rc, out, dt = _run(cmd, wdir, _env(report, {"BENCH_LIB": cfg["pkg"]}), SUITE_TIMEOUT)
        recs = _read_report(report)
        res["suite_seconds"] = round(dt, 2)
        if rc is None:
            res["suite_timeout"] = True
            return res
        res["stage"] = "c"
        sess = [r for r in recs if r["kind"] == "session"]
        res["lib_file"] = sess[-1].get("lib_file") if sess else None
        coll = [r["nodeid"] for r in recs if r["kind"] == "collect_error"]
        failing = []
        for r in recs:
            if r["kind"] == "fail" and r["nodeid"] not in failing:
                failing.append(r["nodeid"])
        res["failing"] = failing
        res["n_failing"] = len(failing)
        res["collect_errors"] = coll
        res["exit"] = rc
        # a mutant that makes the suite rewrite files of the checkout (e.g. the CLI's
        # --in-place default flipped) is rejected: its failures depend on test order
        res["tree_modified"] = _restore_tree(wdir, UPSTREAM / lib, snap, skip={m["file"]})
        if coll or not (1 <= len(failing) <= 3) or not sess or res["tree_modified"]:
            return res
        res["stage"] = "b"
        # (d) first failing node alone, --tb=long
        first = failing[0]
        res["failing_test"] = first
        rc, out, _ = _run([py, "-m", "pytest", "-p", "no:cacheprovider", "--tb=long", first], wdir,
                          _env(report), SUITE_TIMEOUT)
        report.unlink(missing_ok=True)
        if rc != 1:
            res["alone_exit"] = rc
            return res
        locs = []
        for mo in TB_LOC.finditer(out):
            p = Path(mo.group("path"))
            p = (wdir / p).resolve() if not p.is_absolute() else p.resolve()
            try:
                rel = p.relative_to(wdir.resolve()).as_posix()
            except ValueError:
                rel = str(p)
            locs.append([rel, int(mo.group("line"))])
        res["tb_locs"] = locs
        lo, hi = m["stmt_lines"]
        flo, fhi = m["func_lines"]
        in_tb = any(f == m["file"] and (lo <= ln <= hi or ln == m["line"]) for f, ln in locs)
        res["func_in_tb"] = any(f == m["file"] and flo <= ln <= fhi for f, ln in locs)
        mutated_line = target.read_text(encoding="utf-8").splitlines()[m["line"] - 1].strip()
        res["mutated_line_in_output"] = mutated_line in out
        res["error_line"] = next((ln.strip() for ln in out.splitlines() if ln.startswith("E ")), None)
        if in_tb:
            return res
        res["stage"] = "d"
        # (e) depth probe on the same node
        plo, phi = m["probe_lines"]
        extra = {"BENCH_PROBE_FILE": str(target), "BENCH_PROBE_LINES": f"{plo}-{phi}"}
        rc, out, _ = _run([py, "-m", "pytest", "-p", "no:cacheprovider", "-p", "bench_probe", "-q",
                           "--tb=no", first], wdir, _env(report, extra), SUITE_TIMEOUT * 3)
        probes = [r for r in _read_report(report) if r["kind"] == "probe"]
        if not probes:
            res["probe"] = None
            return res
        pr = probes[0]
        res["probe"] = pr
        res["depth"] = pr["min_depth"]
        if pr["min_depth"] is None or pr["min_depth"] < 2:
            return res
        res["stage"] = "e"
        return res
    finally:
        target.write_text(original, encoding="utf-8")
        report.unlink(missing_ok=True)
        _restore_tree(wdir, UPSTREAM / lib, snap)


def baseline(lib, wdir, py):
    """Unmutated suite in a worker copy; must pass completely."""
    cfg = LIBS[lib]
    report = wdir.parent / f"{wdir.name}.report.jsonl"
    report.unlink(missing_ok=True)
    cmd = [py, "-m", "pytest", "-p", "no:cacheprovider", "-p", "bench_probe", "-q", "--tb=short", cfg["tests"]]
    rc, out, dt = _run(cmd, wdir, _env(report, {"BENCH_LIB": cfg["pkg"]}), SUITE_TIMEOUT)
    recs = _read_report(report)
    sess = [r for r in recs if r["kind"] == "session"]
    return {"exit": rc, "seconds": round(dt, 2), "fails": [r for r in recs if r["kind"] != "session"],
            "lib_file": sess[-1].get("lib_file") if sess else None, "tail": out.strip().splitlines()[-1:]}


STAGES = ["generated", "a", "c", "b", "d", "e"]


def cmd_filter(args):
    lib = args.lib
    root = UPSTREAM / lib
    py = str(root / ".venv" / "bin" / "python")
    cands = [json.loads(l) for l in open(MUTANTS / lib / "candidates.jsonl", encoding="utf-8")]
    if args.limit:
        cands = cands[: args.limit]
    wroot = WORK / lib
    shutil.rmtree(wroot, ignore_errors=True)
    wroot.mkdir(parents=True)
    workers = []
    for k in range(args.jobs):
        w = wroot / f"w{k}"
        shutil.copytree(root, w, symlinks=True, ignore=COPY_IGNORE)
        workers.append(w)
    snap = _snapshot(root)
    base = baseline(lib, workers[0], py)
    print("baseline:", json.dumps(base), flush=True)
    base["tree_modified"] = _restore_tree(workers[0], root, snap)
    if base["exit"] != 0 or not str(base["lib_file"]).startswith(str(workers[0])) or base["tree_modified"]:
        sys.exit("baseline failed, imported the library from elsewhere, or modified the checkout")
    free = list(workers)
    lock = threading.Lock()
    results = {}
    done = [0]

    def job(m):
        with lock:
            w = free.pop()
        try:
            r = check_one(lib, m, w, py, snap)
        finally:
            with lock:
                free.append(w)
                done[0] += 1
                if done[0] % 25 == 0:
                    print(f"  {done[0]}/{len(cands)}", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for r in ex.map(job, cands):
            results[r["id"]] = r
    out = MUTANTS / lib / "results.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for m in cands:
            fh.write(json.dumps(dict(m, result=results[m["id"]]), ensure_ascii=False) + "\n")
    counts = {s: sum(1 for r in results.values() if STAGES.index(r["stage"]) >= STAGES.index(s)) for s in STAGES}
    summary = {"lib": lib, "baseline": base, "survivors_by_stage": counts,
               "depth_of_survivors": sorted(r["depth"] for r in results.values() if r["stage"] == "e")}
    (MUTANTS / lib / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["survivors_by_stage"]), summary["depth_of_survivors"])
    if not args.keep_work:
        shutil.rmtree(wroot, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--lib", required=True, choices=sorted(LIBS))
    g.add_argument("-n", type=int, default=N_CANDIDATES)
    g.set_defaults(func=cmd_generate)
    f = sub.add_parser("filter")
    f.add_argument("--lib", required=True, choices=sorted(LIBS))
    f.add_argument("-j", "--jobs", type=int, default=4)
    f.add_argument("--limit", type=int, default=0)
    f.add_argument("--keep-work", action="store_true")
    f.set_defaults(func=cmd_filter)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
