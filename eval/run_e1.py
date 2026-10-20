"""E1: offline evaluation of WhySlice on QuixBugs (no language model involved).

For every failing test of every buggy program:
  * record the failing run with WhySlice and slice from the failure, with
    potential dependences (relevant slicing) and without (classic dynamic slicing);
  * fault-in-slice: is a ground-truth fault line of the program in the slice;
  * slice size relative to the executed lines of the program file;
  * reading cost: how many distinct program lines the causal tree reveals before
    the fault line, and whether the fault shows up in the default report;
  * baseline: Ochiai (SBFL) ranks computed from the executed lines of all tests of
    the program; hybrid = Ochiai restricted to the slice.

    python eval/run_e1.py [--jobs 6] [--out eval/results/e1.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from whyslice.runner import record  # noqa: E402
from whyslice.trace import Trace  # noqa: E402

QB = os.path.join(HERE, "data", "QuixBugs")
PY = os.path.join(HERE, ".venv-quixbugs", "bin", "python")


def run_one(nodeid: str, limit: int) -> Optional[Trace]:
    fd, out = tempfile.mkstemp(suffix=".pkl", prefix="e1-")
    os.close(fd)
    try:
        rec = record(f"pytest -q -p no:cacheprovider {nodeid}", cwd=QB, python=PY, timeout=180,
                     limit=limit, out=out, root=QB)
        return rec.trace
    finally:
        if os.path.exists(out):
            os.remove(out)


def program_lines(trace: Trace, rel: str) -> Tuple[Optional[int], Set[int]]:
    path = os.path.realpath(os.path.join(QB, rel))
    for i, p in enumerate(trace.files):
        if os.path.realpath(p) == path:
            return i, {ln for f, ln in trace.executed_lines() if f == i}
    return None, set()


def ochiai(cover: Dict[str, Set[int]], failing: Set[str]) -> Dict[int, float]:
    lines = set().union(*cover.values()) if cover else set()
    total_fail = len(failing)
    scores = {}
    for ln in lines:
        ef = sum(1 for t in failing if ln in cover.get(t, set()))
        ep = sum(1 for t, s in cover.items() if t not in failing and ln in s)
        denom = math.sqrt(total_fail * (ef + ep)) if total_fail and (ef + ep) else 0.0
        scores[ln] = ef / denom if denom else 0.0
    return scores


def rank_of(targets: Set[int], scores: Dict[int, float], pool: Optional[Set[int]] = None) -> Optional[float]:
    """Mid-rank (ties averaged) of the best-scored target line within ``pool``."""
    pool = set(scores) if pool is None else (pool & set(scores))
    hits = [t for t in targets if t in pool]
    if not hits:
        return None
    best = max(scores[t] for t in hits)
    higher = sum(1 for ln in pool if scores[ln] > best)
    ties = sum(1 for ln in pool if scores[ln] == best)
    return higher + (ties + 1) / 2


def evaluate(task: dict, limit: int) -> dict:
    t0 = time.time()
    prog = task["buggy_path"]
    def ints(xs):
        out = set()
        for x in xs or []:
            if isinstance(x, int):
                out.add(x)
            elif isinstance(x, dict):  # {'kind', 'lines': [buggy-file lines], 'correct_file_line'}
                out |= {v for v in x.get("lines", []) if isinstance(v, int)}
        return out
    raw_faults = ints(task["fault_lines"]) | ints(task.get("fault_stmt_lines")) | ints(task.get("alt_fault_lines"))
    faults = set(raw_faults)
    failing = [n for n in task.get("failing_tests", []) + task.get("timeout_tests", [])]
    passing = list(task.get("passing_tests", []))
    cover: Dict[str, Set[int]] = {}
    per_fail: List[dict] = []
    for nodeid in passing + failing:
        trace = run_one(nodeid, limit)
        if trace is None:
            continue
        f, executed = program_lines(trace, prog)
        cover[nodeid] = executed
        if f is not None:  # fault lines -> the statement units the recorder reports
            unit_of = trace.line_to_unit.get(f, {})
            faults |= {unit_of.get(ln, ln) for ln in raw_faults}
        if nodeid not in failing or f is None:
            continue
        seeds = trace.failure_seeds()
        if not seeds:
            continue
        rel = {ln for ff, ln in trace.lines_of(trace.backward(seeds, True) | set(seeds)) if ff == f}
        dyn = {ln for ff, ln in trace.lines_of(trace.backward(seeds, False) | set(seeds)) if ff == f}
        order = trace.causal_order(seeds[0])
        seen: List[int] = []
        reach = None
        for iid in order:
            ti = trace.insts[iid]
            if ti[2] != f or ti[4] == "entry":
                continue
            if ti[3] not in seen:
                seen.append(ti[3])
            if ti[3] in faults:
                reach = len(seen)
                break
        shown = trace._tree(seeds[0], trace.edges(seeds[0]), 30)
        in_report = any(trace.insts[c][3] in faults and trace.insts[c][2] == f
                        for kids in shown.values() for c, _l, _n in kids) or \
            (trace.insts[seeds[0]][2] == f and trace.insts[seeds[0]][3] in faults)
        per_fail.append({
            "test": nodeid,
            "timeout": nodeid in task.get("timeout_tests", []),
            "truncated": bool(trace.data.get("truncated")),
            "executed": len(executed),
            "slice_rel": len(rel), "slice_dyn": len(dyn),
            "hit_rel": bool(faults & rel), "hit_dyn": bool(faults & dyn),
            "reach": reach, "in_report": bool(in_report),
            "rel_lines": sorted(rel),
            "record_s": trace.data.get("elapsed"),
        })
    fail_set = {p["test"] for p in per_fail}
    scores = ochiai(cover, fail_set)
    executed_all = set().union(*cover.values()) if cover else set()
    for p in per_fail:
        p["sbfl_rank"] = rank_of(faults, scores)
        p["hybrid_rank"] = rank_of(faults, scores, set(p["rel_lines"]))
        p["n_executed_all"] = len(executed_all)
    return {"program": task["program"], "fault_lines": sorted(faults), "fault_kind": task.get("fault_kind"),
            "n_tests": len(cover), "failing": per_fail, "seconds": round(time.time() - t0, 1)}


def summarize(results: List[dict]) -> dict:
    rows = []
    for r in results:
        fs = [p for p in r["failing"]]
        if not fs:
            continue
        p = fs[0]  # first failing test = the one an agent would be handed
        rows.append({**p, "program": r["program"], "fault_kind": r["fault_kind"]})
    n = len(rows)

    def frac(key):
        return sum(1 for x in rows if x[key]) / n if n else 0.0

    def med(vals):
        vals = [v for v in vals if v is not None]
        return statistics.median(vals) if vals else None

    def topk(key, k):
        return sum(1 for x in rows if x[key] is not None and x[key] <= k)

    return {
        "programs_evaluated": n,
        "fault_in_slice_relevant": frac("hit_rel"),
        "fault_in_slice_classic": frac("hit_dyn"),
        "fault_in_default_report": frac("in_report"),
        "median_executed_lines": med([x["executed"] for x in rows]),
        "median_slice_lines_relevant": med([x["slice_rel"] for x in rows]),
        "median_slice_lines_classic": med([x["slice_dyn"] for x in rows]),
        "median_slice_ratio_relevant": med([x["slice_rel"] / x["executed"] for x in rows if x["executed"]]),
        "median_lines_read_before_fault": med([x["reach"] for x in rows]),
        "sbfl_top1": topk("sbfl_rank", 1), "sbfl_top3": topk("sbfl_rank", 3), "sbfl_top5": topk("sbfl_rank", 5),
        "hybrid_top1": topk("hybrid_rank", 1), "hybrid_top3": topk("hybrid_rank", 3),
        "hybrid_top5": topk("hybrid_rank", 5),
        "truncated_runs": sum(1 for x in rows if x["truncated"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=200_000)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "e1.json"))
    args = ap.parse_args()
    tasks = json.load(open(os.path.join(HERE, "quixbugs_tasks.json")))
    if args.only:
        tasks = [t for t in tasks if t["program"] in args.only]
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(evaluate, t, args.limit): t["program"] for t in tasks}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as exc:  # keep going; report the program as failed
                r = {"program": futs[fut], "error": repr(exc), "failing": []}
            results.append(r)
            fs = r["failing"][:1]
            print(f"{r['program']:28s} " + (f"hit_rel={fs[0]['hit_rel']} hit_dyn={fs[0]['hit_dyn']} "
                                            f"slice={fs[0]['slice_rel']}/{fs[0]['executed']} reach={fs[0]['reach']} "
                                            f"sbfl={fs[0]['sbfl_rank']} hybrid={fs[0]['hybrid_rank']}"
                                            if fs else r.get("error", "no failing run")), flush=True)
    results.sort(key=lambda r: r["program"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    summary = summarize(results)
    json.dump({"summary": summary, "results": results}, open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
