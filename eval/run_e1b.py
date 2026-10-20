"""E1b: offline slice quality on the seeded bugs in real libraries (the E2 task set).

QuixBugs programs execute ~8 lines, so slice size says little there. Here a
single failing test executes hundreds of lines across many files, which makes
the reduction measurable. For each task: copy, set up its venv, record the
failing test, then measure fault-in-slice (relevant / classic), slice size vs
executed lines, and how early the causal tree reaches the fault.

    python eval/run_e1b.py [--jobs 4]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from whyslice.runner import record  # noqa: E402

TASKS = os.path.join(REPO, "experiments", "bench", "tasks")


def evaluate(task_id: str) -> dict:
    task = json.load(open(os.path.join(TASKS, task_id + ".task.json")))
    tmp = tempfile.mkdtemp(prefix="e1b-")
    work = os.path.join(tmp, task_id)
    try:
        shutil.copytree(os.path.join(TASKS, task_id), work, symlinks=True)
        setup = task["setup"].replace("{bench}", os.path.join(REPO, "experiments", "bench"))
        subprocess.run(setup, shell=True, cwd=work, check=True, capture_output=True, timeout=600)
        rec = record(f"pytest -q -p no:cacheprovider {task['failing_test']}", cwd=work,
                     python=os.path.join(work, ".venv", "bin", "python"), timeout=300,
                     out=os.path.join(tmp, "t.pkl"), root=work)
        t = rec.trace
        if t is None:
            return {"task": task_id, "error": rec.error}
        seeds = t.failure_seeds()
        fault_path = os.path.realpath(os.path.join(work, task["fault_file"]))
        fidx = next((i for i, p in enumerate(t.files) if os.path.realpath(p) == fault_path), None)
        unit = t.line_to_unit.get(fidx, {}).get(task["fault_line"], task["fault_line"]) if fidx is not None else None
        rel = t.lines_of(t.backward(seeds, True) | set(seeds))
        dyn = t.lines_of(t.backward(seeds, False) | set(seeds))
        executed = t.executed_lines()
        lib_exec = {(f, ln) for f, ln in executed if not t.short(f).startswith("tests")}
        lib_rel = {(f, ln) for f, ln in rel if not t.short(f).startswith("tests")}
        fault = (fidx, unit)
        order = t.causal_order(seeds[0]) if seeds else []
        seen = []
        reach = None
        for iid in order:
            ti = t.insts[iid]
            loc = (ti[2], ti[3])
            if ti[4] == "entry" or t.short(ti[2]).startswith("tests"):
                continue
            if loc not in seen:
                seen.append(loc)
            if loc == fault:
                reach = len(seen)
                break
        shown = t._tree(seeds[0], t.edges(seeds[0]), 30) if seeds else {}
        in_tree = any((t.insts[c][2], t.insts[c][3]) == fault for kids in shown.values() for c, _l, _n in kids)
        sus_rank = None
        if rec.spectrum:
            t.suspects(rec.spectrum)
            ranking = [loc for _s, loc in getattr(t, "_last_ranking", [])]
            sus_rank = ranking.index(fault) + 1 if fault in ranking else None
        in_report = in_tree or (sus_rank is not None and sus_rank <= 10)
        return {"task": task_id, "operator": task.get("operator"), "depth": task.get("depth"),
                "fault_executed": fault in executed, "hit_rel": fault in rel, "hit_dyn": fault in dyn,
                "in_report": in_report, "in_tree": in_tree, "suspect_rank": sus_rank, "reach": reach,
                "executed_lines": len(executed), "lib_executed_lines": len(lib_exec),
                "slice_lines": len(rel), "lib_slice_lines": len(lib_rel), "classic_slice_lines": len(dyn),
                "instances": sum(1 for x in t.insts if x[4] != "entry"), "files": len(t.files),
                "record_s": t.data.get("elapsed"), "truncated": bool(t.data.get("truncated")),
                "recorder_errors": len(t.data.get("recorder_errors") or [])}
    except Exception as exc:  # report and continue
        return {"task": task_id, "error": repr(exc)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "e1b.json"))
    args = ap.parse_args()
    ids = sorted(f[: -len(".task.json")] for f in os.listdir(TASKS) if f.endswith(".task.json"))
    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for fut in as_completed([ex.submit(evaluate, i) for i in ids]):
            r = fut.result()
            rows.append(r)
            print(json.dumps({k: r.get(k) for k in ("task", "hit_rel", "hit_dyn", "in_report", "suspect_rank", "reach",
                                                     "lib_slice_lines", "lib_executed_lines", "error")}), flush=True)
    rows.sort(key=lambda r: r["task"])
    ok = [r for r in rows if "error" not in r]

    def frac(k):
        return sum(1 for r in ok if r[k]) / len(ok) if ok else 0.0

    summary = {
        "tasks": len(rows), "evaluated": len(ok),
        "fault_executed": frac("fault_executed"),
        "fault_in_slice_relevant": frac("hit_rel"), "fault_in_slice_classic": frac("hit_dyn"),
        "fault_in_default_report": frac("in_report"),
        "fault_in_causal_tree": frac("in_tree"),
        "suspect_top1": sum(1 for r in ok if r.get("suspect_rank") == 1),
        "suspect_top5": sum(1 for r in ok if r.get("suspect_rank") and r["suspect_rank"] <= 5),
        "suspect_top10": sum(1 for r in ok if r.get("suspect_rank") and r["suspect_rank"] <= 10),
        "median_lib_executed_lines": statistics.median([r["lib_executed_lines"] for r in ok]) if ok else None,
        "median_lib_slice_lines": statistics.median([r["lib_slice_lines"] for r in ok]) if ok else None,
        "median_slice_ratio": statistics.median([r["lib_slice_lines"] / r["lib_executed_lines"]
                                                 for r in ok if r["lib_executed_lines"]]) if ok else None,
        "median_reach_when_in_slice": statistics.median([r["reach"] for r in ok if r["reach"]])
        if any(r["reach"] for r in ok) else None,
        "median_record_s": statistics.median([r["record_s"] for r in ok if r["record_s"]]) if ok else None,
        "by_library": {lib: {"n": len(rs), "hit_rel": sum(r["hit_rel"] for r in rs)}
                       for lib in ("sqlparse", "tabulate")
                       for rs in [[r for r in ok if r["task"].startswith(lib)]]},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
