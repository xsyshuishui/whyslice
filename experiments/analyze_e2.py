"""Summarize E2 runs: per-arm outcomes and paired comparisons against A0.

    python experiments/analyze_e2.py [experiments/runs] [--out experiments/results/e2.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from collections import defaultdict
from typing import Dict, List, Optional


def load(root: str) -> List[dict]:
    rows = []
    for path in glob.glob(os.path.join(root, "*", "*", "result.json")):
        row = json.load(open(path))
        transcript = os.path.join(os.path.dirname(path), "transcript.jsonl")
        row.update(session_totals(transcript) or {})
        row["first_fault_edit_turn"] = first_fault_edit(transcript, row)
        rows.append(row)
    return rows


def session_totals(transcript: str) -> Optional[dict]:
    """Sum turns and usage over every result event.

    A session that leaves a background command running is resumed when it finishes and emits a
    second result event; run_agent kept only the last one, which undercounts that run. Usage and
    num_turns are per invocation, total_cost_usd is cumulative.
    """
    if not os.path.exists(transcript):
        return None
    results = []
    for line in open(transcript):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "result":
            results.append(ev)
    if len(results) < 2:
        return None
    usage = [ev.get("usage") or {} for ev in results]
    return {"num_turns": sum(ev.get("num_turns") or 0 for ev in results),
            "input_tokens": sum(u.get("input_tokens") or 0 for u in usage),
            "output_tokens": sum(u.get("output_tokens") or 0 for u in usage),
            "cache_read": sum(u.get("cache_read_input_tokens") or 0 for u in usage),
            "cache_write": sum(u.get("cache_creation_input_tokens") or 0 for u in usage),
            "cost_usd": results[-1].get("total_cost_usd"), "result_events": len(results)}


def first_fault_edit(transcript: str, row: dict) -> Optional[int]:
    task_json = None
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench", "tasks", row["task"])
    for cand in (base + ".task.json", os.path.join(base, "task.json")):
        if os.path.exists(cand):
            task_json = json.load(open(cand))
            break
    if not task_json or not os.path.exists(transcript):
        return None
    fault_file = task_json["fault_file"]
    turn = 0
    for line in open(transcript):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") != "assistant":
            continue
        turn += 1
        for block in ev.get("message", {}).get("content", []):
            if block.get("type") == "tool_use" and block.get("name") in ("Edit", "Write", "MultiEdit"):
                if str(block.get("input", {}).get("file_path", "")).endswith(fault_file):
                    return turn
    return None


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test p-value."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def perm_test(deltas: List[float], iters: int = 20000, seed: int = 0) -> Optional[float]:
    """Two-sided sign-flip permutation test for mean(deltas) == 0."""
    import random

    if not deltas:
        return None
    rng = random.Random(seed)
    obs = abs(sum(deltas) / len(deltas))
    hits = 0
    for _ in range(iters):
        m = abs(sum(d if rng.random() < 0.5 else -d for d in deltas) / len(deltas))
        if m >= obs - 1e-12:
            hits += 1
    return hits / iters


def bootstrap_ci(deltas: List[float], iters: int = 20000, seed: int = 1) -> Optional[List[float]]:
    import random

    if not deltas:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(iters))
    return [round(means[int(0.025 * iters)], 3), round(means[int(0.975 * iters)], 3)]


def med(vals: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def tokens(r: dict) -> Optional[int]:
    parts = [r.get("input_tokens"), r.get("output_tokens"), r.get("cache_read"), r.get("cache_write")]
    return sum(p for p in parts if p) if any(parts) else None


def summarize(rows: List[dict]) -> dict:
    by_arm: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        if "error" not in r:
            by_arm[r["arm"]].append(r)
    arms = {}
    for arm, rs in sorted(by_arm.items()):
        used = [r for r in rs if any("whyslice" in k for k in (r.get("tools") or {}))]
        arms[arm] = {
            "runs": len(rs),
            "success_rate": sum(r["success"] for r in rs) / len(rs),
            "successes": sum(r["success"] for r in rs),
            "median_turns": med([r.get("num_turns") or r.get("turns") for r in rs]),
            "median_tokens": med([tokens(r) for r in rs]),
            "median_output_tokens": med([r.get("output_tokens") for r in rs]),
            "median_cost_usd": med([r.get("cost_usd") for r in rs]),
            "median_wall_s": med([r.get("wall_s") for r in rs]),
            "median_first_fault_edit_turn": med([r.get("first_fault_edit_turn") for r in rs]),
            "runs_calling_whyslice": len(used),
            "runs_with_push": sum(1 for r in rs if (r.get("hook_pushes") or 0) > 0),
            "whyslice_calls_total": sum(v for r in rs for k, v in (r.get("tools") or {}).items() if "whyslice" in k),
            "edited_fault_line": sum(1 for r in rs if r.get("edited_fault_line")),
            "touched_tests": sum(1 for r in rs if r.get("touched_tests")),
        }
    # paired by (task, seed)
    key = lambda r: (r["task"], r["seed"])  # noqa: E731
    others = [a for a in sorted(by_arm) if a != "A0"]  # A1, A2 and any ablation arms
    base = {key(r): r for r in by_arm.get("A0", [])}
    paired = {}
    for arm in others:
        wins = losses = 0
        dturns, dtok = [], []
        for r in by_arm.get(arm, []):
            b = base.get(key(r))
            if b is None:
                continue
            if r["success"] and not b["success"]:
                wins += 1
            elif b["success"] and not r["success"]:
                losses += 1
            if r["success"] and b["success"]:
                t1, t0 = (r.get("num_turns") or r.get("turns")), (b.get("num_turns") or b.get("turns"))
                if t1 is not None and t0 is not None:
                    dturns.append(t1 - t0)
                k1, k0 = tokens(r), tokens(b)
                if k1 and k0:
                    dtok.append(k1 / k0)
        paired[arm + "_vs_A0"] = {"success_wins": wins, "success_losses": losses,
                                  "sign_test_p": round(sign_test(wins, losses), 4),
                                  "median_turn_delta_both_solved": med(dturns),
                                  "median_token_ratio_both_solved": med(dtok),
                                  "pairs_both_solved": len(dturns)}
    # task-level paired comparison: average each task over its seeds first (n = tasks)
    task_level = {}
    for arm in others:
        per_task_turns: Dict[str, List[float]] = defaultdict(list)
        per_task_logtok: Dict[str, List[float]] = defaultdict(list)
        per_task_wall: Dict[str, List[float]] = defaultdict(list)
        for r in by_arm.get(arm, []):
            b = base.get(key(r))
            if b is None or not (r["success"] and b["success"]):
                continue
            t1, t0 = (r.get("num_turns") or r.get("turns")), (b.get("num_turns") or b.get("turns"))
            if t1 is not None and t0 is not None:
                per_task_turns[r["task"]].append(t1 - t0)
            k1, k0 = tokens(r), tokens(b)
            if k1 and k0:
                per_task_logtok[r["task"]].append(math.log(k1 / k0))
            if r.get("wall_s") and b.get("wall_s"):
                per_task_wall[r["task"]].append(r["wall_s"] - b["wall_s"])
        dt = [sum(v) / len(v) for v in per_task_turns.values()]
        dk = [sum(v) / len(v) for v in per_task_logtok.values()]
        dw = [sum(v) / len(v) for v in per_task_wall.values()]
        task_level[arm + "_vs_A0"] = {
            "tasks": len(dt),
            "mean_turn_delta": round(sum(dt) / len(dt), 3) if dt else None,
            "turn_delta_ci95": bootstrap_ci(dt), "turn_delta_p": perm_test(dt),
            "tasks_fewer_turns": sum(1 for d in dt if d < 0), "tasks_more_turns": sum(1 for d in dt if d > 0),
            "geo_mean_token_ratio": round(math.exp(sum(dk) / len(dk)), 3) if dk else None,
            "log_token_ratio_ci95": bootstrap_ci(dk), "token_p": perm_test(dk),
            "mean_wall_delta_s": round(sum(dw) / len(dw), 2) if dw else None, "wall_p": perm_test(dw),
        }

    # mechanism: does the report's content matter? split tasks by whether the E1b report
    # (suspects top-10 or causal tree) contained the fault line
    e1b_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval", "results", "e1b.json")
    split = {}
    if os.path.exists(e1b_path):
        info = {r["task"]: r for r in json.load(open(e1b_path))["rows"] if "error" not in r}
        for arm in others:
            for flag in (True, False):
                d_turns, d_tok, n = [], [], 0
                task_t: Dict[str, List[float]] = defaultdict(list)
                task_k: Dict[str, List[float]] = defaultdict(list)
                for r in by_arm.get(arm, []):
                    if info.get(r["task"], {}).get("in_report") is not flag:
                        continue
                    b = base.get(key(r))
                    if b is None or not (r["success"] and b["success"]):
                        continue
                    n += 1
                    t1, t0 = (r.get("num_turns") or r.get("turns")), (b.get("num_turns") or b.get("turns"))
                    if t1 is not None and t0 is not None:
                        d_turns.append(t1 - t0)
                        task_t[r["task"]].append(t1 - t0)
                    k1, k0 = tokens(r), tokens(b)
                    if k1 and k0:
                        d_tok.append(k1 / k0)
                        task_k[r["task"]].append(math.log(k1 / k0))
                tt = [sum(v) / len(v) for v in task_t.values()]
                tk = [sum(v) / len(v) for v in task_k.values()]
                split[f"{arm}_vs_A0_fault_in_report={flag}"] = {
                    "pairs": n, "median_turn_delta": med(d_turns), "mean_turn_delta":
                    (sum(d_turns) / len(d_turns)) if d_turns else None, "median_token_ratio": med(d_tok),
                    "tasks": len(tt), "task_mean_turn_delta": round(sum(tt) / len(tt), 3) if tt else None,
                    "task_turn_ci95": bootstrap_ci(tt), "task_turn_p": perm_test(tt),
                    "task_geo_token_ratio": round(math.exp(sum(tk) / len(tk)), 3) if tk else None,
                    "task_log_token_ci95": bootstrap_ci(tk), "task_token_p": perm_test(tk)}
    return {"arms": arms, "paired": paired, "task_level": task_level, "by_report_content": split}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=os.path.join(os.path.dirname(__file__), "runs"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "e2.json"))
    args = ap.parse_args()
    rows = load(args.root)
    summary = summarize(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    per_task = defaultdict(dict)
    for r in rows:
        per_task[r["task"]][f'{r["arm"]}s{r["seed"]}'] = ("✓" if r.get("success") else "✗") + \
            f'{r.get("num_turns") or r.get("turns") or "?"}'
    for task in sorted(per_task):
        print(f"{task:16s} " + "  ".join(f"{k}:{v}" for k, v in sorted(per_task[task].items())))


if __name__ == "__main__":
    main()
