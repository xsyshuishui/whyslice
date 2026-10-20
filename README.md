# WhySlice

On-demand dynamic causal slicing for coding agents.

When a Python test fails, a coding agent (Claude Code, Codex, Cursor, …) can ask
WhySlice *why a variable had a value at a line*. WhySlice answers with the chain of
statements that caused the value, across function calls, with the runtime values from
that run. The engine uses only the Python standard library.

```text
$ whyslice run -- pytest test_cart.py::test_member_at_tier_threshold
FAILED test_cart.py::test_member_at_tier_threshold: assert 170.0 == 160.0
value returned 170.0 by #47 cart.py:21
  disc=30.0 from #30 cart.py:20
    value returned 30.0 by #46 pricing.py:16
      rate=0.1 from #32 pricing.py:13
        value returned 0.1 by #43 pricing.py:9
          `rate` not re-assigned because (not taken)
          #39 pricing.py:7: if amount > threshold:
              ⟹ not taken with amount=200, threshold=200
```

The bug is on the last line: the tier check should be `>=`. The traceback only points at the
test. The last two lines come from a *potential dependence*: the branch was not taken, so
`rate` kept its old value. A classic dynamic slice cannot show a statement that did not run.

## How it works

1. **Record.** Run one failing test under `sys.settrace` in the project's own interpreter. Every
   execution of every statement becomes a statement instance.
2. **Dynamic dependence graph.** Each instance gets data, control, return-value and exception
   dependences. Memory is tracked at three levels: name bindings, object attributes, and
   container elements. Arguments map to parameters, so aliasing across calls is followed.
3. **Potential dependences** (relevant slicing). If a predicate had taken its other branch and
   would then have assigned the variable, WhySlice adds an edge from the predicate. This is
   how it finds execution-omission bugs like the one above.
4. **Backward slice** from the question `(file:line, variable, occurrence)`; by default, from the
   failing assertion or exception.
5. **Suspicious lines.** Lines that ran up to the failure are ranked by Ochiai over the other
   tests in the same test module (spectrum-based fault localization). Lines in the slice win ties.
   Code that runs only to format the assertion message, such as `__repr__` of the compared
   objects, is left out.
6. **Agent-sized report.** The ten most suspicious lines, then a best-first causal tree of about
   20–30 nodes with values. Plumbing edges are hidden, loops are folded, and the slice is
   listed by function at the end. The agent can then ask `why` about any line in the report.
   The dependence graph is kept, so follow-up questions do not re-run the program.

## Install

Python ≥ 3.10.

```bash
pip install -e '.[mcp]'      # the engine needs no dependencies; the MCP server needs `mcp`
```

WhySlice runs the target in the target project's own interpreter (its `.venv`, or the
interpreter of the `pytest` it finds) and injects itself through `PYTHONPATH`. You do not
need to install it into the project under test.

## Use

**CLI**

```bash
whyslice run -- pytest tests/test_x.py::test_y   # record one failing test and explain it
whyslice why src/mod.py:42 total                 # why does `total` have its value at line 42?
whyslice why src/mod.py:42 total -n 3            # ... at its 3rd execution
whyslice lines src/mod.py:42 total               # the slice for that question, as file:line
```

**MCP server.** It exposes two tools: `record_failure(command, cwd)` and
`why(location, variable, occurrence)`. To add it to Claude Code:

```bash
claude mcp add whyslice -- /path/to/whyslice/.venv/bin/whyslice-mcp
```

**Skill.** Copy `plugin/skills/whyslice/` into `.claude/skills/`. The skill tells the agent when to
use WhySlice and how to read its output.

**Hook (push).** This hook records a failing pytest test automatically and adds the report to the
agent's context. Register it for both events, because a failing pytest exits non-zero and Claude Code
reports that as `PostToolUseFailure`:

```json
{"hooks": {
  "PostToolUse":        [{"matcher": "Bash", "hooks": [{"type": "command", "command": "/path/to/.venv/bin/python /path/to/whyslice/plugin/hooks/push_failure.py"}]}],
  "PostToolUseFailure": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "/path/to/.venv/bin/python /path/to/whyslice/plugin/hooks/push_failure.py"}]}]
}}
```

The hook pushes at most two reports per test and four per session. Set
`WHYSLICE_PUSH_SECTIONS=suspects` to push only the suspicious lines. This is the cheapest
setting; see below.

## Evaluation

All numbers below are in `eval/results/` and `experiments/results/`.

**E1, QuixBugs (40 Python programs, no language model).** The fault line is in the relevant slice for
38/40 programs. The classic dynamic slice gets 37/40; the extra one, `sieve`, needs the potential
dependence. The two misses write to the wrong dictionary (`shortest_paths`) or lack a statement
(`shunting_yard`). No dynamic slice can contain either.

**E1b, real libraries.** The benchmark has 24 seeded single-token bugs in tabulate 0.10.0 and
sqlparse 0.6.0. Each bug fails 1–3 tests, is not in the traceback, and runs at least two calls below the
test; see `experiments/bench/README.md`.
- The fault line is in the relevant slice for 20/24 bugs. The slice is about half of the executed
  library lines.
- Causal chains are long, about 180 distinct lines before the fault in tree order, and the causal tree
  never reaches the fault. The fault is among the top 10 suspicious lines for 14/24 bugs, at rank 1
  for 7. (The version used in E2 got 13/24; see the note under E2.) The slice listing at the end
  of the report (lines by function) contains it for 17/24, but does not say which line is suspect.

**E2, coding agents.** Claude Code fixed the 24 bugs under the same prompt in three arms:
- A0: built-in tools only;
- A1: plus the MCP server and skill, used at the agent's discretion;
- A2: plus the push hook.

A run succeeds when the failing test passes, the whole suite passes, and no test file was edited.
The table compares task-level means of conversation turns with A0. p comes from a sign-flip
permutation test over tasks.

| Model (runs per task × arm) | Success A0 / A1 / A2 | A1 uses the tool | Turns, A1 vs A0 | Turns, A2 vs A0 |
|---|---|---|---|---|
| Claude Haiku 5.5 (×3) | 72 / 72 / 72 of 72 | 18% of runs | −0.4 (p = 0.21) | **−1.5 (p < 0.001)** |
| Claude Haiku 4.5 (×1) | 22 / 24 / 24 of 24 | 83% of runs | +0.1 (p = 0.96) | **−4.6 (p = 0.016)** |
| DeepSeek flash (×1) | 24 / 24 / 24 of 24 | 4% of runs | −0.5 (p = 0.75) | **−3.0 (p = 0.045)** |

Ablations on Haiku 5.5 (×3), each compared with A0:
- push only the suspicious lines: −1.5 turns (p = 0.008), and 16% fewer tokens than pushing the full report;
- push only the causal tree: −1.5 turns (p < 0.001);
- on-demand, but told to call `record_failure` right after the failure: −1.1 turns (p = 0.065).

What this shows, and what it does not:
- **Push helps; pull at the agent's discretion does not.** When the agent decides, it calls late or not
  at all. When it is told to call first, it does about as well as push, though only at p = 0.065.
- **Either half of the report is enough.** The suspicious lines and the causal tree each reduce turns
  by the same amount as the full report.
- **Costs.** Total tokens and wall time did not drop. On Haiku 5.5, pushing the full report raised the
  API-priced cost by 14%, because the report is written to the prompt cache. Pushing only the suspicious
  lines avoids most of this.
- **Limits of the benchmark.** The bugs are easy: success rates are at or near the ceiling. They are
  single-token mutants in two libraries that the models may have seen. A placebo arm that pushes only
  the failure summary was not run.
- **The E2 runs used a version with a bug in the suspicious lines.** That version cut the failing
  run where the failing statement began. When the failing assert itself calls the code under test
  (`assert f(x) == y`), all of that code was left out of the ranking. For 5 sqlparse tasks the
  list was empty, so those reports had no suspicious lines, and the suspicious-lines-only arm
  pushed just the failure summary. The bug is fixed: the cut is now where the exception is raised,
  and code that only formats the assertion message is left out. E1b above uses the fixed version.
  E2 was not re-run.
  - On those 5 tasks no push variant helped much: tree only gave −0.3 turns.
  - So the comparison of the two halves rests on the other 19 tasks, where suspicious lines only
    gave −2.0 turns and tree only −1.8.

## Reproduce

```bash
pytest tests                                   # 18 unit tests
python eval/run_e1.py                          # E1; needs QuixBugs, see eval/QUIXBUGS_NOTES.md
python eval/run_e1b.py                         # E1b; needs the benchmark, see experiments/bench/README.md
python experiments/run_agent.py --arms A0 A1 A2 --seeds 3 --model claude-haiku-5-5 --out experiments/runs
python experiments/analyze_e2.py experiments/runs --out experiments/results/e2.json
```

`run_agent.py` copies each task to a scratch directory and runs `claude -p` with a fixed tool set,
a sandbox, and a 60-turn limit. Personal settings, memory and outside instruction files are kept
out of each run. For a non-Claude model behind an Anthropic-compatible endpoint, add
`--base-url <url> --key-env <ENV_VAR_WITH_KEY>`.

## Limitations

- Writes to the wrong location and missing statements never appear in a dynamic slice.
  Potential dependences are computed within one frame, so execution omissions that span function
  calls are missed.
- Tracing every line is slow: about 300–600× on tight pure-Python loops. Record one failing test,
  not a suite. A failing test usually executes a few thousand statements; recording takes
  1.4–4.5 s on the benchmark, one task at a time.
- Single-threaded CPython only. Data flow inside C extensions is approximated as "the whole object
  changed".

## License

MIT
