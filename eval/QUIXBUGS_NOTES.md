# QuixBugs (Python) as whyslice evaluation input

- Source: github.com/jkoppel/QuixBugs, shallow clone in `eval/data/QuixBugs` at `4257f44b0ff1181dedaedee6a447e133219fcebf` (2022-08-29). whyslice's `.gitignore` already covers `eval/data/` and `eval/.venv-*/`.
- Environment: `uv venv eval/.venv-quixbugs --python 3.12` (CPython 3.12.13), then `uv pip install --python eval/.venv-quixbugs/bin/python pytest pytest-timeout` (pytest 9.1.1, pytest-timeout 2.4.0). Never use the system `python3` (3.9).
- Regenerate `quixbugs_tasks.json`: `cd eval && .venv-quixbugs/bin/python build_quixbugs_tasks.py [--verify-isolated]`. A run takes about 13 min, mostly re-running the timeouts at 30 s; the flag adds about 3 min.

## Running a test (cwd = `eval/data/QuixBugs`)
- One node id against the buggy version (`test_cmd_template`): `../../.venv-quixbugs/bin/python -m pytest -c /dev/null --rootdir=. --confcutdir=. -p no:cacheprovider -q --timeout=5 --timeout-disable-debugger-detection '<node_id>'`
- Correct version: add `--correct`. The repo's `conftest.py` defines it, and each `python_testcases/test_<p>.py` then imports `correct_python_programs.<p>` instead of `python_programs.<p>`. `--runslow` un-skips the 4-minute knapsack case.
- `-c /dev/null --rootdir=. --confcutdir=.` shut out pytest config from the directories above the clone. Without them, the `whyslice/pyproject.toml` that appeared mid-run became pytest's rootdir, and every reported node id turned into `eval/data/QuixBugs/python_testcases/...`. A future `[tool.pytest.ini_options]` or `whyslice/conftest.py` would leak in the same way.
- `--timeout-disable-debugger-detection` is needed because pytest-timeout silently skips the timeout when `sys.gettrace()` belongs to a module named `bdb*` or `pydevd*`, so a bdb-based tracer would hang on the infinite loops.
- Inputs: 31 programs read `json_testcases/<p>.json` (one `[args, expected]` per line; parametrized ids look like `input_dataN-<expected>` or `input_dataN-expectedN`). The 9 graph programs build their inputs inside the test functions: 6 use `Node` from `python_testcases/node.py`, and minimum_spanning_tree, shortest_path_lengths and shortest_paths use plain dicts. `python_programs/node.py` and the `*_test.py` drivers belong to the old `tester.py` and are not used.

## Fault lines
- Normalization: drop comments, blank lines and every bare string-expression statement. That covers docstrings and the trailing triple-quoted blocks in which the correct files keep alternative implementations (three extra `breadth_first_search` defs, for example). Then diff the remaining lines, each keyed as (indent width, token strings), with `difflib`.
- Every program gives exactly one hunk: 36 are `modified`, and 4 are `missing`, whose anchors are the code lines just before and after the insertion: depth_first_search [9,10], reverse_linked_list [5,6], shunting_yard [17,19], wrap [8,10].
- Checked by hand, with both buggy and correct files read in full (21 programs): breadth_first_search, bucketsort, depth_first_search, detect_cycle, find_first_in_sorted, get_factors, is_valid_parenthesization, kth, lis, minimum_spanning_tree, next_palindrome, possible_change, quicksort, reverse_linked_list, rpn_eval, shortest_path_length, shortest_path_lengths, shortest_paths, shunting_yard, to_base, wrap.
- `fault_stmt_lines` is the first line of the innermost statement that contains each fault line. It differs from the fault line in 4 programs: rpn_eval (20 to 19), shortest_path_length (22 to 19) and shortest_path_lengths (13 to 11), where the fault sits inside a multi-line call, and depth_first_search (9 to 7), where the anchor is an `else:` of the `elif` statement.
- `fault_line_hits` counts the failing and timeout tests whose run emits a sys.monitoring LINE event on each fault line. Every fault line is hit by every failing test except two `missing` anchors that no dynamic slice can contain: depth_first_search L9 (an `else:` has no bytecode) and shunting_yard L17 (the inner loop body never runs in the buggy version).
- `alt_fault_lines` lists alternatives from the correct files' string blocks that change a different line and pass every test: get_factors L6, quicksort L6, to_base L10, reverse_linked_list L6, is_valid_parenthesization (insertion between L10 and L12) and possible_change (insertion between L4 and L5). Two were rejected: bucketsort's fails 4 tests, and rpn_eval's relies on an undefined `Stack`.

## Test outcomes (buggy at 5 s per test, correct at 60 s, timeouts re-run at 30 s)
- 278 node ids: 170 fail, 17 time out, 89 pass and 2 are skipped. The correct version passes all 276 runnable tests; its slowest takes 1.0 s. Outcomes were identical across two batch runs and a run of each node id alone through the template.
- Infinite loops, so expect truncated traces: bitcount (9/9 tests) and sqrt (6/7) fail only by timeout, and find_first_in_sorted has 2 timeouts plus 1 IndexError. All 17 still time out at 30 s.
- Deep recursion, ending in RecursionError at about 1000 frames: gcd (5 tests), mergesort (13), find_in_sorted (2), depth_first_search (1).
- Seven programs have no passing test, which degenerates spectrum-based baselines: bitcount, minimum_spanning_tree, next_permutation, shortest_path_lengths, shortest_paths, topological_ordering, wrap.
- The test files skip `test_knapsack[input_data9-13549094]` (unless `--runslow`) and `test_levenshtein[input_data3-42]` (always).
- The slowest untraced cases come from exponential recursion: levenshtein input_data2 (1.0 s) and longest_common_subsequence input_data3 (0.2 s). Give them extra time when tracing.
- There is no randomness, and no program has many tests (at most 14). flatten's failure messages contain memory addresses, and shortest_paths mutates its input dict.
