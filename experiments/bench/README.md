# Seeded-bug debugging benchmark (24 tasks)

Each `tasks/<id>/` is a real pure-Python library with ONE seeded single-token bug, in one git commit (files dated 2026-01-01); CPython 3.12.13,
pytest 9.1.1. Ground truth sits beside it, never inside: `<id>.task.json` (fault, failing tests, `setup`, commands), `<id>.gt.diff` (mutant → original).

**Libraries.** Latest release tags, cloned into `upstream/<lib>`, run unmodified on 3.12: tabulate `v0.10.0` (3b4cd509, 2.3 s), sqlparse `0.6.0`
(2f40da99, 2.8 s), mistune `v3.4.0` (f19057fd, 6.4 s); all pass. Kept the two fastest, which also have one test function per behaviour (mistune's 1324
nodes come from fixture files). Pinned test deps: `upstream/<lib>.requirements.txt` (pytest, plus wcwidth = tabulate's `widechars` extra; no numpy/pandas,
as in tox's default env, so 16 tabulate tests skip). The library is never installed: tests run from the checkout root and import it from there. Task
copies drop .git, .github, CHANGELOG and sqlparse's AGENTS.md/CLAUDE.md (agent instructions that say `uv run --python 3.11`).

**Commands** (from this directory; `.venv` is a stdlib-only `uv venv --python 3.12`):

    (cd upstream/<lib> && uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r ../<lib>.requirements.txt)
    .venv/bin/python make_mutants.py generate --lib <lib> && .venv/bin/python make_mutants.py filter --lib <lib> -j 4
    .venv/bin/python build_tasks.py select --results mutants/run1-contaminated     # reproduces the shipped selection, see caveat
    .venv/bin/python build_tasks.py build && .venv/bin/python build_tasks.py verify

**Mutants and filters.** `tokenize` edits one token in place, only inside function bodies of non-test files (never module- or class-level code,
comments, docstrings or f-strings): `<`↔`<=`, `>`↔`>=`, `==`↔`!=`, `+`↔`-`, `and`↔`or`, `True`↔`False`, digit n→n±1 (not 0→−1), drop a unary
`not`. 250 tokens per library are sampled uniformly (seed 20261020). A mutant survives if (a) the package and the mutated module import; (c) the
full suite ends within 60 s (10 s per test); (b) 1–3 nodes fail, nothing fails to collect, and the run leaves the checkout unchanged; (d) the first
failing node, run alone with `--tb=long`, fails and none of its file:line pairs is on the mutated statement; (e) depth ≥ 2, where depth = number of
named function frames below the test function down to the one running the mutated line (genexpr/lambda frames not counted), minimum over all
executions (own `sys.settrace` probe, `tools/bench_probe.py`). Survivors, sampled → a → c → b → d → e: tabulate 250 → 250 → 248 → 50 → 36 → 30 (of 467 eligible tokens);
sqlparse 250 → 250 → 247 → 80 → 69 → 63 (of 717).

**Selection** (`build_tasks.py select`): 12 per library, greedy; at most 2 per function and 1 per line, node id without blanks; rank by depth ≥ 3,
function not yet used, faulty function absent from the failing traceback, least-used operator, failing test not yet used, seeded tie-break. `verify`
copies each task, runs `setup`, checks the library imports from the copy, the failing test fails and the suite passes after `git apply <id>.gt.diff`: 24/24.
**Caveat.** The shipped set was selected from a first filter pass (`mutants/run1-contaminated/`). There, one sqlparse mutant flipped the CLI's
`--in-place` default and rewrote `tests/files/function.sql` in its worker copy, so later mutants in that worker showed one extra failure. Workers
are now restored after every mutant, and the counts above come from the clean pass. All 24 shipped mutants pass a–e on the clean pass with the same
failing test and depth; `select` on it would swap 7 sqlparse picks. task.json of sqlparse-03/07/08 still lists the spurious extra failing test
`tests/test_split.py::test_split_create_function[function.sql]` (hence their false failing-set check in `tasks/verify.json`); the table has clean counts.

| tabulate | fault | operator | failing test (n failing) | depth | sqlparse | fault | operator | failing test (n failing) | depth |
|---|---|---|---|---|---|---|---|---|---|
| tabulate-01 | __init__.py:141 | arith `-`→`+` | test_output.py::test_pipe (2) | 6 | sqlparse-01 | engine/grouping.py:124 | boolop `or`→`and` | test_regressions.py::test_issue489_tzcasts (1) | 6 |
| tabulate-02 | __init__.py:166 | arith `-`→`+` | test_output.py::test_colon_grid (3) | 6 | sqlparse-02 | engine/grouping.py:199 | compare `==`→`!=` | test_regressions.py::test_dont_alias_keywords (2) | 6 |
| tabulate-03 | __init__.py:168 | arith `-`→`+` | test_output.py::test_colon_grid (2) | 6 | sqlparse-03 | engine/grouping.py:386 | bool-const `True`→`False` | test_grouping.py::test_grouping_create_table (1) | 5 |
| tabulate-04 | __init__.py:203 | int+1 `0`→`1` | test_output.py::test_textile (3) | 5 | sqlparse-04 | engine/statement_splitter.py:67 | int+1 `1`→`2` | test_split.py::test_split_multiple_for_loops_in_begin_end (1) | 5 |
| tabulate-05 | __init__.py:297 | bool-const `False`→`True` | test_output.py::test_asciidoc_headerless (1) | 5 | sqlparse-05 | engine/statement_splitter.py:113 | remove-not | test_regressions.py::test_issue193_splitting_function (3) | 4 |
| tabulate-06 | __init__.py:307 | bool-const `True`→`False` | test_output.py::test_asciidoc (1) | 5 | sqlparse-06 | engine/statement_splitter.py:189 | boolop `and`→`or` | test_split.py::test_split_mysql_handler_for (2) | 3 |
| tabulate-07 | __init__.py:943 | remove-not | test_output.py::test_float_conversions (1) | 4 | sqlparse-07 | filters/aligned_indent.py:39 | arith `+`→`-` | test_format.py::TestFormatReindentAligned::test_basic (2) | 10 |
| tabulate-08 | __init__.py:947 | boolop `or`→`and` | test_output.py::test_float_conversions (1) | 4 | sqlparse-08 | filters/aligned_indent.py:87 | arith `-`→`+` | test_format.py::TestFormatReindentAligned::test_case_statement (2) | 15 |
| tabulate-09 | __init__.py:1189 | remove-not | test_output.py::test_preserve_whitespace (3) | 3 | sqlparse-09 | filters/aligned_indent.py:123 | int-1 `3`→`2` | test_format.py::TestFormatReindentAligned::test_group_by (2) | 8 |
| tabulate-10 | __init__.py:1569 | int-1 `1`→`0` | test_input.py::test_list_of_dicts_firstrow (1) | 2 | sqlparse-10 | filters/others.py:46 | bool-const `True`→`False` | test_format.py::TestFormat::test_strip_comments_preserves_hint (1) | 4 |
| tabulate-11 | __init__.py:2801 | boolop `and`→`or` | test_output.py::test_break_long_words (1) | 5 | sqlparse-11 | filters/reindent.py:161 | int-1 `1`→`0` | test_format.py::TestFormatReindent::test_parenthesis (3) | 7 |
| tabulate-12 | __init__.py:2902 | compare `>`→`>=` | test_output.py::test_maxcolwidth_single_value (3) | 4 | sqlparse-12 | filters/reindent.py:228 | int+1 `0`→`1` | test_format.py::TestFormatReindent::test_long_identifier_list_with_functions (1) | 18 |
