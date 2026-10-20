---
name: whyslice
description: Explain WHY a failing Python test or program computes a wrong value. Records one execution of the failing test and returns its causal chain — the executed statements, with the runtime values they read and wrote, that led to the failure, across function calls — including branch decisions that kept a variable from being re-assigned. Use when a pytest test (or a script) fails with a wrong value and the traceback does not show where the value went wrong.
---

# WhySlice: ask why a value is wrong

WhySlice records one run of a failing test and builds its dynamic dependence graph.
You then ask *why* a variable had a value at a line and get the chain of causes with
the values observed in that run. It replaces rounds of adding print statements.

## When to use

- A test fails with a wrong value (assertion mismatch, wrong output) and the faulty
  line is not in the traceback.
- You are about to add prints or a debugger to find where a value went wrong.

Not for syntax or import errors, and never for a whole suite: record one test.

## Workflow

1. Run the single failing test once to see the failure.
2. Record it: MCP tool `record_failure(command="<single-test command>")`,
   or in Bash: `whyslice run -- <single-test command>`.
3. Read the causal tree from the top. Each indented line is a cause of the line above:
   - `x=3 from #12 f.py:40 in g: x = a + b` — `x` got its value at that execution.
   - `value returned 7 by #20 f.py:55 in h: return t` — a callee's return flowed in.
   - `` `y` not re-assigned because (not taken) #31 f.py:70: if a > b: `` — a branch
     decision kept `y` unchanged; check that condition first.
   - `runs because (taken) #18 f.py:66: if ok:` — the line ran because of that branch.
   - `[run 2/5]` — the second of five executions of that line in this run.
4. Find the earliest value that is already wrong and inspect the line that produced it.
   Go deeper with `why(location="f.py:LINE", variable="name", occurrence=N)`
   (`whyslice why f.py:LINE name -n N` in Bash); `occurrence` 1 = first execution,
   0 = last.
5. Fix the faulty line and re-run the test. The recording describes the code as it was
   when recorded: after an edit, record again before asking more questions.
