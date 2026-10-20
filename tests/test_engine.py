"""Behavioural tests for the recorder and the slicer, on small programs."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from whyslice.trace import Trace

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def record(tmp_path, files: dict, target: list, limit: int = 300_000) -> Trace:
    for name, body in files.items():
        (tmp_path / name).write_text(textwrap.dedent(body))
    out = tmp_path / "run.pkl"
    env = dict(os.environ, PYTHONPATH=SRC)
    subprocess.run([sys.executable, "-m", "whyslice.record", "--out", str(out), "--root", str(tmp_path),
                    "--limit", str(limit), "--", *target], cwd=tmp_path, env=env, capture_output=True,
                   text=True, timeout=60)
    return Trace.load(str(out))


def slice_lines(t: Trace, file: str, line: int, var=None, occurrence=-1, potential=True):
    iid = t.find(file, line, occurrence)
    return {ln for f, ln in t.lines_of(t.backward(t.seeds(iid, var), potential) | {iid})
            if t.short(f) == file}


def test_data_flow_across_calls_and_omission_error(tmp_path):
    t = record(tmp_path, {
        "prog.py": """\
            def rate_for(amount):
                rate = 0.0
                for limit, r in [(100, 0.1), (200, 0.2)]:
                    if amount > limit:
                        rate = r
                return rate

            def price(amount):
                unused = amount * 2
                return amount * (1 - rate_for(amount))

            assert price(200) == 160
        """}, ["prog.py"])
    f = t.failure()
    assert f and f["exc"].startswith("AssertionError")
    report = t.failure_report()
    assert "not re-assigned because (not taken)" in report and "prog.py:4" in report
    lines = slice_lines(t, "prog.py", 6, "rate")
    # rate = 0.0 (line 2) was overwritten by the first iteration, so it is not a cause
    assert {3, 4, 5} <= lines and 2 not in lines and 9 not in lines
    # classic dynamic slicing misses the omission; relevant slicing keeps the predicate
    classic = slice_lines(t, "prog.py", 6, "rate", potential=False)
    assert 4 in classic  # still reached through the first iteration's control dependence


def test_alias_mutation_is_tracked(tmp_path):
    t = record(tmp_path, {
        "prog.py": """\
            a = []
            b = a
            b.append(1)
            c = 5
            assert a == []
        """}, ["prog.py"])
    assert 3 in slice_lines(t, "prog.py", 5)
    assert 4 not in slice_lines(t, "prog.py", 5)


def test_attribute_level_mutations(tmp_path):
    t = record(tmp_path, {
        "prog.py": """\
            class Box:
                def __init__(self):
                    self.a = 1
                    self.b = 2
                def bump(self):
                    self.b += 10
            x = Box()
            x.bump()
            y = x.a
            assert y == 0
        """}, ["prog.py"])
    lines = slice_lines(t, "prog.py", 9, "x.a")
    assert 3 in lines and 6 not in lines


def test_guard_clause_control_dependence(tmp_path):
    t = record(tmp_path, {
        "prog.py": """\
            def f(x):
                if x < 0:
                    return 0
                y = x * 2
                return y
            assert f(3) == 7
        """}, ["prog.py"])
    iid = t.find("prog.py", 4)
    parent = t.insts[iid][7]
    assert t.insts[parent][3] == 2  # control parent is the guard `if x < 0`


def test_exception_from_callee(tmp_path):
    t = record(tmp_path, {
        "prog.py": """\
            def get(d, k):
                return d[k]
            def main():
                d = {"a": 1}
                key = "b"
                return get(d, key)
            main()
        """}, ["prog.py"])
    f = t.failure()
    assert f["exc"].startswith("KeyError")
    assert t.insts[f["origin"]][3] == 2
    assert 5 in slice_lines(t, "prog.py", 2)


def test_generator_values_flow_to_consumer(tmp_path):
    t = record(tmp_path, {
        "prog.py": """\
            def gen(n):
                for i in range(n):
                    yield i * 3
            total = 0
            for v in gen(3):
                total += v
            assert total == 0
        """}, ["prog.py"])
    lines = slice_lines(t, "prog.py", 7)
    assert {3, 6} <= lines


def test_loop_is_folded_in_tree(tmp_path):
    t = record(tmp_path, {
        "prog.py": """\
            s = 0
            for i in range(50):
                s += i
            assert s == 1
        """}, ["prog.py"])
    assert "⟲" in t.failure_report()


def test_closure_variable(tmp_path):
    t = record(tmp_path, {
        "prog.py": """\
            def outer():
                k = 4
                def inner(x):
                    return x + k
                return inner(1)
            assert outer() == 0
        """}, ["prog.py"])
    assert 2 in slice_lines(t, "prog.py", 4)


def test_instance_limit_stops_infinite_loop(tmp_path):
    t = record(tmp_path, {
        "prog.py": """\
            i = 0
            while i < 10:
                i = i * 1
        """}, ["prog.py"], limit=2_000)
    assert t.data["truncated"]
    assert "prog.py:3" in t.hot_lines()


def test_pytest_failure_origin(tmp_path):
    t = record(tmp_path, {
        "lib.py": """\
            def area(w, h):
                return w + h
        """,
        "test_lib.py": """\
            from lib import area
            def test_area():
                assert area(2, 3) == 6
        """}, ["pytest", "-q", "test_lib.py"])
    f = t.failure()
    assert f["test"].endswith("test_lib.py::test_area")
    assert "lib.py:2" in t.failure_report()


@pytest.mark.parametrize("command,expected", [
    ("python -m pytest t.py::x", ("python", ["-m", "pytest", "t.py::x"], None)),
    ("python prog.py < in.txt", ("python", ["prog.py"], "in.txt")),
    ("pytest -q t.py", (None, ["pytest", "-q", "t.py"], None)),
])
def test_parse_command(command, expected):
    from whyslice.runner import parse_command
    assert parse_command(command) == expected


def test_item_level_dependences(tmp_path):
    t = record(tmp_path, {
        "prog.py": """\
            d = {}
            for i in range(3):
                d[i] = i * 10
            d[1] = 99
            x = d[2]
            assert x == 0
        """}, ["prog.py"])
    lines = slice_lines(t, "prog.py", 5, "d[2]")
    assert 3 in lines and 4 not in lines  # d[1] = 99 does not feed d[2]


def test_recursion_error_still_explained(tmp_path):
    t = record(tmp_path, {
        "prog.py": """\
            def down(n):
                return down(n - 1)
            down(5)
        """}, ["prog.py"])
    assert t.failure()["exc"].startswith("RecursionError")
    assert "prog.py:2" in t.failure_report()


def test_predicate_reading_subscript_with_potential_dependence(tmp_path):
    t = record(tmp_path, {
        "prog.py": """\
            def first_positive(xs):
                best = None
                for i in range(len(xs)):
                    if xs[i] > 10:
                        best = xs[i]
                return best
            assert first_positive([1, 5, 7]) == 7
        """}, ["prog.py"])
    assert not t.data.get("recorder_errors")
    assert 4 in slice_lines(t, "prog.py", 6, "best")
