"""Query side of WhySlice: load a recorded run, slice it, explain values.

A slicing criterion is an executed instance of a line, optionally narrowed to
one variable.  ``backward`` computes the dynamic backward slice (transitive
closure over data, return, exception and control edges); ``explain`` renders
the same graph as a bounded causal tree with the values observed at run time,
which is what a coding agent reads.
"""

from __future__ import annotations

import math
import os
import pickle
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

# instance tuple layout (see recorder.Inst.as_tuple)
IID, FID, FILE, LINE, KIND, USES, DEPS, CD, DEFS, MUTS, EXC, OUT = range(12)


class QueryError(ValueError):
    pass


# reprs that carry no information beyond identity: <pkg.Cls>, <pkg.Cls object>, functions, classes
_OPAQUE = re.compile(r"<(?:[\w.]+(?: object)?|(?:function|bound method|class|module|built-in|lambda)\b.*?)(?:>|\.\.\.)")


def shown_label(label: str) -> str:
    """``'amount<subtotal'`` (param ``amount`` bound from caller's ``subtotal``) -> ``'amount'``."""
    return label.split("<", 1)[0] if "<" in label and not label.startswith("<") else label


def source_label(label: str) -> str:
    return label.split("<", 1)[1] if "<" in label and not label.startswith("<") else label


class Trace:
    def __init__(self, data: dict) -> None:
        self.data = data
        self.files: List[str] = data["files"]
        self.insts: List[tuple] = data["insts"]
        self.frames = {f[0]: f for f in data["frames"]}
        self.units: Dict[int, Dict[int, tuple]] = data["units"]
        self.sources: Dict[int, List[str]] = data["sources"]
        self.root: str = data["root"]
        self.by_loc: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for t in self.insts:
            if t[KIND] != "entry":
                self.by_loc[(t[FILE], t[LINE])].append(t[IID])
        self.line_to_unit: Dict[int, Dict[int, int]] = {}
        for f, units in self.units.items():
            m: Dict[int, int] = {}
            for key, (end, *_rest) in units.items():
                for ln in range(key, end + 1):
                    m.setdefault(ln, key)
            self.line_to_unit[f] = m

    @classmethod
    def load(cls, path: str) -> "Trace":
        with open(path, "rb") as fh:
            return cls(pickle.load(fh))

    # ---- locating --------------------------------------------------------------------

    def short(self, f: int) -> str:
        path = self.files[f]
        try:
            rel = os.path.relpath(path, self.root)
        except ValueError:
            return path
        return rel if not rel.startswith("..") else path

    def resolve_file(self, name: str) -> int:
        name = name.strip()
        norm = os.path.normpath(name)
        for i, path in enumerate(self.files):
            if os.path.normpath(path) == norm or os.path.normpath(os.path.join(self.root, name)) == path:
                return i
        hits = [i for i, p in enumerate(self.files)
                if p.endswith(os.sep + norm) or os.path.basename(p) == os.path.basename(norm)]
        exact_tail = [i for i in hits if self.files[i].endswith(os.sep + norm)]
        hits = exact_tail or hits
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise QueryError(f"{name} was not executed in this run. Traced files: "
                             + ", ".join(self.short(i) for i in range(len(self.files))))
        raise QueryError(f"{name} is ambiguous: " + ", ".join(self.short(i) for i in hits))

    def unit_key(self, f: int, line: int) -> int:
        key = self.line_to_unit.get(f, {}).get(line)
        if key is None:
            raise QueryError(f"{self.short(f)}:{line} is not a statement line")
        return key

    def find(self, file: str, line: int, occurrence: int = -1) -> int:
        f = self.resolve_file(file)
        key = self.unit_key(f, line)
        iids = self.by_loc.get((f, key), [])
        if not iids:
            raise QueryError(f"{self.short(f)}:{key} never executed in this run")
        n = len(iids)
        if occurrence < 0:
            occurrence += n
        else:
            occurrence -= 1  # callers count executions from 1
        if not 0 <= occurrence < n:
            raise QueryError(f"{self.short(f)}:{key} executed {n} times; occurrence out of range")
        return iids[occurrence]

    def occurrence(self, iid: int) -> Tuple[int, int]:
        t = self.insts[iid]
        iids = self.by_loc.get((t[FILE], t[LINE]), [iid])
        lo, hi = 0, len(iids)
        while lo < hi:  # iids are sorted
            mid = (lo + hi) // 2
            if iids[mid] < iid:
                lo = mid + 1
            else:
                hi = mid
        return lo + 1, len(iids)

    # ---- slicing -----------------------------------------------------------------------

    def edges(self, iid: int) -> List[Tuple[int, str]]:
        t = self.insts[iid]
        out = list(t[DEPS])
        if t[CD] is not None:
            out.append((t[CD], "·ctrl"))
        return out

    def seeds(self, iid: int, var: Optional[str]) -> List[int]:
        """Where a backward slice starts for ``var`` at instance ``iid``."""
        if not var:
            return [iid]
        t = self.insts[iid]
        if var in t[DEFS] or var in t[MUTS] or var == "<return>":
            return [iid]
        root = var.split(".")[0].split("[")[0]
        picked = [d for d, label in t[DEPS] if shown_label(label) == var or label == "?" + var]
        if not picked:
            picked = [d for d, label in t[DEPS] if shown_label(label).split(".")[0] == root]
        if not picked and (root in t[USES] or var in t[USES]):
            return [iid]  # read but never assigned in traced code (a constant or builtin)
        if not picked:
            raise QueryError(f"`{var}` is neither read nor assigned by this statement; "
                             f"it reads {sorted(t[USES]) or 'nothing'}")
        return picked

    def edges_via(self, iid: int, via: Optional[str]) -> List[Tuple[int, str]]:
        """Edges to follow out of ``iid`` when it was reached through label ``via``.

        Reaching a call's entry instance through one parameter only makes that
        parameter's argument (and the call site) relevant, not the others.
        """
        t = self.insts[iid]
        if via and via.startswith("<"):
            # a call site reached from its callee's entry: only what happened before the call
            limit = int(via[1:])
            return [(d, lab) for d, lab in self.edges(iid) if d < limit]
        if t[KIND] == "entry" and via and via in t[DEFS]:
            return [(d, lab) for d, lab in self.edges(iid)
                    if shown_label(lab) == via or lab in ("·ret", "·ctrl")]
        return self.edges(iid)

    def via(self, parent: int, child: int, label: str) -> Optional[str]:
        """Traversal context for ``child`` when reached from ``parent`` over ``label``."""
        if label == "·ctrl" and self.insts[parent][KIND] == "entry":
            return f"<{parent}"
        if self.insts[child][KIND] == "entry":
            return source_label(label) if "<" in label else label
        return None

    def backward(self, seeds: Iterable[int], potential: bool = True) -> Set[int]:
        seen: Set[Tuple[int, Optional[str]]] = set()
        out: Set[int] = set()
        stack: List[Tuple[int, Optional[str]]] = [(s, None) for s in seeds]
        while stack:
            i, via = stack.pop()
            if (i, via) in seen:
                continue
            seen.add((i, via))
            out.add(i)
            for d, lab in self.edges_via(i, via):
                if lab.startswith("?") and not potential:
                    continue
                stack.append((d, self.via(i, d, lab)))
        return out

    def lines_of(self, iids: Iterable[int]) -> Set[Tuple[int, int]]:
        return {(self.insts[i][FILE], self.insts[i][LINE]) for i in iids if self.insts[i][KIND] != "entry"}

    def executed_lines(self) -> Set[Tuple[int, int]]:
        return set(self.by_loc)

    # ---- failure / output criteria -------------------------------------------------------

    def failure(self) -> Optional[dict]:
        for f in self.data.get("failures", []):
            if f.get("origin") is not None:
                return f
        fails = self.data.get("failures", [])
        return fails[0] if fails else None

    def failure_seeds(self) -> List[int]:
        f = self.failure()
        if f and f.get("origin") is not None:
            return [f["origin"]]
        if f and self.insts:
            # the raise point was not observed (e.g. RecursionError disables tracing):
            # explain the last statement the recorder saw
            last = max((t[IID] for t in self.insts if t[KIND] != "entry"), default=None)
            return [last] if last is not None else []
        return list(self.data.get("outputs", []))

    # ---- rendering ---------------------------------------------------------------------

    def code(self, iid: int) -> str:
        t = self.insts[iid]
        unit = self.units.get(t[FILE], {}).get(t[LINE])
        return unit[3] if unit else ""

    def func(self, iid: int) -> str:
        meta = self.frames.get(self.insts[iid][FID])
        return meta[1] if meta else "?"

    def where(self, iid: int) -> str:
        t = self.insts[iid]
        return f"{self.short(t[FILE])}:{t[LINE]}"

    def node(self, iid: int) -> str:
        t = self.insts[iid]
        if t[KIND] == "entry":
            params = ", ".join(f"{k}={v}" for k, v in t[DEFS].items() if k != "<return>")
            return f"#{iid} {self.where(iid)} call {self.func(iid)}({params})"
        k, n = self.occurrence(iid)
        rep = f" [run {k}/{n}]" if n > 1 else ""
        effects = [f"{k2}={v}" for k2, v in t[DEFS].items() if not k2.startswith("<")]
        if "<branch>" in t[DEFS]:
            effects.insert(0, t[DEFS]["<branch>"] + (" with " + ", ".join(f"{k2}={v}" for k2, v in t[USES].items()
                                                                         if not v.startswith("<"))
                                                    if t[USES] else ""))
        effects += [f"{k2}→{v}" for k2, v in t[MUTS].items()]
        if "<return>" in t[DEFS]:
            effects.append(f"returns {t[DEFS]['<return>']}")
        if t[EXC]:
            effects.append(f"raises {t[EXC]}")
        tail = ("   ⟹ " + "; ".join(effects)) if effects else ""
        return f"#{iid} {self.where(iid)} in {self.func(iid)}{rep}: {self.code(iid)}{tail}"

    def edge_label(self, parent: int, child: int, label: str) -> str:
        p = self.insts[parent]
        c = self.insts[child]
        if label == "·ctrl":
            if p[KIND] == "entry":
                return "called from"
            branch = c[DEFS].get("<branch>")
            return f"runs because ({branch})" if branch else "runs because of"
        if label == "·ret":
            return f"value returned {c[DEFS].get('<return>', '?')} by"
        if label == "·exc":
            return "exception raised by"
        if label.startswith("?"):
            branch = c[DEFS].get("<branch>", "")
            return f"`{label[1:]}` not re-assigned because ({branch})"
        if p[KIND] == "entry":
            name = shown_label(label)
            return f"param {name}={p[DEFS].get(name, '?')} from"
        value = p[USES].get(label)
        return f"{label}={value} from" if value is not None else f"{label} from"

    # -- edge triage for the causal tree --------------------------------------------------

    def _penalty(self, parent: int, child: int, label: str) -> Optional[float]:
        """Expansion cost of an edge; None hides it from the tree (it stays in the slice)."""
        c = self.insts[child]
        p = self.insts[parent]
        if c[KIND] == "def" or (c[KIND] == "stmt" and self.code(child).startswith(("import ", "from "))):
            return None                                    # definitions and imports are plumbing
        if label == "·ctrl":
            if c[KIND] == "entry":
                return None                                # "the function was called"
            return 2.0 if p[KIND] == "entry" else 0.75     # call site / branch decision
        if label in ("·ret", "·exc"):
            return 0.0
        if label.startswith("?"):
            return 0.25                                    # a branch that skipped an assignment: rare, telling
        label = shown_label(label) if p[KIND] == "entry" else label
        name = label.split(".")[0]
        value = p[DEFS].get(label) if p[KIND] == "entry" else p[USES].get(label)
        if name in ("self", "cls") and "." not in label:
            return None
        if value is not None and _OPAQUE.fullmatch(value):
            return None                                    # opaque object identity
        return 1.0 if "." in label else 0.0                 # object-state history comes after values

    def _tree(self, root: int, first_edges: List[Tuple[int, str]], max_nodes: int) -> Dict[int, list]:
        """Best-first expansion: cheap, value-carrying edges are explored first."""
        import heapq

        kids: Dict[int, list] = defaultdict(list)
        placed: Set[int] = {root}
        heap: List[tuple] = []
        counter = 0

        def push(parent: int, edges: List[Tuple[int, str]], depth: int) -> None:
            nonlocal counter
            for child, label in edges:
                pen = self._penalty(parent, child, label)
                if pen is None:
                    continue
                counter += 1
                heapq.heappush(heap, (depth + pen, counter, parent, child, label, depth))

        def follow(parent: int, child: int, label: str) -> List[Tuple[int, str]]:
            return self.edges_via(child, self.via(parent, child, label))

        push(root, first_edges, 1)
        budget = max_nodes
        while heap and budget > 0:
            _prio, _n, parent, child, label, depth = heapq.heappop(heap)
            if child in placed:
                if not any(c == child for c, _l, _k in kids[parent]):
                    kids[parent].append((child, label, "ref"))
                continue
            note = None
            # loop-carried chain: the same line feeding itself through the same variable
            cur, run = child, 0
            while run < 10_000:
                nxt = [d for d, lab in self.insts[cur][DEPS]
                       if lab == label and self.insts[d][LINE] == self.insts[cur][LINE]
                       and self.insts[d][FILE] == self.insts[cur][FILE]]
                if not nxt:
                    break
                cur, run = nxt[0], run + 1
            if run >= 2:
                note = f"⟲ {run} earlier runs of this line fed it (loop); earliest shown below"
                kids[parent].append((child, label, note))
                placed.add(child)
                kids[child].append((cur, label, "earliest"))
                placed.add(cur)
                push(cur, follow(child, cur, label), depth + 1)
            else:
                kids[parent].append((child, label, None))
                placed.add(child)
                push(child, follow(parent, child, label), depth + 1)
            budget -= 1
        return kids

    def causal_order(self, iid: int, limit: int = 100_000) -> List[int]:
        """Instances in the order the causal tree for ``iid`` would reveal them."""
        kids = self._tree(iid, self.edges(iid), limit)
        order: List[int] = []
        seen: Set[int] = {iid}

        def walk(p: int) -> None:
            for child, _label, note in sorted(kids.get(p, []), key=lambda e: e[0], reverse=True):
                if note == "ref" or child in seen:
                    continue
                seen.add(child)
                order.append(child)
        # breadth-first over the tree reflects the best-first priorities closely enough
        frontier = [iid]
        while frontier:
            nxt: List[int] = []
            for p in frontier:
                before = len(order)
                walk(p)
                nxt.extend(order[before:])
            frontier = nxt
        return order

    def explain(self, iid: int, var: Optional[str] = None, max_nodes: int = 30,
                summary: bool = True) -> str:
        t = self.insts[iid]
        out: List[str] = []
        if var and var not in t[DEFS] and var not in t[MUTS]:
            value = t[USES].get(var)
            head = f"WHY does `{var}` = {value} at {self.where(iid)}" if value is not None \
                else f"WHY `{var}` at {self.where(iid)}"
        elif var:
            head = f"WHY is `{var}` set to {t[DEFS].get(var, t[MUTS].get(var, '?'))} at {self.where(iid)}"
        elif t[EXC]:
            head = f"WHY does {self.where(iid)} raise {t[EXC]}"
        else:
            head = f"WHY does {self.where(iid)} execute with these values"
        out.append(head + "  (each indented line is a cause of the line above it; values are from this run)")
        out.append(self.node(iid))
        roots = self.seeds(iid, var)
        if roots == [iid]:
            first = self.edges(iid)
        else:
            first = [(d, lab) for d, lab in t[DEPS]
                     if d in roots and (shown_label(lab).split(".")[0] == var.split(".")[0].split("[")[0]
                                        or lab == "?" + var)] or [(d, var or "") for d in roots]
        kids = self._tree(iid, first, max_nodes)
        self._print(iid, kids, 1, out, set())
        hidden = [d for d, lab in first if self._penalty(iid, d, lab) is None and lab not in ("·ctrl",)]
        if hidden:
            out.append("  (also depends on object identity/definitions: "
                       + ", ".join(f"{lab}←#{d}" for d, lab in first if d in hidden)[:200] + ")")
        if summary:
            out.append("")
            out.append(self.summary(self.backward(roots) | {iid}))
            out.append("Ask `why` with a file, line and variable from any #id above to go deeper.")
        return "\n".join(out)

    def _print(self, parent: int, kids: Dict[int, list], depth: int, out: List[str], done: Set[int]) -> None:
        pad = "  " * depth
        order = {"·exc": 0, "·ret": 1}
        children = sorted(kids.get(parent, []),
                          key=lambda e: (order.get(e[1], 2 if not e[1].startswith(("?", "·")) else 3), e[0]))
        for child, label, note in children:
            edge = self.edge_label(parent, child, label)
            if note == "ref" or child in done:
                out.append(f"{pad}{edge} #{child} (shown above)")
                continue
            if note == "earliest":
                out.append(f"{pad}{self.node(child)}")
            else:
                out.append(f"{pad}{edge} {self.node(child)}")
                if note:
                    out.append(f"{pad}  {note}")
            done.add(child)
            self._print(child, kids, depth + 1, out, done)

    def summary(self, iids: Set[int], max_lines: int = 40) -> str:
        lines = self.lines_of(iids)
        executed = self.executed_lines()
        by_func: Dict[Tuple[int, str], Set[int]] = defaultdict(set)
        for i in iids:
            t = self.insts[i]
            if t[KIND] != "entry":
                by_func[(t[FILE], self.func(i))].add(t[LINE])
        pct = 100.0 * len(lines) / max(1, len(executed))
        out = [f"Slice: {len(lines)} of {len(executed)} executed lines ({pct:.0f}%) "
               f"can influence this value:"]
        shown = 0
        for (f, fn), lns in sorted(by_func.items(), key=lambda kv: (self.short(kv[0][0]), min(kv[1]))):
            if shown >= max_lines:
                out.append("  …")
                break
            out.append(f"  {self.short(f)} {fn}: " + ", ".join(str(n) for n in sorted(lns)))
            shown += 1
        return "\n".join(out)

    def failure_report(self, max_nodes: int = 30) -> str:
        f = self.failure()
        head: List[str] = []
        if f:
            first_line = f["exc"].strip().splitlines()[0] if f["exc"].strip() else f["exc"]
            head.append(f"FAILED {f['test']} ({f['phase']}): {first_line}")
        elif self.data.get("outputs"):
            head.append("No exception; explaining what the program printed.")
        if self.data.get("truncated"):
            head.append("NOTE: recording stopped at the instance limit (possible infinite loop). "
                        "Most executed lines: " + self.hot_lines())
        seeds = self.failure_seeds()
        if f and f.get("origin") is None and seeds:
            head.append("NOTE: the exact raise point was not recorded (tracing stops on RecursionError); "
                        "explaining the last statement executed before the failure: "
                        + self.where(seeds[0]) + " — hot lines: " + self.hot_lines())
        if not seeds:
            head.append("Nothing to explain: no failure and no output statement executed.")
            return "\n".join(head)
        if len(seeds) == 1:
            head.append(self.explain(seeds[0], max_nodes=max_nodes))
        else:
            for s in seeds[-3:]:
                head.append(self.explain(s, max_nodes=max_nodes // 3, summary=False))
            head.append(self.summary(self.backward(seeds)))
        return "\n".join(head)

    def suspects(self, spectrum: dict, k: int = 10) -> str:
        """Lines of the failing run ranked by Ochiai over the tests of the same module.

        A line executed by the failing run but by few passing tests is specific to
        the failure; lines inside the causal slice win ties.
        """
        outcomes = spectrum.get("outcomes", {})
        cover = {t: {(p, ln) for p, ln in lines} for t, lines in spectrum.get("lines", {}).items()}
        failing = [t for t, o in outcomes.items() if o == "failed" and t in cover]
        passing = [t for t, o in outcomes.items() if o == "passed" and t in cover]
        if not failing or not passing:
            return ""
        path_idx = {os.path.realpath(p): i for i, p in enumerate(self.files)}
        seeds = self.failure_seeds()
        in_slice = self.lines_of(self.backward(seeds) | set(seeds)) if seeds else set()
        # only what ran up to the failure: pytest's own failure reporting (reprs of the
        # compared objects) runs afterwards and appears in the failing test alone
        cutoff = max(seeds) if seeds else len(self.insts)
        executed = {(t[FILE], t[LINE]) for t in self.insts[: cutoff + 1] if t[KIND] != "entry"}

        def unit(p: str, ln: int):
            f = path_idx.get(p)
            if f is None:
                return None
            return f, self.line_to_unit.get(f, {}).get(ln, ln)

        ef: Dict[tuple, int] = defaultdict(int)
        ep: Dict[tuple, int] = defaultdict(int)
        for t in failing:
            for u in {unit(p, ln) for p, ln in cover[t]} - {None}:
                ef[u] += 1
        for t in passing:
            for u in {unit(p, ln) for p, ln in cover[t]} - {None}:
                ep[u] += 1
        scored = []
        for loc in executed:
            f, ln = loc
            if os.path.basename(self.files[f]).startswith("test") or os.sep + "tests" + os.sep in self.files[f]:
                continue
            e_f = ef.get(loc, 0)
            if not e_f:
                continue
            score = e_f / math.sqrt(len(failing) * (e_f + ep.get(loc, 0)))
            scored.append((score + (0.05 if loc in in_slice else 0.0), score, loc))
        scored.sort(key=lambda x: (-x[0], x[2]))
        self._last_ranking = [(score, loc) for _t, score, loc in scored]
        if not scored:
            return ""
        out = [f"SUSPICIOUS LINES — executed in this failing run but by few of the {len(passing)} passing tests "
               f"of the same test module (Ochiai; ★ = in the causal slice below):"]
        for rank, (_tot, score, (f, ln)) in enumerate(scored[:k], 1):
            iids = self.by_loc.get((f, ln), [])
            last = iids[-1] if iids else None
            vals = ""
            if last is not None:
                t = self.insts[last]
                shown = [f"{a}={b}" for a, b in list(t[DEFS].items()) + list(t[USES].items())
                         if not a.startswith("<") and not _OPAQUE.fullmatch(b)][:3]
                if "<branch>" in t[DEFS]:
                    shown.insert(0, t[DEFS]["<branch>"])
                vals = ("  [last run: " + "; ".join(shown) + "]") if shown else ""
            star = "★" if (f, ln) in in_slice else " "
            code = self.units.get(f, {}).get(ln, (0, "", "", ""))[3]
            out.append(f"{rank:2d}. {star} {self.short(f)}:{ln}  score {score:.2f}, passing tests running it: "
                       f"{ep.get((f, ln), 0)}  | {code}{vals}")
        return "\n".join(out)

    def hot_lines(self, k: int = 5) -> str:
        counts = sorted(((len(v), key) for key, v in self.by_loc.items()), reverse=True)[:k]
        return ", ".join(f"{self.short(f)}:{ln} ×{n}" for n, (f, ln) in counts)

    def overview(self) -> str:
        n = sum(1 for t in self.insts if t[KIND] != "entry")
        files = ", ".join(self.short(i) for i in range(len(self.files)))
        return (f"Recorded {n} statement executions over {len(self.by_loc)} distinct lines "
                f"in {len(self.files)} file(s): {files}. Exit code {self.data.get('exit_code')}, "
                f"{self.data.get('elapsed')} s.")
