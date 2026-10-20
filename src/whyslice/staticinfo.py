"""Static half of WhySlice: per-line def/use sets and control-dependence parents.

The recorder's unit of execution is a *unit*: every statement that starts on a
source line, together with the continuation lines of a multi-line statement.
(Line-level granularity is what ``sys.settrace`` reports, so a finer unit
could not be observed anyway.)  For each unit this module computes

* ``uses``   names whose values the unit reads,
* ``defs``   names the unit (re)binds,
* ``muts``   root names whose *objects* the unit may change in place
             (``a[i] = x``, ``obj.f = y``, ``lst.append(z)``),
* ``cd_enclosing`` / ``cd_guards``  its static control-dependence parents.

Control dependence is read off the structured AST instead of a post-dominator
tree: a statement depends on its nearest enclosing predicate (``if``,
``while``, ``for``, ``except``, ``case``) and on earlier sibling statements
that contain a jump able to skip it (guard clauses such as ``if x: return``).
At run time the recorder turns this static set into one dynamic parent: the
most recent executed instance among the candidates.

Only the standard library is used, so the module can be injected into any
target environment.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# Methods that mutate their receiver in place (builtins and common stdlib types).
MUTATING_METHODS = frozenset({
    "append", "extend", "insert", "remove", "pop", "clear", "sort", "reverse",
    "update", "add", "discard", "setdefault", "popitem", "appendleft",
    "extendleft", "popleft", "rotate", "difference_update",
    "intersection_update", "symmetric_difference_update", "__setitem__",
    "__delitem__", "fill", "put", "put_nowait",
})
# Functions that mutate their first argument in place.
MUTATING_FUNCTIONS = frozenset({
    "heappush", "heappop", "heapify", "heapreplace", "heappushpop", "shuffle",
    "insort", "insort_left", "insort_right",
})
OUTPUT_FUNCTIONS = frozenset({"print"})
OUTPUT_METHODS = frozenset({"write", "writelines"})

PREDICATE_KINDS = frozenset({"if", "while", "for", "handler", "case"})


@dataclass
class CallSite:
    """One call expression inside a unit, kept for parameter mapping."""

    name: str                                  # callee name as written (last attribute)
    receiver: Optional[FrozenSet[str]]         # value labels read by ``obj`` in ``obj.m(...)``
    args: Tuple[FrozenSet[str], ...]           # value labels read by each positional argument
    kwargs: Dict[str, FrozenSet[str]]          # value labels read by each keyword argument
    starred: bool                              # uses *args / **kwargs (mapping is unreliable)
    # A value label is a bare name (``x``) or an attribute path (``self.total``).


@dataclass
class Unit:
    line: int
    end: int
    kind: str                                  # stmt | if | while | for | handler | case | def | return | try ...
    func: str                                  # qualified name of the enclosing function ('' = module)
    uses: Set[str] = field(default_factory=set)
    uses_attr: Set[Tuple[str, str]] = field(default_factory=set)   # (root, attr) read as root.attr
    defs: Set[str] = field(default_factory=set)
    muts: Set[Tuple[str, str]] = field(default_factory=set)        # (root, attr or '') assigned in place
    muts_call: Set[Tuple[str, str]] = field(default_factory=set)   # changed by a mutating builtin method
    uses_item: Set[Tuple[str, str]] = field(default_factory=set)   # (root, index source) read as root[index]
    muts_item: Set[Tuple[str, str]] = field(default_factory=set)   # (root, index source) assigned
    bare: Set[str] = field(default_factory=set)                    # names read as whole values
    calls: List[CallSite] = field(default_factory=list)
    cd_enclosing: Optional[int] = None         # line key of the nearest enclosing predicate
    cd_guards: Set[int] = field(default_factory=set)
    is_output: bool = False
    is_return: bool = False
    body_span: Optional[Tuple[int, int]] = None   # lines of the branch/loop body (if/while/for)
    else_span: Optional[Tuple[int, int]] = None   # lines of the ``else`` block, if any
    body_defs: Set[str] = field(default_factory=set)    # names (re)bound or changed in the body
    else_defs: Set[str] = field(default_factory=set)    # ... in the else block


def _root_path(expr: ast.AST) -> Optional[Tuple[str, str]]:
    """``self.items[0]`` -> ``('self', 'items')``; ``a[i].x`` -> ``('a', '')``."""
    node = expr
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred)):
        inner = node.value
        if isinstance(inner, ast.Name):
            return inner.id, node.attr if isinstance(node, ast.Attribute) else ""
        node = inner
    if isinstance(node, ast.Name):
        return node.id, ""
    return None


_INDEX_OPS = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)


def simple_index(node: ast.AST) -> Optional[str]:
    """Source of a subscript index that is safe to evaluate (names, constants, + - * // %)."""
    def ok(n: ast.AST) -> bool:
        if isinstance(n, ast.Name):
            return True
        if isinstance(n, ast.Constant):
            return isinstance(n.value, (int, str, bool, type(None)))
        if isinstance(n, ast.Tuple):
            return all(ok(e) for e in n.elts)
        if isinstance(n, ast.BinOp):
            return isinstance(n.op, _INDEX_OPS) and ok(n.left) and ok(n.right)
        if isinstance(n, ast.UnaryOp):
            return isinstance(n.op, (ast.USub, ast.UAdd)) and ok(n.operand)
        return False
    return ast.unparse(node) if ok(node) else None


def _arg_names(args: ast.arguments) -> Set[str]:
    names = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


def _bound_by_target(target: ast.AST) -> Set[str]:
    out: Set[str] = set()
    for node in ast.walk(target):
        if isinstance(node, ast.Name):
            out.add(node.id)
    return out


class _Reads(ast.NodeVisitor):
    """Names an expression loads, minus names local to comprehensions/lambdas.

    Also records walrus targets, mutating calls, output calls and call sites,
    because they are found while walking the same expression.
    """

    def __init__(self) -> None:
        self.names: Set[str] = set()
        self.bare: Set[str] = set()            # names read other than as ``name.attr`` / ``name[i]``
        self.attrs: Set[Tuple[str, str]] = set()
        self.items: Set[Tuple[str, str]] = set()
        self.walrus: Set[str] = set()
        self.muts: Set[Tuple[str, str]] = set()
        self.muts_call: Set[Tuple[str, str]] = set()
        self.calls: List[CallSite] = []
        self.output = False
        self._bound: List[Set[str]] = []

    def _is_bound(self, name: str) -> bool:
        return any(name in b for b in self._bound)

    def labels(self) -> FrozenSet[str]:
        return frozenset(self.bare | {f"{r}.{a}" for r, a in self.attrs})

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and not self._is_bound(node.id):
            self.names.add(node.id)
            self.bare.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        base = node.value
        if (isinstance(base, ast.Name) and isinstance(base.ctx, ast.Load)
                and not self._is_bound(base.id)):
            self.attrs.add((base.id, node.attr))
            self.names.add(base.id)  # still a read of the binding, but not a bare value
            return
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        base = node.value
        idx = simple_index(node.slice)
        if (idx is not None and isinstance(base, ast.Name) and isinstance(base.ctx, ast.Load)
                and isinstance(node.ctx, ast.Load) and not self._is_bound(base.id)):
            self.items.add((base.id, idx))
            self.names.add(base.id)
            self.visit(node.slice)
            return
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self.walrus.add(node.target.id)

    def _comprehension(self, node: ast.AST, elts: List[ast.AST]) -> None:
        gens = node.generators  # type: ignore[attr-defined]
        self.visit(gens[0].iter)  # evaluated in the enclosing scope
        bound: Set[str] = set()
        for g in gens:
            bound |= _bound_by_target(g.target)
        self._bound.append(bound)
        for i, g in enumerate(gens):
            if i:
                self.visit(g.iter)
            for cond in g.ifs:
                self.visit(cond)
        for e in elts:
            self.visit(e)
        self._bound.pop()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._comprehension(node, [node.elt])

    visit_SetComp = visit_ListComp

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._comprehension(node, [node.elt])

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._comprehension(node, [node.key, node.value])

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for d in node.args.defaults:
            self.visit(d)
        for d in node.args.kw_defaults:
            if d is not None:
                self.visit(d)
        self._bound.append(_arg_names(node.args))
        self.visit(node.body)
        self._bound.pop()

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = ""
        receiver: Optional[FrozenSet[str]] = None
        if isinstance(func, ast.Name):
            name = func.id
            if name in OUTPUT_FUNCTIONS:
                self.output = True
            if name in MUTATING_FUNCTIONS and node.args:
                path = _root_path(node.args[0])
                if path:
                    self.muts_call.add(path)
        elif isinstance(func, ast.Attribute):
            name = func.attr
            sub = _Reads()
            sub._bound = self._bound
            sub.visit(func.value)
            receiver = sub.labels()
            if name in MUTATING_METHODS:
                path = _root_path(func.value)
                if path:
                    self.muts_call.add(path)
            elif name in MUTATING_FUNCTIONS and node.args:
                path = _root_path(node.args[0])
                if path:
                    self.muts_call.add(path)
            if name in OUTPUT_METHODS:
                self.output = True
        args = []
        starred = False
        for a in node.args:
            sub = _Reads()
            sub._bound = self._bound
            sub.visit(a)
            args.append(sub.labels())
            starred = starred or isinstance(a, ast.Starred)
        kwargs: Dict[str, FrozenSet[str]] = {}
        for kw in node.keywords:
            sub = _Reads()
            sub._bound = self._bound
            sub.visit(kw.value)
            if kw.arg is None:
                starred = True
            else:
                kwargs[kw.arg] = sub.labels()
        if name:
            self.calls.append(CallSite(name, receiver, tuple(args), kwargs, starred))
        self.generic_visit(node)


def _absorb(unit: Unit, *exprs: Optional[ast.AST]) -> None:
    """Add everything an expression reads to ``unit``."""
    for e in exprs:
        if e is None:
            continue
        r = _Reads()
        r.visit(e)
        unit.uses |= r.names
        unit.bare |= r.bare
        unit.uses_attr |= r.attrs
        unit.uses_item |= r.items
        unit.defs |= r.walrus
        unit.muts |= r.muts
        unit.muts_call |= r.muts_call
        unit.calls.extend(r.calls)
        unit.is_output = unit.is_output or r.output


def _assign_target(unit: Unit, target: ast.AST, augmented: bool = False) -> None:
    if isinstance(target, ast.Name):
        unit.defs.add(target.id)
        if augmented:
            unit.uses.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for e in target.elts:
            _assign_target(unit, e, augmented)
    elif isinstance(target, ast.Starred):
        _assign_target(unit, target.value, augmented)
    elif (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
          and simple_index(target.slice) is not None):
        root, idx = target.value.id, simple_index(target.slice)
        unit.muts_item.add((root, idx))
        unit.uses.add(root)
        if augmented:
            unit.uses_item.add((root, idx))
        _absorb(unit, target.slice)
    elif isinstance(target, (ast.Attribute, ast.Subscript)):
        path = _root_path(target)
        if path:
            unit.muts.add(path)
            if augmented:  # ``x.f += 1`` also reads x.f
                unit.uses.add(path[0])
                if path[1]:
                    unit.uses_attr.add(path)
        _absorb(unit, target.value)
        if isinstance(target, ast.Subscript):
            _absorb(unit, target.slice)


def _pattern_captures(pattern: ast.AST) -> Set[str]:
    out: Set[str] = set()
    for node in ast.walk(pattern):
        for attr in ("name", "rest"):
            val = getattr(node, attr, None)
            if isinstance(val, str):
                out.add(val)
    return out


# --- jump analysis for guard clauses -------------------------------------------------

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
_LOOPS = (ast.For, ast.AsyncFor, ast.While)


def _escaping_jumps(stmt: ast.stmt) -> List[ast.stmt]:
    """Jump statements inside ``stmt`` that can skip the statements after it.

    ``return``/``raise`` always leave the block (a ``raise`` inside a ``try``
    of the same statement is assumed to be caught there); ``break`` and
    ``continue`` count only when their loop lies outside ``stmt``.
    """
    found: List[ast.stmt] = []

    def walk(node: ast.AST, in_loop: bool, in_try: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _SCOPES):
                continue
            if isinstance(child, ast.Return):
                found.append(child)
            elif isinstance(child, ast.Raise) and not in_try:
                found.append(child)
            elif isinstance(child, (ast.Break, ast.Continue)) and not in_loop:
                found.append(child)
            walk(child,
                 in_loop or isinstance(child, _LOOPS),
                 in_try or isinstance(child, ast.Try))

    walk(stmt, isinstance(stmt, _LOOPS), isinstance(stmt, ast.Try))
    return found


class FileInfo:
    """All units of one source file, keyed by the unit's first line."""

    def __init__(self, path: str, source: str) -> None:
        self.path = path
        self.lines = source.splitlines()
        self.units: Dict[int, Unit] = {}
        self.line_to_unit: Dict[int, int] = {}
        self.func_first_line: Dict[str, int] = {}
        self._pred_of_stmt: Dict[int, int] = {}   # id(ast node) -> unit key of the jump's nearest predicate
        tree = ast.parse(source, filename=path)
        self._block(tree.body, enclosing=None, func="")
        self._region_defs()

    # ---- construction ----------------------------------------------------------

    def _unit_for(self, first: int, last: int, kind: str, func: str) -> Unit:
        unit = self.units.get(first)
        if unit is None:
            unit = Unit(line=first, end=last, kind=kind, func=func)
            self.units[first] = unit
        else:  # several statements share the line: merge into one unit
            unit.end = max(unit.end, last)
            if kind in PREDICATE_KINDS or unit.kind == "stmt":
                unit.kind = kind
        for ln in range(first, last + 1):
            self.line_to_unit.setdefault(ln, first)
        return unit

    def _header_span(self, node: ast.stmt) -> Tuple[int, int]:
        first = node.lineno
        decorators = getattr(node, "decorator_list", None) or []
        if decorators:
            first = min(first, min(d.lineno for d in decorators))
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            last = max(node.lineno, body[0].lineno - 1)
            if body[0].lineno == node.lineno:
                last = node.lineno
        else:
            last = node.end_lineno or node.lineno
        return first, last

    def _block(self, stmts: List[ast.stmt], enclosing: Optional[int], func: str) -> None:
        guards: Set[int] = set()
        for stmt in stmts:
            key = self._stmt(stmt, enclosing, guards, func)
            for jump in _escaping_jumps(stmt):
                pred = self._pred_of_stmt.get(id(jump))
                guards.add(pred if pred is not None else key)

    def _stmt(self, s: ast.stmt, enclosing: Optional[int], guards: Set[int], func: str) -> int:
        first, last = self._header_span(s)
        kind = self._kind(s)
        unit = self._unit_for(first, last, kind, func)
        # a body statement written on its predicate's line joins that unit: no self edge
        if unit.cd_enclosing is None and enclosing != unit.line:
            unit.cd_enclosing = enclosing
        unit.cd_guards |= {g for g in guards if g != unit.line}
        key = unit.line

        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
            unit.defs.add(s.name)
            _absorb(unit, *s.decorator_list, *s.args.defaults,
                    *[d for d in s.args.kw_defaults if d is not None], s.returns)
            inner = f"{func}.{s.name}" if func else s.name
            self.func_first_line.setdefault(inner, first)
            self._block(s.body, enclosing=None, func=inner)
        elif isinstance(s, ast.ClassDef):
            unit.defs.add(s.name)
            _absorb(unit, *s.decorator_list, *s.bases, *[k.value for k in s.keywords])
            inner = f"{func}.{s.name}" if func else s.name
            self._block(s.body, enclosing=None, func=inner)
        elif isinstance(s, ast.If):
            _absorb(unit, s.test)
            self._spans(unit, s)
            self._block(s.body, key, func)
            self._block(s.orelse, key, func)
        elif isinstance(s, ast.While):
            _absorb(unit, s.test)
            self._spans(unit, s)
            self._block(s.body, key, func)
            self._block(s.orelse, key, func)
        elif isinstance(s, (ast.For, ast.AsyncFor)):
            _absorb(unit, s.iter)
            _assign_target(unit, s.target)
            self._spans(unit, s)
            self._block(s.body, key, func)
            self._block(s.orelse, key, func)
        elif isinstance(s, (ast.With, ast.AsyncWith)):
            for item in s.items:
                _absorb(unit, item.context_expr)
                if item.optional_vars is not None:
                    _assign_target(unit, item.optional_vars)
            self._block(s.body, enclosing, func)
        elif isinstance(s, ast.Try) or s.__class__.__name__ == "TryStar":
            self._block(s.body, enclosing, func)  # type: ignore[attr-defined]
            for h in s.handlers:  # type: ignore[attr-defined]
                hfirst = h.lineno
                hlast = max(h.lineno, h.body[0].lineno - 1) if h.body[0].lineno != h.lineno else h.lineno
                hu = self._unit_for(hfirst, hlast, "handler", func)
                hu.cd_enclosing = enclosing
                _absorb(hu, h.type)
                if h.name:
                    hu.defs.add(h.name)
                self._block(h.body, hu.line, func)
            self._block(s.orelse, enclosing, func)  # type: ignore[attr-defined]
            self._block(s.finalbody, enclosing, func)  # type: ignore[attr-defined]
        elif s.__class__.__name__ == "Match":
            _absorb(unit, s.subject)  # type: ignore[attr-defined]
            for case in s.cases:  # type: ignore[attr-defined]
                cl = case.pattern.lineno
                cu = self._unit_for(cl, cl, "case", func)
                cu.cd_enclosing = key
                cu.defs |= _pattern_captures(case.pattern)
                _absorb(cu, case.guard)
                self._block(case.body, cu.line, func)
        elif isinstance(s, ast.Assign):
            _absorb(unit, s.value)
            for t in s.targets:
                _assign_target(unit, t)
        elif isinstance(s, ast.AugAssign):
            _absorb(unit, s.value)
            _assign_target(unit, s.target, augmented=True)
        elif isinstance(s, ast.AnnAssign):
            _absorb(unit, s.value)
            if s.value is not None:
                _assign_target(unit, s.target)
        elif isinstance(s, ast.Return):
            _absorb(unit, s.value)
            unit.is_return = True
        elif isinstance(s, ast.Delete):
            for t in s.targets:
                if isinstance(t, ast.Name):
                    unit.defs.add(t.id)
                else:
                    _assign_target(unit, t)
        elif isinstance(s, (ast.Import, ast.ImportFrom)):
            for alias in s.names:
                if alias.name != "*":
                    unit.defs.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(s, (ast.Global, ast.Nonlocal, ast.Pass, ast.Break, ast.Continue)):
            pass
        else:  # Expr, Assert, Raise and anything newer: read every sub-expression
            for child in ast.iter_child_nodes(s):
                if isinstance(child, ast.expr):
                    _absorb(unit, child)

        if isinstance(s, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            # the predicate that decides whether this jump happens
            self._pred_of_stmt[id(s)] = enclosing if enclosing is not None else key
        return key

    @staticmethod
    def _spans(unit: Unit, s: ast.stmt) -> None:
        def span(block: List[ast.stmt]) -> Tuple[int, int]:
            return block[0].lineno, block[-1].end_lineno or block[-1].lineno

        unit.body_span = span(s.body)  # type: ignore[attr-defined]
        orelse = getattr(s, "orelse", None)
        if orelse:
            unit.else_span = span(orelse)

    def _region_defs(self) -> None:
        """Names a predicate's *other* outcome could have (re)assigned.

        Used for relevant slicing: if a branch was not taken (or a loop
        stopped) and its body assigns ``v``, a later read of ``v`` potentially
        depends on that decision even though no assignment executed.
        """
        for unit in self.units.values():
            for span, target in ((unit.body_span, unit.body_defs), (unit.else_span, unit.else_defs)):
                if span is None:
                    continue
                lo, hi = span
                for key, other in self.units.items():
                    if lo <= key <= hi and other.func == unit.func and other is not unit:
                        target |= other.defs
                        target |= {root for root, _ in other.muts | other.muts_call | other.muts_item}

    @staticmethod
    def _kind(s: ast.stmt) -> str:
        if isinstance(s, ast.If):
            return "if"
        if isinstance(s, ast.While):
            return "while"
        if isinstance(s, (ast.For, ast.AsyncFor)):
            return "for"
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return "def"
        if isinstance(s, ast.Return):
            return "return"
        if s.__class__.__name__ in ("Try", "TryStar"):
            return "try"
        if s.__class__.__name__ == "Match":
            return "match"
        return "stmt"

    # ---- queries ----------------------------------------------------------------

    def unit_at(self, line: int) -> Optional[Unit]:
        key = self.line_to_unit.get(line)
        return self.units.get(key) if key is not None else None

    def text(self, unit: Unit, width: int = 100) -> str:
        parts = [self.lines[i - 1].strip() for i in range(unit.line, min(unit.end, unit.line + 2) + 1)
                 if 0 < i <= len(self.lines)]
        txt = " ".join(p for p in parts if p)
        if unit.end > unit.line + 2:
            txt += " …"
        return txt if len(txt) <= width else txt[: width - 1] + "…"
