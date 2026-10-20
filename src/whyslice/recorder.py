"""Dynamic half of WhySlice: record one execution as a dynamic dependence graph.

Every executed unit (see ``staticinfo``) becomes an *instance*.  An instance
keeps the values it read, the values it bound or mutated, and its incoming
dependence edges:

* data edges     -- for each name it reads, the instance that last bound that
                    name in the right scope, plus the last in-place mutation of
                    the object (or of ``obj.attr``) the name refers to;
* return edges   -- from the return statement of every callee whose result
                    flowed into this statement;
* exception edges -- from the statement where a propagating exception was
                    raised;
* control edge   -- the dynamic control-dependence parent: the most recent
                    executed instance among the unit's static parents, or the
                    call site for top-level statements of a function.

Parameters are bound by a synthetic ``entry`` instance per call whose edges
point at the definitions of the names used in the matching argument
expression, so slices stay argument-precise across calls.

Only the standard library is used: the recorder runs inside the target
program's own interpreter.
"""

from __future__ import annotations

import builtins
import collections
import os
import re
import reprlib
import sys
from typing import Any, Dict, List, Optional, Tuple

from .staticinfo import PREDICATE_KINDS, FileInfo

_SKIP_CODE_NAMES = frozenset({"<genexpr>", "<listcomp>", "<dictcomp>", "<setcomp>", "<lambda>"})
_IMMUTABLE = (int, float, complex, str, bytes, bool, type(None), tuple, frozenset, range, type)
# receivers for which a call such as ``x.append(...)`` is taken to mutate ``x``;
# methods of user classes are traced, so their own assignments say what changed
_CONTAINERS = (list, dict, set, bytearray, collections.deque)
_CO_VARARGS = 0x04
_CO_VARKEYWORDS = 0x08
_CO_GENERATOR_LIKE = 0x20 | 0x80 | 0x200  # generator, coroutine, async generator
_PKG_DIR = os.path.dirname(os.path.realpath(__file__))
# Helpers that build the message of a failing assertion: pytest's rewritten asserts and
# unittest's assert* methods call them after the comparison and before the raise. Project
# code they run (reprs of the compared objects) belongs to the report, not to the failure.
_EXPLAIN_HELPERS = {
    "_saferepr": "_pytest/assertion/rewrite.py",
    "_call_reprcompare": "_pytest/assertion/rewrite.py",
    "_format_explanation": "_pytest/assertion/rewrite.py",
    "_format_assertmsg": "_pytest/assertion/rewrite.py",
    "_format_boolop": "_pytest/assertion/rewrite.py",
    "_should_repr_global_name": "_pytest/assertion/rewrite.py",
    "safe_repr": "unittest/util.py",
    "_common_shorten_repr": "unittest/util.py",
}


class TraceLimit(BaseException):
    """Raised into the traced program when the instance budget is exhausted."""


class _ValueRepr(reprlib.Repr):
    def __init__(self) -> None:
        super().__init__()
        self.maxlevel = 3
        self.maxstring = 60
        self.maxother = 60
        self.maxlong = 40
        self.maxlist = self.maxtuple = 8
        self.maxdict = self.maxset = self.maxfrozenset = self.maxdeque = 6

    def repr_defaultdict(self, obj: Any, level: int) -> str:
        return self.repr_dict(obj, level)

    repr_OrderedDict = repr_Counter = repr_defaultdict

    def repr_instance(self, obj: Any, level: int) -> str:
        try:
            text = builtins.repr(obj)
        except Exception:
            return f"<{type(obj).__name__} object>"
        if len(text) > self.maxother:
            text = text[: self.maxother - 3] + "..."
        return text


_REPR = _ValueRepr()


_ADDRESS = re.compile(r" (?:object )?at 0x[0-9a-fA-F]+")


def show(value: Any) -> str:
    try:
        text = _REPR.repr(value)
    except Exception:
        return f"<{type(value).__name__} object>"
    return _ADDRESS.sub("", text) if " at 0x" in text else text


class Inst:
    """One executed unit."""

    __slots__ = ("iid", "fid", "file", "line", "kind", "uses", "deps", "cd", "defs", "muts", "exc", "out")

    def __init__(self, iid: int, fid: int, file: int, line: int, kind: str) -> None:
        self.iid = iid
        self.fid = fid
        self.file = file
        self.line = line
        self.kind = kind
        self.uses: Dict[str, str] = {}
        self.deps: List[Tuple[int, str]] = []
        self.cd: Optional[int] = None
        self.defs: Dict[str, str] = {}
        self.muts: Dict[str, str] = {}
        self.exc: Optional[str] = None
        self.out = False

    def as_tuple(self) -> tuple:
        return (self.iid, self.fid, self.file, self.line, self.kind, self.uses, self.deps,
                self.cd, self.defs, self.muts, self.exc, self.out)


class FrameRec:
    __slots__ = ("fid", "frame", "info", "file", "func", "cur", "last", "last_def", "pred_last",
                 "entry", "is_gen", "scope", "local_names", "cell_names", "exc_value",
                 "pending_exc", "own_key", "calls_seen", "pred_hist")

    def __init__(self, fid: int, frame: Any, info: FileInfo, file: int, func: str) -> None:
        code = frame.f_code
        self.fid = fid
        self.frame = frame          # keeps id(frame) unique while the record lives
        self.info = info
        self.file = file
        self.func = func
        self.cur: Optional[Inst] = None
        self.last: Optional[Inst] = None
        self.last_def: Dict[str, int] = {}
        self.pred_last: Dict[int, int] = {}
        self.entry: Optional[int] = None
        self.is_gen = bool(code.co_flags & _CO_GENERATOR_LIKE)
        if frame.f_locals is frame.f_globals:
            self.scope = "module"
        elif code.co_flags & 0x01:  # CO_OPTIMIZED: a function body
            self.scope = "function"
        else:
            self.scope = "class"
        self.local_names = frozenset(code.co_varnames)
        self.cell_names = frozenset(code.co_cellvars) | frozenset(code.co_freevars)
        self.exc_value: Any = None
        self.pending_exc = False
        self.own_key = info.line_to_unit.get(code.co_firstlineno)
        self.calls_seen: Dict[Tuple[int, str], int] = {}
        self.pred_hist: List[Tuple[int, int]] = []   # (iid, unit key) of executed predicates


class Recorder:
    def __init__(self, root: str, limit: int = 300_000) -> None:
        self.root = os.path.realpath(root)
        self.limit = limit
        self.files: List[str] = []
        self.file_index: Dict[str, int] = {}
        self.infos: Dict[str, Optional[FileInfo]] = {}
        self.code_ok: Dict[Any, Optional[Tuple[FileInfo, int]]] = {}
        self.insts: List[Inst] = []
        self.frames: Dict[int, FrameRec] = {}
        self.frame_meta: List[tuple] = []
        self.gdefs: Dict[Tuple[int, str], int] = {}
        self.cells: Dict[str, int] = {}
        self.last_mut: Dict[int, int] = {}
        self.last_mut_attr: Dict[Tuple[int, str], int] = {}
        self.last_mut_item: Dict[Tuple[int, Any], int] = {}   # (id(container), key) -> writer
        self.last_mut_whole: Dict[int, int] = {}              # last change not tied to one key
        self._index_code: Dict[str, Any] = {}
        self.exc_origin: Dict[int, int] = {}
        self._alive: Dict[int, Any] = {}
        self.exc_raised_at: Dict[int, int] = {}   # exception -> instances recorded when raised
        self.explain_at: List[int] = []           # instances recorded when a failure message began
        self.outputs: List[int] = []
        self.truncated = False
        self.stopped = False
        self.errors: List[str] = []

    # ---- scope -------------------------------------------------------------------------

    def _want(self, code: Any) -> Optional[Tuple[FileInfo, int]]:
        hit = self.code_ok.get(code, False)
        if hit is not False:
            return hit
        result = None
        filename = code.co_filename
        if code.co_name not in _SKIP_CODE_NAMES and filename and not filename.startswith("<"):
            path = os.path.realpath(filename)
            if (path.startswith(self.root + os.sep) and "site-packages" not in path
                    and not path.startswith(_PKG_DIR + os.sep) and path.endswith(".py")):
                info = self.infos.get(path, False)
                if info is False:
                    try:
                        with open(path, encoding="utf-8") as fh:
                            info = FileInfo(path, fh.read())
                    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
                        info = None
                    self.infos[path] = info
                if info is not None:
                    idx = self.file_index.get(path)
                    if idx is None:
                        idx = len(self.files)
                        self.files.append(path)
                        self.file_index[path] = idx
                    result = (info, idx)
        self.code_ok[code] = result
        return result

    def _traced_ancestor(self, frame: Any) -> Optional[FrameRec]:
        while frame is not None:
            rec = self.frames.get(id(frame))
            if rec is not None and rec.frame is frame:
                return rec
            frame = frame.f_back
        return None

    # ---- bookkeeping -------------------------------------------------------------------

    def _new_inst(self, rec: FrameRec, line: int, kind: str) -> Inst:
        if len(self.insts) >= self.limit:
            self.truncated = True
            self.stop()
            raise TraceLimit(f"WhySlice stopped after {self.limit} statement instances")
        inst = Inst(len(self.insts), rec.fid, rec.file, line, kind)
        self.insts.append(inst)
        return inst

    def _lookup(self, rec: FrameRec, frame: Any, name: str) -> Tuple[Optional[tuple], bool, Any]:
        """Return (scope key, found, value) for a name as the current frame sees it."""
        if rec.scope == "function":
            if name in rec.local_names:
                loc = frame.f_locals
                if name in loc:
                    return ("l", name), True, loc[name]
                return None, False, None
            if name in rec.cell_names:
                loc = frame.f_locals
                if name in loc:
                    return ("c", name), True, loc[name]
                return None, False, None
        elif rec.scope == "class":
            loc = frame.f_locals
            if name in loc:
                return ("l", name), True, loc[name]
        glb = frame.f_globals
        if name in glb:
            return ("g", name), True, glb[name]
        bi = glb.get("__builtins__", builtins)
        bi = bi if isinstance(bi, dict) else vars(bi)
        if name in bi:
            return None, True, bi[name]
        return None, False, None

    def _def_of(self, rec: FrameRec, frame: Any, key: Optional[tuple]) -> Optional[int]:
        if key is None:
            return None
        kind, name = key
        if kind == "l":
            return rec.last_def.get(name)
        if kind == "c":
            return self.cells.get(name)
        return self.gdefs.get((id(frame.f_globals), name))

    def _set_def(self, rec: FrameRec, frame: Any, name: str, iid: int) -> None:
        if rec.scope == "module":
            self.gdefs[(id(frame.f_globals), name)] = iid
        elif rec.scope == "class":
            rec.last_def[name] = iid
        elif name in rec.local_names:
            rec.last_def[name] = iid
        elif name in rec.cell_names:
            self.cells[name] = iid
        else:  # declared ``global``
            self.gdefs[(id(frame.f_globals), name)] = iid

    @staticmethod
    def _attr(obj: Any, attr: str) -> Tuple[bool, Any]:
        """Read ``obj.attr`` without running properties or ``__getattr__``."""
        try:
            d = object.__getattribute__(obj, "__dict__")
        except Exception:
            return False, None
        if isinstance(d, dict) and attr in d:
            return True, d[attr]
        return False, None

    def _key(self, frame: Any, obj: Any, src: str) -> Tuple[bool, Any]:
        """Evaluate a simple index expression in the frame (names, constants, + - * // %)."""
        if not isinstance(obj, _CONTAINERS):
            return False, None
        code = self._index_code.get(src)
        if code is None:
            code = self._index_code[src] = compile(src, "<index>", "eval")
        try:
            key = eval(code, frame.f_globals, frame.f_locals)  # noqa: S307 - vetted by simple_index
            hash(key)
        except Exception:
            return False, None
        if isinstance(obj, (list, bytearray)) and isinstance(key, int) and key < 0:
            key += len(obj)
        return True, key

    def _remember_mut(self, obj: Any, iid: int, attr: str = "") -> None:
        if isinstance(obj, _IMMUTABLE):
            return
        oid = id(obj)
        self._alive[oid] = obj
        if attr:
            self.last_mut_attr[(oid, attr)] = iid
        self.last_mut[oid] = iid

    # ---- trace callbacks ---------------------------------------------------------------

    def start(self) -> None:
        sys.settrace(self._global)

    def stop(self) -> None:
        self.stopped = True
        sys.settrace(None)

    def _internal_error(self) -> None:
        """A recorder bug must never change the traced program: report once, stop recording."""
        import traceback

        self.errors.append(traceback.format_exc())
        sys.stderr.write("WhySlice recorder error (recording stopped):\n" + self.errors[-1])
        self.truncated = True
        self.stop()

    def _global(self, frame: Any, event: str, arg: Any) -> Any:
        if self.stopped or event != "call":
            return None
        try:
            helper = _EXPLAIN_HELPERS.get(frame.f_code.co_name)
            if helper and frame.f_code.co_filename.replace(os.sep, "/").endswith(helper):
                if not self.explain_at or self.explain_at[-1] != len(self.insts):
                    self.explain_at.append(len(self.insts))
            hit = self._want(frame.f_code)
            if hit is None:
                return None
            rec = self.frames.get(id(frame))
            if rec is not None and rec.frame is frame:  # generator or coroutine resumed
                return self._local
            self._enter(frame, *hit)
        except TraceLimit:
            raise
        except RecursionError:
            return None
        except Exception:
            self._internal_error()
            return None
        return self._local

    def _local(self, frame: Any, event: str, arg: Any) -> Any:
        if self.stopped:
            return None
        rec = self.frames.get(id(frame))
        if rec is None or rec.frame is not frame:
            return None
        try:
            if event == "line":
                self._on_line(rec, frame)
            elif event == "return":
                self._on_return(rec, frame, arg)
                return None
            elif event == "exception":
                self._on_exception(rec, arg)
        except TraceLimit:
            raise
        except RecursionError:
            return self._local
        except Exception:
            self._internal_error()
            return None
        return self._local

    # ---- events --------------------------------------------------------------------------

    def _enter(self, frame: Any, info: FileInfo, file: int) -> None:
        code = frame.f_code
        func = getattr(code, "co_qualname", code.co_name)
        fid = len(self.frame_meta)
        rec = FrameRec(fid, frame, info, file, func)
        self.frames[id(frame)] = rec
        caller = self._traced_ancestor(frame.f_back)
        caller_inst = caller.cur if caller is not None else None
        self.frame_meta.append((fid, func, file, None, caller_inst.iid if caller_inst else None))
        if rec.scope != "function":
            return
        line = rec.own_key if rec.own_key is not None else code.co_firstlineno
        entry = self._new_inst(rec, line, "entry")
        entry.cd = caller_inst.iid if caller_inst is not None else None
        nparams = code.co_argcount + code.co_kwonlyargcount
        nparams += bool(code.co_flags & _CO_VARARGS) + bool(code.co_flags & _CO_VARKEYWORDS)
        params = list(code.co_varnames[:nparams])
        loc = frame.f_locals
        for p in params:
            if p in loc:
                entry.defs[p] = show(loc[p])
            rec.last_def[p] = entry.iid
        if caller_inst is not None and caller is not None:
            entry.deps = self._param_deps(caller, caller_inst, code, params)
        rec.entry = entry.iid
        self.frame_meta[fid] = (fid, func, file, entry.iid, caller_inst.iid if caller_inst else None)
        rec.last = entry

    def _param_deps(self, caller: FrameRec, cinst: Inst, code: Any, params: List[str]) -> List[Tuple[int, str]]:
        """Map each parameter to the definitions its argument expression read."""
        by_label: Dict[str, List[int]] = {}
        returns: List[int] = []
        for iid, label in cinst.deps:
            if label == "·ret":
                returns.append(iid)
            elif not label.startswith(("?", "·")):
                by_label.setdefault(label, []).append(iid)
        unit = caller.info.units.get(cinst.line)
        sites = []
        if unit is not None:
            sites = [c for c in unit.calls if c.name == code.co_name]
            if not sites and code.co_name == "__init__":
                sites = [c for c in unit.calls if c.name[:1].isupper()]
        deps: List[Tuple[int, str]] = []
        if len(sites) >= 1 and not sites[0].starred:
            n_seen = caller.calls_seen.get((cinst.iid, code.co_name), 0)
            caller.calls_seen[(cinst.iid, code.co_name)] = n_seen + 1
            site = sites[min(n_seen, len(sites) - 1)]
            positional = list(params[: code.co_argcount])
            sources: Dict[str, frozenset] = {}
            if site.receiver is not None and positional and positional[0] in ("self", "cls"):
                sources[positional[0]] = site.receiver
                positional = positional[1:]
            elif code.co_name == "__init__" and positional:
                positional = positional[1:]
            for p, names in zip(positional, site.args):
                sources[p] = names
            for kw, names in site.kwargs.items():
                if kw in params:
                    sources[kw] = names
            for p, labels in sources.items():
                for lab in labels:
                    hits = by_label.get(lab)
                    if hits is None and "." in lab:  # attribute never written in traced code
                        hits = by_label.get(lab.split(".")[0], ())
                    for iid in hits or ():
                        deps.append((iid, f"{p}<{lab}"))
            for iid in returns:  # nested calls inside the arguments, e.g. f(g(x))
                deps.append((iid, "·ret"))
            return deps
        # unknown call shape: every parameter may depend on everything the call site read
        for lab, label_iids in by_label.items():
            for iid in label_iids:
                for p in params:
                    deps.append((iid, f"{p}<{lab}"))
        for iid in returns:
            deps.append((iid, "·ret"))
        return deps

    def _dynamic_cd(self, rec: FrameRec, unit: Any) -> Optional[int]:
        enc = unit.cd_enclosing
        best = rec.pred_last.get(enc) if enc is not None else None
        floor = best if best is not None else (rec.entry if rec.entry is not None else -1)
        for g in unit.cd_guards:
            gi = rec.pred_last.get(g)
            if gi is not None and gi > floor and (best is None or gi > best):
                best = gi
        return best if best is not None else rec.entry

    def _on_line(self, rec: FrameRec, frame: Any) -> None:
        key = rec.info.line_to_unit.get(frame.f_lineno)
        if key is None or key == rec.own_key:
            return
        cur = rec.cur
        if cur is not None and cur.line == key:
            return  # continuation line of a multi-line statement
        self._complete(rec, frame, key)
        unit = rec.info.units[key]
        inst = self._new_inst(rec, key, unit.kind)
        inst.cd = self._dynamic_cd(rec, unit)
        seen = set()
        for name in unit.uses:
            skey, found, val = self._lookup(rec, frame, name)
            if not found or skey is None:
                continue
            inst.uses[name] = show(val)
            d = self._def_of(rec, frame, skey)
            if d is not None:
                inst.deps.append((d, name))
                seen.add(d)
            if skey[0] == "l":
                self._potential(rec, inst, name, d)
            if name in unit.bare and not isinstance(val, _IMMUTABLE):
                m = self.last_mut.get(id(val))
                if m is not None and m not in seen:
                    inst.deps.append((m, name))
                    seen.add(m)
        for root, src in unit.uses_item:
            skey, found, obj = self._lookup(rec, frame, root)
            if not found or skey is None:
                continue
            ok, ikey = self._key(frame, obj, src)
            label = f"{root}[{show(ikey)}]" if ok else f"{root}[{src}]"
            if not ok:  # unknown element: depend on any change of the container
                m = self.last_mut.get(id(obj))
                if m is not None and m not in seen:
                    inst.deps.append((m, root))
                    seen.add(m)
                continue
            try:
                inst.uses[label] = show(obj[ikey]) if (ikey in obj if isinstance(obj, dict) else 0 <= ikey < len(obj)) else "<missing>"
            except Exception:
                pass
            w = self.last_mut_item.get((id(obj), ikey))
            whole = self.last_mut_whole.get(id(obj))
            for m in (w, whole if whole is not None and (w is None or whole > w) else None):
                if m is not None and m not in seen:
                    inst.deps.append((m, label))
                    seen.add(m)
        for root, attr in unit.uses_attr:
            skey, found, obj = self._lookup(rec, frame, root)
            if not found or skey is None or isinstance(obj, _IMMUTABLE):
                continue
            label = f"{root}.{attr}"
            ok, val = self._attr(obj, attr)
            if ok:
                inst.uses[label] = show(val)
            else:  # a method or class attribute (``d.values()``): reads the object's state
                m = self.last_mut.get(id(obj))
                if m is not None and m not in seen:
                    inst.deps.append((m, root))
                    seen.add(m)
            m = self.last_mut_attr.get((id(obj), attr))
            if m is not None and m not in seen:
                inst.deps.append((m, label))
                seen.add(m)
            if ok and not isinstance(val, _IMMUTABLE):
                m2 = self.last_mut.get(id(val))
                if m2 is not None and m2 not in seen:
                    inst.deps.append((m2, label))
                    seen.add(m2)
        if unit.kind == "handler" and rec.exc_value is not None:
            origin = self.exc_origin.get(id(rec.exc_value))
            if origin is not None:
                inst.deps.append((origin, "·exc"))
        rec.pending_exc = False
        if unit.is_output:
            inst.out = True
            self.outputs.append(inst.iid)
        if unit.kind in PREDICATE_KINDS:
            rec.pred_last[key] = inst.iid
            rec.pred_hist.append((inst.iid, key))
        rec.cur = inst

    def _potential(self, rec: FrameRec, inst: Inst, name: str, d: Optional[int]) -> None:
        """Relevant-slicing edges: predicates evaluated after ``name`` was last bound
        whose other outcome would have re-bound it (execution-omission errors)."""
        floor = d if d is not None else (rec.entry if rec.entry is not None else -1)
        units = rec.info.units
        found: List[Tuple[int, int]] = []
        for iid, key in reversed(rec.pred_hist[-40:]):
            if iid <= floor:
                break
            outcome = self.insts[iid].defs.get("<branch>")
            unit = units[key]
            if outcome in ("not taken", "exit loop"):
                other = unit.body_defs
            elif outcome in ("taken", "enter body"):
                other = unit.else_defs
            else:
                continue
            if name in other:
                found.append((iid, key))
        found.sort(key=lambda e: (units[e[1]].kind in ("for", "while"), -e[0]))
        for iid, _key in found[:4]:
            inst.deps.append((iid, "?" + name))

    def _complete(self, rec: FrameRec, frame: Any, next_key: Optional[int] = None) -> None:
        inst = rec.cur
        if inst is None:
            return
        rec.cur = None
        rec.last = inst
        unit = rec.info.units.get(inst.line)
        if unit is None:
            return
        if unit.body_span is not None and unit.body_span[0] != unit.line:
            inside = next_key is not None and unit.body_span[0] <= next_key <= unit.body_span[1]
            if unit.kind == "if":
                inst.defs["<branch>"] = "taken" if inside else "not taken"
            else:
                inst.defs["<branch>"] = "enter body" if inside else "exit loop"
        for name in unit.defs:
            _, found, val = self._lookup(rec, frame, name)
            if not found:
                continue
            inst.defs[name] = show(val)
            self._set_def(rec, frame, name, inst.iid)
        for root, attr, via_call in [(r, a, False) for r, a in unit.muts] + \
                [(r, a, True) for r, a in unit.muts_call]:
            _, found, obj = self._lookup(rec, frame, root)
            if not found or isinstance(obj, _IMMUTABLE):
                continue
            if via_call:
                target = self._attr(obj, attr)[1] if attr else obj
                if not isinstance(target, _CONTAINERS):
                    continue
            if attr:
                ok, val = self._attr(obj, attr)
                inst.muts[f"{root}.{attr}"] = show(val) if ok else "?"
                self._remember_mut(obj, inst.iid, attr)
                if ok:
                    self._remember_mut(val, inst.iid)
                    if via_call:
                        self.last_mut_whole[id(val)] = inst.iid
            else:
                inst.muts[root] = show(obj)
                self._remember_mut(obj, inst.iid)
                self.last_mut_whole[id(obj)] = inst.iid
        for root, src in unit.muts_item:
            _, found, obj = self._lookup(rec, frame, root)
            if not found or isinstance(obj, _IMMUTABLE):
                continue
            ok, key = self._key(frame, obj, src)
            self._remember_mut(obj, inst.iid)
            if ok:
                self.last_mut_item[(id(obj), key)] = inst.iid
                try:
                    inst.muts[f"{root}[{show(key)}]"] = show(obj[key])
                except Exception:
                    inst.muts[f"{root}[{show(key)}]"] = "?"
            else:
                self.last_mut_whole[id(obj)] = inst.iid
                inst.muts[root] = show(obj)

    def _on_return(self, rec: FrameRec, frame: Any, value: Any) -> None:
        self._complete(rec, frame)
        last = rec.last
        if last is not None and not rec.pending_exc:
            last.defs["<return>"] = show(value)
            caller = self._traced_ancestor(frame.f_back)
            if caller is not None and caller.cur is not None:
                caller.cur.deps.append((last.iid, "·ret"))
        if not rec.is_gen:
            del self.frames[id(frame)]

    def _on_exception(self, rec: FrameRec, arg: Any) -> None:
        exc_type, exc_value, _tb = arg
        cur = rec.cur if rec.cur is not None else rec.last
        rec.exc_value = exc_value
        rec.pending_exc = True
        if cur is None:
            return
        key = id(exc_value)
        origin = self.exc_origin.get(key)
        if origin is None:
            self.exc_origin[key] = cur.iid
            self.exc_raised_at[key] = len(self.insts)
            self._alive[key] = exc_value
            msg = str(exc_value).strip().splitlines()[0] if str(exc_value).strip() else ""
            cur.exc = f"{exc_type.__name__}: {msg[:120]}" if msg else exc_type.__name__
        elif origin != cur.iid and all(d != (origin, "·exc") for d in cur.deps):
            cur.deps.append((origin, "·exc"))

    # ---- output --------------------------------------------------------------------------

    def origin_of(self, exc: BaseException) -> Optional[int]:
        return self.exc_origin.get(id(exc))

    def raised_at_of(self, exc: BaseException) -> Optional[int]:
        """Number of instances recorded when ``exc`` was raised: all of them ran before it."""
        return self.exc_raised_at.get(id(exc))

    def snapshot(self) -> dict:
        units = {}
        sources = {}
        for idx, path in enumerate(self.files):
            info = self.infos[path]
            assert info is not None
            sources[idx] = info.lines
            units[idx] = {k: (u.end, u.kind, u.func, info.text(u)) for k, u in info.units.items()}
        return {
            "version": 1,
            "root": self.root,
            "files": list(self.files),
            "sources": sources,
            "units": units,
            "insts": [i.as_tuple() for i in self.insts],
            "frames": list(self.frame_meta),
            "outputs": list(self.outputs),
            "truncated": self.truncated,
            "explain_at": list(self.explain_at),
            "recorder_errors": self.errors[:3],
        }
