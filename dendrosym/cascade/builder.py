"""cascade_builder.py -- Structure-aware cascade decomposition.

Instead of trying to reverse-engineer layer boundaries from flat CSE output,
this module captures the natural mathematical structure by wrapping the
equation-building process. Each geometric operation (inverse metric,
Christoffels, Ricci, etc.) becomes a "chunk" that forms a cascade layer.

CSE is then run *within* each chunk to optimize it, rather than globally.

User guide (spec contract, emitters, onboarding): findings/cascade_api_guide.md
Test suite: tests/run_all.py

Usage:
    from dendrosym.cascade.systems.bssn.legacy import dendro
    from dendrosym.cascade.systems.bssn.legacy import bssn
    from dendrosym.cascade.builder import CascadeBuilder

    builder = CascadeBuilder()
    # ... call dendro functions, capture intermediates as chunks ...
    builder.add_chunk("inverse_metric", {"igt00": expr, ...})
    builder.add_chunk("first_christoffel", {"C1_000": expr, ...})
    # ...
    result = builder.build()
    result.summary()
    cpp_code = result.emit_cpp()
"""

import warnings

import sympy as sym
from sympy import Symbol, Matrix
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional


def expand_integer_pows(expr):
    """Rewrite x**n (2<=n<10 integer) as x*x*...*x so sym.ccode emits chained
    multiplications instead of pow() libm calls. Mirrors dendro.replace_pow;
    pow() blocks scheduling and is slower than a few muls at scalar/SIMD."""
    p, q = sym.Wild("p"), sym.Wild("q")
    def pow_to_mul(p, q):
        # accept 2 and 2.0 alike: EMDA matter sources write float exponents
        n = int(float(q)) if (q.is_number and q.is_real
                              and float(q).is_integer()) else None
        if n is not None and 1 < n < 10:
            return sym.Mul(*([p] * n), evaluate=False)
        if n is not None and -10 < n < -1:
            # gcc only folds pow(x, n) for n = 2 without unsafe-math; a negative
            # exponent survives as a libm call that clobbers every xmm register.
            return sym.Pow(sym.Mul(*([p] * -n), evaluate=False), -1,
                           evaluate=False)
        return sym.Pow(p, q)
    return expr.replace(sym.Pow(p, q), pow_to_mul)


import re as regex


# Verbatim copy of the legacy dendro.py `change_deriv_names` (the only thing the
# generic builder needed from that 4k-line module); regex text pass over the
# printed C++ that renames grad(...)/pow(...) forms. Do not edit: emitted bytes.
def change_deriv_names(str):
    c_str = str
    derivs = ["agrad", "grad", "kograd"]
    for deriv in derivs:
        key = deriv + r"\(\d, \w+\[pp\]\)"
        slist = regex.findall(key, c_str)
        for s in slist:
            # print(s)
            w1 = s.split("(")
            w2 = w1[1].split(")")[0].split(",")
            # print(w1[0]+'_'+w2[0].strip()+'_'+w2[1].strip()+';')
            rep = w1[0]
            for v in w2:
                rep = rep + "_" + v.strip()
            # rep=rep+';'
            c_str = c_str.replace(s, rep)

    derivs2 = ["grad2"]
    for deriv in derivs2:
        key = deriv + r"\(\d, \d, \w+\[pp\]\)"
        slist = regex.findall(key, c_str)
        for s in slist:
            # print(s)
            w1 = s.split("(")
            w2 = w1[1].split(")")[0].split(",")
            # print(w1[0]+'_'+w2[0].strip()+'_'+w2[1].strip()+';')
            rep = w1[0]
            for v in w2:
                rep = rep + "_" + v.strip()
            # rep=rep+';'
            c_str = c_str.replace(s, rep)

    func_list = ["pow"]
    for func in func_list:
        key = func + r"\(\w+, \d\)"
        slist = regex.findall(key, c_str)
        for s in slist:
            # print(s)
            w1 = s.split("(")
            w2 = w1[1].split(")")[0].split(",")
            # print(w1[0]+'_'+w2[0].strip()+'_'+w2[1].strip()+';')
            if int(w2[1].strip()) == 2:
                rep = "(" + w2[0].strip() + " * " + w2[0].strip() + ")"
                c_str = c_str.replace(s, rep)
            # rep=w1[0]
            # for v in w2:
            #     rep=rep+'_'+v.strip()
            # #rep=rep+';'
            # c_str=c_str.replace(s,rep)
    return c_str


def _change_deriv_names():
    """Kept for callers of the old lazy loader."""
    return change_deriv_names


def to_cpp_expr(expr, dendro_var_style=True):
    """SymPy expr -> C++ string with DendroGR naming (pow->mul, deriv names)."""
    import re
    expr = expand_integer_pows(expr)
    s = sym.ccode(expr, standard='c99',
                  user_functions={"grad": "grad", "grad2": "grad2",
                                  "agrad": "agrad", "kograd": "kograd"})
    s = _change_deriv_names()(s)
    if dendro_var_style:
        s = re.sub(r'grad\((\d), (\w+)\[pp\]\)', r'grad_\1_\2[pp]', s)
        s = re.sub(r'grad2\((\d), (\d), (\w+)\[pp\]\)', r'grad2_\1_\2_\3[pp]', s)
        s = re.sub(r'agrad\((\d), (\w+)\[pp\]\)', r'agrad_\1_\2[pp]', s)
        s = re.sub(r'kograd\((\d), (\w+)\[pp\]\)', r'kograd_\1_\2[pp]', s)
    return s


# Symmetric 3x3 index helpers
_SYM_TBL = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
def _sym(i, j):
    return _SYM_TBL[min(i, j)][max(i, j)]

E_I = [0, 1, 2]
E_IJ = [(i, j) for i in range(3) for j in range(3)]
E_IJ_SYM = [(i, j) for i in range(3) for j in range(i, 3)]


def flatten_sym33(mat, prefix):
    """Flatten a 3x3 symmetric matrix into named scalar expressions."""
    d = OrderedDict()
    for i, j in E_IJ_SYM:
        d[f"{prefix}{i}{j}"] = mat[i, j]
    return d


def flatten_tensor3(tensor, prefix):
    """Flatten a 3x3x3 tensor into named scalar expressions."""
    d = OrderedDict()
    for i in range(3):
        for j in range(3):
            for k in range(3):
                d[f"{prefix}{i}{j}{k}"] = tensor[i, j, k]
    return d


def flatten_vec3(vec, prefix):
    """Flatten a 3-vector into named scalar expressions."""
    d = OrderedDict()
    for i in range(3):
        d[f"{prefix}{i}"] = vec[i]
    return d


def flatten_scalar(expr, name):
    """Wrap a scalar expression."""
    return OrderedDict([(name, expr)])


@dataclass
class ChunkResult:
    """Result of CSE within a single chunk."""
    name: str
    cse_temps: list        # [(sym, expr), ...] CSE temporaries
    outputs: OrderedDict   # name -> reduced expression
    input_symbols: set     # symbols from previous chunks + leaves
    n_temps: int = 0
    n_ops: int = 0
    n_prior_refs: int = 0  # distinct prior-chunk outputs this chunk reads
    emit_seq: list = None  # topo-ordered [(name, expr, is_output)] or None


@dataclass
class CascadeResult:
    """Complete cascade decomposition result."""
    chunks: List[ChunkResult]
    leaf_symbols: set

    def summary(self):
        total_temps = sum(c.n_temps for c in self.chunks)
        total_outputs = sum(len(c.outputs) for c in self.chunks)
        print(f"\nCascade Decomposition Summary")
        print(f"{'=' * 60}")
        print(f"Total chunks (layers): {len(self.chunks)}")
        print(f"Total CSE temporaries: {total_temps}")
        print(f"Total named outputs: {total_outputs}")
        print(f"Leaf symbols: {len(self.leaf_symbols)}")
        print()
        for i, c in enumerate(self.chunks):
            print(f"  L{i}: {c.name:30s}  {c.n_temps:4d} temps, {len(c.outputs):3d} outputs")
        print()

    def emit_cpp_tensor_loop(self, dialect, dendro_var_style=True,
                             inline_threshold=1):
        """Tensor-array emit (opt-in alternative to emit_cpp_unrolled).

        Non-RHS chunk outputs that the supplied `dialect` recognises as
        rank-1 (vec3) or symmetric rank-2 (sym33) tensors are emitted as
        local arrays:

            double igt[3][3];
            igt[0][0] = ...;  igt[1][0] = igt[0][1];

        Subsequent chunks that reference `igt00`, `igt01`, ... have those
        leaf symbols rewritten to `igt[0][0]`, `igt[0][1]`, ... at print
        time. Scalars, rank-3 tensors, RHS write-targets, and CSE temps
        emit exactly as in emit_cpp_unrolled (flat scalar form).

        The point of the array form is that gcc gets to see the tensor as
        a struct with constant indices, which can reduce .text size and
        improve register allocation versus N independently-named locals.

        Returns the C++ body string.
        """
        from collections import OrderedDict as _OD
        import re


        # --- 1. Build the global Symbol-name -> array-access map.
        # Iterate the dialect; for every rank>=1 tensor, register every
        # component name as a rewrite target. Use a placeholder symbol name
        # that round-trips through sym.ccode safely; we substitute back to
        # the real array-access form in a post-pass.
        # Placeholder format: __TL_<tensor>__<i>[_<j>[_<k>]] (all letters/digits).
        component_to_access = {}   # str -> str (array-access C++ form)
        component_to_placeholder = {}  # str -> sym.Symbol (placeholder name)
        placeholder_to_access = {}     # placeholder-symbol-str -> array-access

        def _placeholder_name(tname, idx):
            return "__TL_" + tname + "__" + "_".join(str(i) for i in idx)

        def _access_string(tname, idx):
            return tname + "".join(f"[{i}]" for i in idx)

        if dialect is not None:
            for spec in dialect.tensors.values():
                if spec.rank == 0:
                    continue
                # Skip tensors used purely as RHS write-targets (e.g. gt_rhs,
                # At_rhs, Gt_rhs, B_rhs); those keep flat-scalar emit so the
                # write to the output array stays compact.
                if "_rhs" in spec.name:
                    continue
                for idx, comp_name in spec.components.items():
                    if comp_name in component_to_access:
                        continue  # already registered by an alias
                    ph = _placeholder_name(spec.name, idx)
                    access = _access_string(spec.name, idx)
                    component_to_access[comp_name] = access
                    component_to_placeholder[comp_name] = sym.Symbol(ph)
                    placeholder_to_access[ph] = access

        # SymPy substitution map: Symbol("igt00") -> Symbol("__TL_igt__0_0").
        leaf_subs = {sym.Symbol(name): ph
                     for name, ph in component_to_placeholder.items()}

        # --- 2. Identify RHS output names (write-to-array targets).
        rhs_names = set()
        for c in self.chunks:
            for name in c.outputs:
                if "_rhs" in name:
                    rhs_names.add(name)

        # --- 3. Per-chunk: classify outputs into (tensor_arrays, flat_scalars).
        # tensor_arrays: dict[tensor_name -> list[(idx_tuple, sym_expr)]]
        # flat_scalars: list[(output_name, sym_expr)]
        def _classify(chunk):
            tensors_here = {}
            scalars_here = []
            for out_name, out_expr in chunk.outputs.items():
                if out_name in rhs_names:
                    scalars_here.append((out_name, out_expr))
                    continue
                if dialect is None:
                    scalars_here.append((out_name, out_expr))
                    continue
                # Find which TensorSpec owns this output name (search rank>=1
                # tensors only; skip _rhs target specs).
                matched = None
                for spec in dialect.tensors.values():
                    if spec.rank == 0 or "_rhs" in spec.name:
                        continue
                    for idx, comp_name in spec.components.items():
                        if comp_name == out_name:
                            matched = (spec, idx)
                            break
                    if matched is not None:
                        break
                if matched is None:
                    scalars_here.append((out_name, out_expr))
                else:
                    spec, idx = matched
                    tensors_here.setdefault(spec.name, []).append((idx, out_expr))
            return tensors_here, scalars_here

        # --- 4. Printer (same post-processing as emit_cpp_unrolled).
        def to_cpp(expr):
            expr = expr.xreplace(leaf_subs) if leaf_subs else expr
            expr = expand_integer_pows(expr)
            s = sym.ccode(expr, standard='c99',
                          user_functions={"grad": "grad", "grad2": "grad2",
                                          "agrad": "agrad", "kograd": "kograd"})
            s = change_deriv_names(s)
            if dendro_var_style:
                s = re.sub(r'grad\((\d), (\w+)\[pp\]\)', r'grad_\1_\2[pp]', s)
                s = re.sub(r'grad2\((\d), (\d), (\w+)\[pp\]\)', r'grad2_\1_\2_\3[pp]', s)
                s = re.sub(r'agrad\((\d), (\w+)\[pp\]\)', r'agrad_\1_\2[pp]', s)
                s = re.sub(r'kograd\((\d), (\w+)\[pp\]\)', r'kograd_\1_\2[pp]', s)
            # Post-pass: __TL_igt__0_0 -> igt[0][0]
            for ph_name, access in placeholder_to_access.items():
                # Use word-boundary so __TL_igt__0_0 doesn't false-match __TL_igt__0_01
                s = re.sub(r'\b' + re.escape(ph_name) + r'\b', access, s)
            return s

        # --- 5. Emit.
        lines = []
        lines.append("// Cascade RHS -- tensor-loop emit (opt-in)")
        lines.append(f"// {len(self.chunks)} layers, "
                     f"{sum(c.n_temps for c in self.chunks)} CSE temps, "
                     f"{sum(len(c.outputs) for c in self.chunks)} named intermediates")
        lines.append("")

        for chunk in self.chunks:
            tensors_here, scalars_here = _classify(chunk)
            n_tensor_outs = sum(len(v) for v in tensors_here.values())
            lines.append(
                f"// === {chunk.name} ({chunk.n_temps} temps, "
                f"{n_tensor_outs} tensor-outs, {len(scalars_here)} scalar-outs) ==="
            )

            # CSE temps (always flat scalar)
            for temp_sym, temp_expr in chunk.cse_temps:
                lines.append(f"const double {temp_sym} = {to_cpp(temp_expr)};")

            # Tensor outputs grouped by tensor name
            for tensor_name, components in tensors_here.items():
                if dialect is None:
                    continue  # impossible (tensors_here only populated when dialect set)
                spec = dialect.spec(tensor_name)
                if spec.rank == 1:
                    lines.append(f"double {tensor_name}[3];")
                    for idx, expr in components:
                        lines.append(f"{tensor_name}[{idx[0]}] = {to_cpp(expr)};")
                elif spec.rank == 2:
                    lines.append(f"double {tensor_name}[3][3];")
                    emitted = set()
                    for idx, expr in components:
                        if idx in emitted:
                            continue
                        emitted.add(idx)
                        lines.append(
                            f"{tensor_name}[{idx[0]}][{idx[1]}] = {to_cpp(expr)};"
                        )
                    # Mirror sym entries we didn't emit (e.g. (1,0) := (0,1))
                    if spec.symmetric:
                        for idx, comp_name in spec.components.items():
                            if idx in emitted:
                                continue
                            sym_idx = tuple(sorted(idx))
                            if sym_idx in emitted:
                                lines.append(
                                    f"{tensor_name}[{idx[0]}][{idx[1]}] = "
                                    f"{tensor_name}[{sym_idx[0]}][{sym_idx[1]}];"
                                )
                                emitted.add(idx)
                else:
                    # Higher-rank: emit flat (don't try to array-ify rank-3 yet)
                    for idx, expr in components:
                        comp_name = spec.components[idx]
                        lines.append(f"const double {comp_name} = {to_cpp(expr)};")

            # Scalar outputs (and RHS write targets)
            for out_name, out_expr in scalars_here:
                cpp = to_cpp(out_expr)
                if out_name in rhs_names:
                    lines.append(f"{out_name}[pp] = {cpp};")
                else:
                    lines.append(f"const double {out_name} = {cpp};")
            lines.append("")

        return "\n".join(lines)

    def emit_cpp_unrolled(self, dendro_var_style=True, inline_threshold=0,
                          short_names=True):
        """Emit fully-unrolled C++ code for the entire cascade.

        Parameters
        ----------
        dendro_var_style : bool
            Transform derivative function calls to DendroGR naming.
        inline_threshold : int
            CSE temps referenced <= this many times get inlined to reduce
            register pressure. Set to 0 to keep all temps. Default 2.
        short_names : bool
            Use short DENDRO_NNN names instead of CASC_CHUNK_NNN.
        """
        import re


        def to_cpp(expr):
            """Convert SymPy expression to C++ string with DendroGR naming."""
            expr = expand_integer_pows(expr)
            s = sym.ccode(expr, standard='c99',
                         user_functions={"grad": "grad", "grad2": "grad2",
                                         "agrad": "agrad", "kograd": "kograd"})
            s = change_deriv_names(s)
            if dendro_var_style:
                # Transform derivative calls:
                # grad(0, alpha[pp]) -> grad_0_alpha[pp]
                # grad2(0, 1, chi[pp]) -> grad2_0_1_chi[pp]
                # agrad(0, beta0[pp]) -> agrad_0_beta0[pp]
                # kograd(0, alpha[pp]) -> kograd_0_alpha[pp]
                s = re.sub(r'grad\((\d), (\w+)\[pp\]\)', r'grad_\1_\2[pp]', s)
                s = re.sub(r'grad2\((\d), (\d), (\w+)\[pp\]\)', r'grad2_\1_\2_\3[pp]', s)
                s = re.sub(r'agrad\((\d), (\w+)\[pp\]\)', r'agrad_\1_\2[pp]', s)
                s = re.sub(r'kograd\((\d), (\w+)\[pp\]\)', r'kograd_\1_\2[pp]', s)
            return s

        # Which output names are final RHS outputs (written to arrays)?
        # Scan ALL chunks, not just the last: with split (Phase 2), RHS outputs
        # can live in any of several rhs_assembly_pN sub-chunks.
        rhs_names = set()
        for c in self.chunks:
            for name in c.outputs:
                if "_rhs" in name:
                    rhs_names.add(name)

        lines = []
        lines.append("// Cascade RHS -- auto-decomposed, per-chunk CSE")
        lines.append(f"// {len(self.chunks)} layers, {sum(c.n_temps for c in self.chunks)} CSE temps")
        lines.append(f"// {sum(len(c.outputs) for c in self.chunks)} named intermediates")
        lines.append("")

        for chunk in self.chunks:
            lines.append(f"// === {chunk.name} ({chunk.n_temps} temps, {len(chunk.outputs)} outputs) ===")

            # Emit in dependency order (emit_seq covers merged by-symbol
            # chunks whose temps reference same-chunk outputs); fall back to
            # temps-then-outputs for results built before emit_seq existed.
            seq = chunk.emit_seq
            if seq is None:
                seq = ([(str(t), e, False) for t, e in chunk.cse_temps]
                       + [(nm, oe, True) for nm, oe in chunk.outputs.items()])
            for name_, expr_, is_out in seq:
                cpp_expr = to_cpp(expr_)
                if is_out and name_ in rhs_names:
                    # RHS output: write to DendroGR array
                    lines.append(f"{name_}[pp] = {cpp_expr};")
                else:
                    lines.append(f"const double {name_} = {cpp_expr};")

            lines.append("")

        # Post-processing: inline low-use temps and rename
        if inline_threshold > 0 or short_names:
            full_text = "\n".join(lines)

            # Find all CSE-temp definitions and count references. Match any
            # ALLCAPS-prefixed identifier (CASC_*, EMDA_*, MAT_*, etc.) so the
            # inline/rename passes work regardless of the cse_prefix used
            # in builder.build(). Previously hardcoded to "CASC_", which
            # silently disabled inlining for any non-default prefix.
            temp_pattern = re.compile(r'const double ([A-Z][A-Z0-9_]*_\w+) = (.+);')
            temp_defs = {}  # name -> rhs expression string
            for line in lines:
                m = temp_pattern.match(line.strip())
                if m:
                    temp_defs[m.group(1)] = m.group(2)

            # Count references (excluding definition line)
            ref_counts = {}
            for name in temp_defs:
                ref_counts[name] = full_text.count(name) - 1

            # Phase 1: Inline temps with ref_count <= threshold
            # Process in reverse order so inlining a temp that references
            # another temp doesn't break things
            inlined = set()
            if inline_threshold > 0:
                temp_names_rev = list(reversed(temp_defs.keys()))
                for name in temp_names_rev:
                    if ref_counts.get(name, 0) <= inline_threshold:
                        rhs = temp_defs[name]
                        # Wrap in parens for safety
                        replacement = f"({rhs})"
                        # Remove the definition line
                        def_line = f"const double {name} = {rhs};"
                        full_text = full_text.replace(def_line, f"// inlined: {name}")
                        # Replace all references
                        full_text = full_text.replace(name, replacement)
                        inlined.add(name)

                # Clean up empty "inlined" comment lines
                full_text = re.sub(r'\n\s*// inlined: \w+', '', full_text)

            # Phase 2: Rename remaining CASC_ temps to short DENDRO_NNN names.
            # IMPORTANT: process longest names first. Naive substring-replace
            # otherwise corrupts e.g. CASC_RICCI_45 when CASC_RICCI_4 is replaced
            # first (the latter is a prefix of the former, producing
            # DENDRO_00005 from the suffix `5`).
            if short_names:
                remaining = [n for n in temp_defs if n not in inlined]
                # Stable order: by descending length, then by original order.
                rename_order = sorted(
                    enumerate(remaining), key=lambda e: (-len(e[1]), e[0])
                )
                # idx in the resulting DENDRO_NNNN reflects original ordering,
                # so the renamed names follow source order, not length order.
                for idx, name in rename_order:
                    short = f"DENDRO_{idx:04d}"
                    full_text = full_text.replace(name, short)

            # Phase 3: Fix pow(x[pp], 2) -> (x[pp] * x[pp])
            # The change_deriv_names handles pow(word, digit) but not pow(word[pp], digit)
            def fix_pow(m):
                base = m.group(1)
                exp = int(m.group(2))
                if exp == 2:
                    return f"(({base})*({base}))"
                return m.group(0)  # keep other powers
            full_text = re.sub(r'pow\(([^,]+),\s*(\d+)\)', fix_pow, full_text)

            lines = full_text.split("\n")

        return "\n".join(lines)

    def global_cse_result(self):
        """One global symbol-aware CSE over all chunk outputs, returned as a
        topo-ordered list of (name, sympy_expr, kind), kind in
        {'temp','inter','rhs'}.

        Reconstructs each chunk's named outputs in terms of (leaves + prior
        output symbols), then runs a single sym.cse over all of them. Named
        tensors (igt, R, ...) stay atomic symbols -- unlike the L1 collapse,
        which inlines them by value and blows expressions up. This removes the
        cross-chunk subexpression recompute that per-chunk CSE leaves behind
        (e.g. (1/4)*chi_inv^2 emitted in both Ricci and Derived chunks).
        Shared by the scalar and SIMD emitters."""
        import sys as _sys
        all_names, all_exprs = [], []
        for chunk in self.chunks:
            resolved = {}
            for s, e in chunk.cse_temps:        # cse_temps are topo-ordered
                resolved[s] = e.xreplace(resolved)
            for nm, oe in chunk.outputs.items():
                all_names.append(nm)
                all_exprs.append(oe.xreplace(resolved))
        gt, gr = sym.cse(all_exprs, symbols=sym.numbered_symbols("CASC_G_"))
        temp_def = {str(t): te for t, te in gt}
        name_expr = {nm: gr[i] for i, nm in enumerate(all_names)}
        node_names = set(temp_def) | set(all_names)
        dep = {}
        for t, te in gt:
            dep[str(t)] = {str(s) for s in te.free_symbols if str(s) in node_names}
        for nm in all_names:
            dep[nm] = {str(s) for s in name_expr[nm].free_symbols if str(s) in node_names}
        _sys.setrecursionlimit(100000)
        emitted, seen = [], set()
        def visit(n):
            if n in seen:
                return
            seen.add(n)
            for d in dep.get(n, ()):
                visit(d)
            emitted.append(n)
        for t, _ in gt:
            visit(str(t))
        for nm in all_names:
            visit(nm)
        rhs_names = {nm for nm in all_names if "_rhs" in nm}
        temp_set = set(temp_def)
        nodes = []
        for n in emitted:
            if n in temp_set:
                nodes.append((n, temp_def[n], "temp"))
            elif n in rhs_names:
                nodes.append((n, name_expr[n], "rhs"))
            else:
                nodes.append((n, name_expr[n], "inter"))
        return nodes

    def emit_cpp_global_cse(self, dendro_var_style=True, short_names=True):
        """Scalar emit with ONE global symbol-aware CSE instead of per-chunk.

        Emits the global_cse_result() node list in topological order (temps and
        intermediates interleaved, since a temp may depend on a named
        intermediate). No inlining -- inlining multiply-used temps duplicates
        their expressions and inflates code at scalar register counts."""
        import re

        nodes = self.global_cse_result()
        n_temp = sum(1 for _, _, k in nodes if k == "temp")
        n_named = sum(1 for _, _, k in nodes if k != "temp")
        lines = ["// Cascade RHS -- global symbol-aware CSE",
                 f"// {n_temp} global temps, {n_named} named intermediates",
                 ""]
        for name, e, kind in nodes:
            if kind == "rhs":
                lines.append(f"{name}[pp] = {to_cpp_expr(e, dendro_var_style)};")
            else:
                lines.append(f"const double {name} = {to_cpp_expr(e, dendro_var_style)};")
        text = "\n".join(lines)

        # 5. Rename CASC_G_NNN -> DENDRO_NNNN (word-boundary, prefix-safe).
        if short_names:
            found = set(re.findall(r'CASC_G_\d+', text))
            mapping = {nm: f"DENDRO_{i:04d}"
                       for i, nm in enumerate(sorted(found, key=lambda s: int(s.split('_')[-1])))}
            for nm in sorted(found, key=lambda s: -len(s)):
                text = re.sub(r'\b' + nm + r'\b', mapping[nm], text)
        return text


class CascadeBuilder:
    """Build a cascade decomposition by defining chunks in dependency order.

    Each chunk represents a geometric operation (inverse metric, Christoffels,
    etc.) and its outputs become available as inputs to subsequent chunks.
    CSE is run within each chunk independently.
    """

    def __init__(self):
        self._chunks = []          # list of (name, outputs_dict)
        self._leaf_symbols = set()
        self._available = set()    # symbols available as inputs (leaves + previous chunk outputs)
        self._chunk_symbols = {}   # chunk_output_name -> Symbol used for substitution

    def set_leaves(self, symbols):
        """Set the leaf symbols (state vars, derivatives, parameters)."""
        if isinstance(symbols, set):
            self._leaf_symbols = symbols
        else:
            self._leaf_symbols = set(symbols)
        self._available = set(self._leaf_symbols)

    def auto_detect_leaves(self, exprs):
        """Auto-detect leaf symbols from expressions."""
        all_syms = set()
        for expr in exprs:
            if isinstance(expr, sym.Basic):
                all_syms.update(expr.free_symbols)
            elif isinstance(expr, Matrix):
                all_syms.update(expr.free_symbols)
        self._leaf_symbols = all_syms
        self._available = set(all_syms)

    def declare(self, name, outputs):
        """Declare one named object in evaluation order.

        The paper's vocabulary: the practitioner declares OBJECTS; the layer
        boundaries that group them are computed (cascade_autolayer). Kept as an
        alias of add_chunk so existing specs keep working.
        """
        return self.add_chunk(name, outputs)

    def add_chunk(self, name, outputs):
        """Add a chunk (layer) to the cascade.

        Parameters
        ----------
        name : str
            Human-readable name (e.g., "inverse_metric")
        outputs : dict
            Maps output name (str) -> SymPy expression, in terms of leaves
            and previous chunk outputs. Prior outputs may be referenced
            *by symbol* (a bare Symbol carrying the output's name --
            recommended for new specs; see mhd.py) or *by value* (the
            earlier output's full expression tree, which build() replaces
            via exact-tree xreplace; bssn_cascade does this). A by-value
            reference that isn't tree-identical is silently recomputed --
            slower, never wrong.
        """
        self._chunks.append((name, OrderedDict(outputs)))

    def build(self, cse_prefix="CASC_", verbose=False, deep_substitute=True):
        """Run the decomposition: apply per-chunk CSE and build the result.

        Key: before running CSE on each chunk, we use xreplace (fast dict-based
        substitution) to replace previous chunk outputs with named symbols.

        Returns
        -------
        CascadeResult
        """
        results = []
        available = set(self._leaf_symbols)
        # Map from full expression -> substitute symbol for previous chunk outputs
        # Using xreplace for speed (exact match, not structural pattern matching)
        subs_dict = {}

        for chunk_name, chunk_outputs in self._chunks:
            if verbose:
                print(f"Processing chunk: {chunk_name} ({len(chunk_outputs)} outputs)")

            exprs = list(chunk_outputs.values())
            names = list(chunk_outputs.keys())

            # Fast substitution of previous chunk outputs
            n_prior_refs = 0
            if subs_dict:
                exprs = [e.xreplace(subs_dict) for e in exprs]
                if deep_substitute:
                    # F14: Mul-flattening hides Mul/Pow-headed outputs from
                    # xreplace; recover them with factor-multiset matching.
                    pats = sorted(
                        ((k, v) for k, v in subs_dict.items()
                         if k.is_Mul or k.is_Pow),
                        key=lambda kv: -kv[0].count_ops())
                    # fresh memo per chunk: the pattern set grows as chunks
                    # are processed, so cached rewrites would go stale.
                    exprs = _deep_substitute(exprs, pats)
                # Counts both conventions: by-value refs were just xreplace'd
                # into symbols; by-symbol refs were symbols all along.
                prev_syms = set(subs_dict.values())
                used = set()
                for e in exprs:
                    used.update(e.free_symbols & prev_syms)
                n_prior_refs = len(used)
                if verbose:
                    print(f"  References {n_prior_refs} outputs from previous chunks")

            # Run CSE on this chunk's expressions
            prefix_str = f"{cse_prefix}{chunk_name.upper()}_"
            cse_syms = sym.numbered_symbols(prefix_str)
            cse_temps, cse_reduced = sym.cse(exprs, symbols=cse_syms)

            # Topological emission order: with by-symbol specs, a merged
            # chunk's temps/outputs may reference outputs defined in the
            # SAME chunk; the default temps-then-outputs order would emit
            # uses before defs. Stable Kahn sort over in-chunk names.
            _stmts = ([(str(t), e, False) for t, e in cse_temps]
                      + list(zip(names, cse_reduced, [True] * len(names))))
            _here = {nm for nm, _, _ in _stmts}
            _emitted, _seq, _pending = set(), [], list(_stmts)
            while _pending:
                _progressed = False
                _rest = []
                for nm, e, is_out in _pending:
                    deps = {str(fs) for fs in e.free_symbols} & _here
                    if deps <= _emitted:
                        _seq.append((nm, e, is_out))
                        _emitted.add(nm)
                        _progressed = True
                    else:
                        _rest.append((nm, e, is_out))
                if not _progressed:      # no cycles possible; belt-and-braces
                    _seq.extend(_rest)
                    break
                _pending = _rest

            n_temps = len(cse_temps)
            if verbose:
                print(f"  CSE found {n_temps} common subexpressions")

            # Build reduced output dict
            reduced_outputs = OrderedDict()
            for name, reduced_expr in zip(names, cse_reduced):
                reduced_outputs[name] = reduced_expr

            # Register this chunk's ORIGINAL outputs for xreplace into future chunks
            orig_exprs = list(chunk_outputs.values())
            for name, orig_expr in zip(names, orig_exprs):
                out_sym = Symbol(name)
                subs_dict[orig_expr] = out_sym
                available.add(out_sym)

            result = ChunkResult(
                name=chunk_name,
                cse_temps=cse_temps,
                outputs=reduced_outputs,
                input_symbols=set(available),
                n_temps=n_temps,
                n_prior_refs=n_prior_refs,
                emit_seq=_seq,
            )
            results.append(result)

        if len(results) >= 2 and all(c.n_prior_refs == 0 for c in results):
            warnings.warn(
                "cascade build: no chunk references any prior chunk output, "
                "so layer boundaries carry nothing. If specs were "
                "post-processed (expand/simplify), by-value references no "
                "longer match exactly and sharing was silently lost.")

        return CascadeResult(chunks=results, leaf_symbols=self._leaf_symbols)


def _deep_substitute(exprs, patterns, memo=None):
    """Substitute Mul/Pow-headed prior-output expressions that SymPy's
    automatic Mul-flattening hides from exact-tree xreplace (F14).

    `patterns` is a list of (pattern_expr, symbol), largest first. For a
    Mul pattern, any Mul node whose factor multiset contains the pattern's
    factors gets the pattern replaced by its symbol; for a Pow pattern
    (base**e), factors base**(k*e) become symbol**k. Single traversal per
    expression with memoization; semantically the subset of sympy.subs we
    need, ~100x faster on large trees.
    """
    from collections import Counter

    mul_pats = []
    pow_pats = []
    for pat, s in patterns:
        if pat.is_Mul:
            mul_pats.append((Counter(pat.args), len(pat.args), s))
        elif pat.is_Pow and pat.exp.is_Number:
            pow_pats.append((pat.base, pat.exp, s))
    if not mul_pats and not pow_pats:
        return exprs
    if memo is None:
        memo = {}

    def sub_pow(factor):
        # base**(k*e) -> sym**k for a Pow pattern base**e
        if factor.is_Pow and factor.exp.is_Number:
            for base, e, s in pow_pats:
                if factor.base == base:
                    k = factor.exp / e
                    if k.is_Integer and k >= 1:
                        return s**int(k)
        return factor

    def rec(e):
        if e.is_Atom:
            return e
        cached = memo.get(e)
        if cached is not None:
            return cached
        args = tuple(rec(a) for a in e.args)
        e2 = e if args == e.args else e.func(*args)
        if e2.is_Mul:
            factors = Counter(e2.args)
            changed = False
            for pat_count, n_args, s in mul_pats:
                # repeat in case the pattern divides the node multiple times
                while all(factors[a] >= c for a, c in pat_count.items()):
                    for a, c in pat_count.items():
                        factors[a] -= c
                    factors[s] += 1
                    changed = True
            if changed:
                e2 = sym.Mul(*[a**1 if c == 1 else a for a, c in
                               factors.items() for _ in range(c)])
        elif e2.is_Pow:
            e2 = sub_pow(e2)
        memo[e] = e2
        return e2

    return [rec(e) for e in exprs]


def build_cascade_ir(chunks, leaves, target_L=None, smart_split=None,
                     cse_prefix="CASC_", verbose=False, auto_layers=False,
                     auto_search_order=False):
    """Spec -> CascadeResult: the shared driver pipeline.

    Applies the optional collapse/split L-knob, then runs CascadeBuilder.
    Single home for the logic previously copy-pasted across the bssn/mhd/
    neohook/emda drivers. Usage guide: findings/cascade_api_guide.md.

    Parameters
    ----------
    chunks : list[(name, OrderedDict[str, sym.Expr])]
        Chunk specs in dependency order (see add_chunk for the reference
        conventions). Spec functions return (chunks, leaves) in this order.
    leaves : set[sym.Symbol]
    target_L : int or None
        None or the natural depth: build as declared. Smaller: greedy
        adjacent collapse (cascade_collapse). Larger: split (cascade_split).
    smart_split : bool or None
        For target_L > natural only. True: post-CSE smart split (each split
        adds 2 chunks, so delta must be even). False: pre-CSE dumb split.
        None: auto -- smart when delta is even, dumb otherwise.
    cse_prefix : str
        Prefix for per-chunk CSE temp names.
    """
    if auto_layers:
        # Layer boundaries chosen by exact DP over the declared order
        # (cascade_autolayer); collapse/split do not apply -- the DP
        # reaches any L up to one-chunk-per-object directly.
        # Auto kernels use the AUTO_ prefix: sympy's CSE output depends on
        # the temp-symbol names (they enter canonical ordering of later
        # candidates), so the prefix is part of byte-reproducibility -- and
        # AUTO_ is what the archived/timed kernels carry.
        if cse_prefix == "CASC_":
            cse_prefix = "AUTO_"
        from dendrosym.cascade.autolayer import auto_chunks
        chunks = auto_chunks(chunks, leaves, L=target_L, verbose=verbose,
                             search_order=auto_search_order)
        target_L = None

    def _build(chunks_):
        builder = CascadeBuilder()
        builder.set_leaves(leaves)
        for name, outputs in chunks_:
            builder.add_chunk(name, outputs)
        return builder.build(cse_prefix=cse_prefix, verbose=verbose)

    natural_L = len(chunks)
    if smart_split is None and target_L is not None and target_L > natural_L:
        smart_split = ((target_L - natural_L) % 2 == 0)
        if not smart_split:
            warnings.warn(
                f"build_cascade_ir: target_L={target_L} is an odd delta from "
                f"natural_L={natural_L}, so smart split is unreachable; "
                "falling back to dumb (pre-CSE) split, which duplicates "
                "shared work (+18% on BSSN at L=9). Pass smart_split=False "
                "to silence.")

    if smart_split and target_L is not None and target_L > natural_L:
        delta = target_L - natural_L
        if delta % 2 != 0:
            raise ValueError(
                f"smart_split: target_L={target_L} not reachable from "
                f"natural_L={natural_L} (each smart split adds 2 chunks)")
        from dendrosym.cascade.split import smart_split_result
        return smart_split_result(_build(chunks), num_splits=delta // 2,
                                  verbose=verbose)

    if target_L is not None and target_L != natural_L:
        if target_L < natural_L:
            from dendrosym.cascade.collapse import collapse_to_target
            chunks = collapse_to_target(chunks, target_L, verbose=verbose)
            if target_L == 1:
                # L=1 is global CSE: one chunk, one CSE pass over everything.
                # Rename so temps read CASC_GLOBAL_n, not the concatenation
                # of every merged chunk name.
                chunks = [("global", chunks[0][1])]
        else:
            from dendrosym.cascade.split import split_to_target
            chunks = split_to_target(chunks, target_L, verbose=verbose)
    return _build(chunks)


def build_bssn_cascade(dendro_module, bssn_module, verbose=False):
    """Build the BSSN cascade using the natural geometric structure.

    This captures the intermediate geometric quantities (inverse metric,
    Christoffels, Ricci, etc.) as cascade chunks, matching the hand-crafted
    7-layer structure.

    Parameters
    ----------
    dendro_module : the dendro.py module
    bssn_module : the bssn.py module (with its variables already imported)

    Returns
    -------
    CascadeResult
    """
    # Import the variables from bssn module
    a = bssn_module.a
    chi = bssn_module.chi
    K = bssn_module.K
    Gt = bssn_module.Gt
    b = bssn_module.b
    B = bssn_module.B
    gt = bssn_module.gt
    At = bssn_module.At
    d = bssn_module.d
    d2 = bssn_module.d2
    ad = bssn_module.ad
    igt = bssn_module.igt
    eta_damp = bssn_module.eta
    weight = bssn_module.weight

    l1 = bssn_module.l1
    l2 = bssn_module.l2
    l3 = bssn_module.l3
    l4 = bssn_module.l4
    lf0 = bssn_module.lf0
    lf1 = bssn_module.lf1

    Rational = sym.Rational

    # Step 1: Compute all geometric intermediates (same as bssn.py does)
    C1 = dendro_module.get_first_christoffel()
    C2 = dendro_module.get_second_christoffel()
    C3 = dendro_module.get_complete_christoffel(chi)
    [R, Rt, Rphi, CalGt] = dendro_module.compute_ricci(Gt, chi)

    # Step 2: Compute RHS expressions
    a_rhs = l1 * dendro_module.lie(b, a) - 2 * a * K

    b_rhs = [
        Rational(3, 4) * (lf0 + lf1 * a) * B[i]
        + l2 * dendro_module.vec_j_ad_j(b, b[i])
        for i in dendro_module.e_i
    ]

    gt_rhs = dendro_module.lie(b, gt, weight) - 2 * a * At

    chi_rhs = dendro_module.lie(b, chi, weight) + Rational(2, 3) * (chi * a * K)

    AikAkj = Matrix([
        sum([
            At[i, k] * sum([dendro_module.inv_metric[k, l] * At[l, j]
                           for l in dendro_module.e_i])
            for k in dendro_module.e_i
        ])
        for i, j in dendro_module.e_ij
    ])

    At_rhs = (dendro_module.lie(b, At, weight)
              + chi * dendro_module.trace_free(a * R - dendro_module.DiDj(a))
              + a * (K * At - 2 * AikAkj.reshape(3, 3)))

    K_rhs = (dendro_module.lie(b, K)
             - dendro_module.laplacian(a, chi)
             + a * (K * K / 3 + dendro_module.sqr(At)))

    At_UU = dendro_module.up_up(At)

    Gt_rhs = (
        Matrix([sum(b[j] * ad(j, Gt[i]) for j in dendro_module.e_i) for i in dendro_module.e_i])
        - Matrix([sum(CalGt[j] * d(j, b[i]) for j in dendro_module.e_i) for i in dendro_module.e_i])
        + Rational(2, 3) * Matrix([CalGt[i] * sum(d(j, b[j]) for j in dendro_module.e_i) for i in dendro_module.e_i])
        + Matrix([sum([igt[j, k] * d2(j, k, b[i]) + igt[i, j] * d2(j, k, b[k]) / 3 for j, k in dendro_module.e_ij]) for i in dendro_module.e_i])
        - Matrix([sum([2 * At_UU[i, j] * d(j, a) for j in dendro_module.e_i]) for i in dendro_module.e_i])
        + Matrix([sum([2 * a * dendro_module.C2[i, j, k] * At_UU[j, k] for j, k in dendro_module.e_ij]) for i in dendro_module.e_i])
        - Matrix([sum([a * (3 / chi * At_UU[i, j] * d(j, chi) + Rational(4, 3) * dendro_module.inv_metric[i, j] * d(j, K)) for j in dendro_module.e_i]) for i in dendro_module.e_i])
    )
    Gt_rhs_list = [item for sublist in Gt_rhs.tolist() for item in sublist]

    B_rhs = [
        Gt_rhs_list[i] - eta_damp * B[i]
        + l3 * dendro_module.vec_j_ad_j(b, B[i])
        - l4 * dendro_module.vec_j_ad_j(b, Gt[i])
        for i in dendro_module.e_i
    ]

    # Step 3: Build cascade with natural chunk boundaries
    builder = CascadeBuilder()

    # Auto-detect leaves from the final RHS expressions
    all_rhs = ([a_rhs] + b_rhs
               + [gt_rhs[i, j] for i, j in E_IJ_SYM]
               + [chi_rhs]
               + [At_rhs[i, j] for i, j in E_IJ_SYM]
               + [K_rhs]
               + Gt_rhs_list
               + B_rhs)

    # Leaves = everything that's a free symbol in the geometric intermediates
    # (state vars, derivatives, parameters)
    all_syms = set()
    for expr in all_rhs:
        all_syms.update(expr.free_symbols)
    builder.set_leaves(all_syms)

    # Chunk L1: Inverse metric + 1/chi
    igt_outputs = flatten_sym33(igt, "igt")
    igt_outputs["chi_inv"] = 1 / chi
    builder.add_chunk("inverse_metric", igt_outputs)

    # Chunk L2: First Christoffel C1[k,i,j]
    c1_outputs = OrderedDict()
    for k in range(3):
        for i in range(3):
            for j in range(i, 3):  # symmetric in i,j
                c1_outputs[f"C1_{k}{i}{j}"] = C1[k, i, j]
    builder.add_chunk("first_christoffel", c1_outputs)

    # Chunk L3: Second Christoffel C2[i,j,k]
    c2_outputs = flatten_tensor3(C2, "C2_")
    builder.add_chunk("second_christoffel", c2_outputs)

    # Chunk L4: Complete Christoffel C3[i,j,k]
    c3_outputs = flatten_tensor3(C3, "C3_")
    builder.add_chunk("complete_christoffel", c3_outputs)

    # Chunk L5: Ricci tensor R[i,j] + CalGt[i]
    ricci_outputs = flatten_sym33(R, "R")
    ricci_outputs.update(flatten_vec3(CalGt, "CalGt"))
    builder.add_chunk("ricci", ricci_outputs)

    # Chunk L6: Derived quantities
    derived = OrderedDict()
    derived.update(flatten_sym33(At_UU, "At_UU"))
    derived.update(flatten_sym33(AikAkj.reshape(3, 3), "AikAkj"))
    DiDj_a = dendro_module.DiDj(a)
    derived.update(flatten_sym33(DiDj_a, "DiDj_a"))
    # trace_free(a*R - DiDj_a) -- compute inline
    tf = dendro_module.trace_free(a * R - DiDj_a)
    derived.update(flatten_sym33(tf, "tf"))
    # sqr(At) and laplacian(a, chi)
    derived["At_sqr"] = dendro_module.sqr(At)
    derived["lap_a"] = dendro_module.laplacian(a, chi)
    builder.add_chunk("derived_quantities", derived)

    # Chunk L7: RHS assembly
    rhs_outputs = OrderedDict()
    rhs_outputs["a_rhs"] = a_rhs
    for i in range(3):
        rhs_outputs[f"b_rhs{i}"] = b_rhs[i]
    for i, j in E_IJ_SYM:
        rhs_outputs[f"gt_rhs{i}{j}"] = gt_rhs[i, j]
    rhs_outputs["chi_rhs"] = chi_rhs
    for i, j in E_IJ_SYM:
        rhs_outputs[f"At_rhs{i}{j}"] = At_rhs[i, j]
    rhs_outputs["K_rhs"] = K_rhs
    for i in range(3):
        rhs_outputs[f"Gt_rhs{i}"] = Gt_rhs_list[i]
    for i in range(3):
        rhs_outputs[f"B_rhs{i}"] = B_rhs[i]
    builder.add_chunk("rhs_assembly", rhs_outputs)

    # Step 4: Build (runs per-chunk CSE)
    result = builder.build(verbose=verbose)
    return result


if __name__ == "__main__":
    import sys
    import os

    # Add cascade/ to path for dendro/bssn imports
    codegen_dir = os.path.dirname(os.path.abspath(__file__))
    cascade_dir = os.path.join(os.path.dirname(codegen_dir), "cascade")
    sys.path.insert(0, cascade_dir)

    from dendrosym.cascade.systems.bssn.legacy import dendro
    from dendrosym.cascade.systems.bssn.legacy import bssn

    print("Building BSSN cascade from equation structure...")
    result = build_bssn_cascade(dendro, bssn, verbose=True)
    result.summary()
